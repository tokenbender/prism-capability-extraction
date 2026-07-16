from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "compare_arithmetic_physical_parity.py"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_arithmetic_physical_parity import (  # noqa: E402
    GATE_FAILURE_EXIT_CODE,
    compare_dense_recovery,
    compare_physical_acceptance,
    compare_predictions,
)


def row(
    record_id: str,
    text: str,
    correct: bool,
) -> dict[str, object]:
    return {
        "id": record_id,
        "prediction_text": text,
        "prediction_token_ids": [ord(value) for value in text],
        "exact_numeric_correct": correct,
    }


def test_parity_retention_uses_logical_correct_denominator() -> None:
    logical = [
        row("a", "42", True),
        row("b", "10", True),
        row("c", "wrong", False),
    ]
    physical = [
        row("a", "42", True),
        row("b", "9", False),
        row("c", "11", True),
    ]
    summary, differences = compare_predictions(
        logical,
        physical,
        retention_floor=0.5,
    )
    assert summary["status"] == "pass"
    assert summary["logical_correct"] == 2
    assert summary["physical_correct"] == 2
    assert summary["retained_logical_correct"] == 1
    assert summary["logical_correctness_retention"] == 0.5
    assert summary["logical_only_correct"] == 1
    assert summary["physical_only_correct"] == 1
    assert summary["identical_prediction_rows"] == 1
    assert len(differences) == 2


def test_parity_floor_and_ids_fail_closed() -> None:
    logical = [row("a", "42", True)]
    physical = [row("a", "41", False)]
    summary, _ = compare_predictions(logical, physical)
    assert summary["status"] == "fail"
    with pytest.raises(ValueError, match="ID sets differ"):
        compare_predictions(logical, [row("b", "42", True)])
    with pytest.raises(ValueError, match="zero correct"):
        compare_predictions([row("a", "bad", False)], [row("a", "bad", False)])


def test_physical_acceptance_requires_parity_and_dense_recovery() -> None:
    dense = [
        row(f"d{index}", "ok" if index < 10 else "bad", index < 10)
        for index in range(10)
    ] + [
        row(f"x{index}", "bad", False)
        for index in range(91)
    ]
    logical = [
        row(f"d{index}", "ok" if index < 9 else "bad", index < 9)
        for index in range(10)
    ] + [
        row(f"x{index}", "ok", True)
        for index in range(91)
    ]
    physical = [
        row(
            f"d{index}",
            "ok" if 1 <= index < 9 else "bad",
            1 <= index < 9,
        )
        for index in range(10)
    ] + [
        row(f"x{index}", "ok", True)
        for index in range(91)
    ]

    summary, _ = compare_physical_acceptance(
        logical,
        physical,
        dense,
        retention_floor=0.99,
        recovery_floor=0.90,
    )
    assert summary["parity_status"] == "pass"
    assert summary["logical_correctness_retention"] == 0.99
    assert summary["dense_recovery"]["status"] == "fail"
    assert summary["dense_recovery"]["matched_dense_recovery"] == 0.8
    assert summary["status"] == "fail"


def test_physical_acceptance_passes_both_frozen_gates() -> None:
    dense = [
        row("a", "42", True),
        row("b", "10", True),
        row("c", "bad", False),
    ]
    summary, _ = compare_physical_acceptance(
        dense,
        dense,
        dense,
        retention_floor=0.99,
        recovery_floor=0.90,
    )
    assert summary["status"] == "pass"
    assert summary["parity_status"] == "pass"
    assert summary["dense_recovery"]["status"] == "pass"
    assert summary["gates"]["logical_correctness_retention"]["value"] == 1.0
    assert summary["gates"]["matched_dense_recovery"]["value"] == 1.0


def test_dense_recovery_fails_closed_on_invalid_parent() -> None:
    physical = [row("a", "42", True)]
    with pytest.raises(ValueError, match="ID sets differ"):
        compare_dense_recovery(
            [row("b", "42", True)],
            physical,
        )
    with pytest.raises(ValueError, match="zero correct"):
        compare_dense_recovery(
            [row("a", "bad", False)],
            physical,
        )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))


def test_cli_distinguishes_gate_rejection_from_runtime_failure(
    tmp_path: Path,
) -> None:
    logical_path = tmp_path / "logical.jsonl"
    physical_path = tmp_path / "physical.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output_path = tmp_path / "parity.json"
    differences_path = tmp_path / "differences.jsonl"
    write_jsonl(logical_path, [row("a", "42", True)])
    write_jsonl(physical_path, [row("a", "41", False)])
    write_jsonl(dense_path, [row("a", "42", True)])

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--logical",
        str(logical_path),
        "--physical",
        str(physical_path),
        "--dense",
        str(dense_path),
        "--output",
        str(output_path),
        "--differences",
        str(differences_path),
    ]
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode == GATE_FAILURE_EXIT_CODE
    assert json.loads(output_path.read_text())["status"] == "fail"
    assert differences_path.is_file()

    runtime_failure = subprocess.run(
        [
            *command[: command.index("--dense") + 1],
            str(tmp_path / "missing.jsonl"),
            *command[command.index("--output") :],
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert runtime_failure.returncode not in (0, GATE_FAILURE_EXIT_CODE)
