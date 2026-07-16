from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_arithmetic_physical_parity import compare_predictions  # noqa: E402


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
