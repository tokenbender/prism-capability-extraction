from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_arithmetic_physical_throughput import (  # noqa: E402
    compare_attempts,
    load_attempts,
    percent_improvement,
)


def stats(median: float, p95: float) -> dict[str, float]:
    return {
        "mean": median,
        "median": median,
        "p50": median,
        "p95": p95,
        "min": median,
        "max": p95,
        "stdev": 0.0,
    }


def attempt(
    name: str,
    role: str,
    *,
    batch_size: int = 16,
    correct_eps: tuple[float, float] = (10.0, 12.0),
    generated_tps: tuple[float, float] = (25.0, 30.0),
    generated_slot_tps: tuple[float, float] = (30.0, 35.0),
    latency: tuple[float, float] = (50.0, 70.0),
    ttft: tuple[float, float] = (5.0, 7.0),
    quality_pass: bool = True,
    status: str = "pass",
) -> dict[str, object]:
    contract = {
        "workload_sha256": "workload-sha",
        "examples": 1500,
        "prompt_contract": "generation_prompt_if_present_else_prompt_exact",
        "quality_metric": "strict_first_line_integer_v1",
        "isolation": "standalone_model_no_counterfactual_donor",
        "generation_mode": "fixed_cap_greedy_unguided",
        "max_new_tokens": 8,
        "generated_tokens_contract": (
            "non_stop_token_ids_after_eos_and_pad_trimming"
        ),
        "throughput_timing_contract": (
            "synchronized_pretokenized_generate_only_decode_scoring_excluded"
        ),
        "batch_size": batch_size,
    }
    candidate = {
        "name": name,
        "role": role,
        "model_kind": "dense" if role == "dense_parent" else "physical",
    }
    if status != "pass":
        return {
            "status": status,
            "candidate": candidate,
            "batch_size": batch_size,
            "contract": contract,
            "failure_stage": "candidate_load",
            "error": {"type": "RuntimeError", "message": "unavailable"},
            "quality_gate": None,
        }
    accuracy = 0.91 if quality_pass else 0.2
    return {
        "status": "pass",
        "candidate": candidate,
        "batch_size": batch_size,
        "contract": contract,
        "quality_gate": {
            "pass": quality_pass,
            "accuracy": accuracy,
            "correct": int(accuracy * 1500),
            "examples": 1500,
        },
        "summary": {
            "correct_examples_per_second": stats(*correct_eps),
            "generated_tokens_per_second": stats(*generated_tps),
            "generated_token_slots_per_second": stats(
                *generated_slot_tps
            ),
            "examples_per_second": stats(11.0, 13.0),
            "elapsed_seconds": stats(150.0, 160.0),
            "accuracy": stats(accuracy, accuracy),
            "peak_allocated_bytes": stats(1000.0, 1200.0),
            "peak_reserved_bytes": stats(1400.0, 1600.0),
        },
        "batch_latency_summary": {
            "batch_milliseconds": stats(*latency),
            "per_example_milliseconds": stats(
                latency[0] / batch_size,
                latency[1] / batch_size,
            ),
        },
        "ttft": {
            "status": "pass",
            "summary": {
                "batch_milliseconds": stats(*ttft),
                "per_example_milliseconds": stats(
                    ttft[0] / batch_size,
                    ttft[1] / batch_size,
                ),
            },
        },
        "setup_summary": {
            "load_seconds": 4.0,
            "outer_compile_wrap_seconds": 2.0,
            "after_load_allocated_bytes": 800.0,
            "after_load_reserved_bytes": 900.0,
        },
    }


def test_comparison_preserves_failures_gates_quality_and_selects_winner() -> None:
    dense = attempt(
        "dense",
        "dense_parent",
        correct_eps=(20.0, 22.0),
        generated_tps=(45.0, 48.0),
    )
    canonical = attempt(
        "canonical",
        "canonical_physical",
        correct_eps=(10.0, 12.0),
        generated_tps=(25.0, 28.0),
    )
    winner = attempt(
        "packed",
        "optimized_physical",
        correct_eps=(30.0, 34.0),
        generated_tps=(70.0, 76.0),
        latency=(30.0, 42.0),
    )
    quality_failed = attempt(
        "wrong-fast",
        "optimized_physical",
        correct_eps=(1000.0, 1200.0),
        quality_pass=False,
    )
    unsupported = attempt(
        "triton-missing",
        "optimized_physical",
        status="unsupported",
    )
    loaded = [
        (Path("dense.json"), dense),
        (Path("canonical.json"), canonical),
        (Path("winner.json"), winner),
        (Path("wrong.json"), quality_failed),
        (Path("unsupported.jsonl"), unsupported),
    ]

    result = compare_attempts(loaded)

    assert result["fastest_quality_passing"]["name"] == "packed"
    assert result["fastest_optimized_physical"]["name"] == "packed"
    assert result["quality_gate_failures"] == 1
    assert result["attempt_status_counts"] == {
        "pass": 4,
        "failed": 0,
        "unsupported": 1,
    }
    normalized_winner = next(
        row for row in result["attempts"] if row["name"] == "packed"
    )
    assert normalized_winner["versus_dense_parent"]["metrics"][
        "correct_examples_per_second"
    ]["median_improvement_percent"] == pytest.approx(50.0)
    assert normalized_winner["versus_canonical_physical"]["metrics"][
        "correct_examples_per_second"
    ]["median_improvement_percent"] == pytest.approx(200.0)
    normalized_unsupported = next(
        row for row in result["attempts"] if row["name"] == "triton-missing"
    )
    assert not normalized_unsupported["eligible"]
    assert normalized_unsupported["error"]["message"] == "unavailable"


def test_winners_are_selected_separately_for_each_batch_size() -> None:
    rows = [
        (
            Path("b8.json"),
            attempt(
                "packed",
                "optimized_physical",
                batch_size=8,
                correct_eps=(15.0, 16.0),
            ),
        ),
        (
            Path("b32.json"),
            attempt(
                "packed",
                "optimized_physical",
                batch_size=32,
                correct_eps=(35.0, 37.0),
            ),
        ),
    ]

    result = compare_attempts(rows)

    assert result["winners_by_batch_size"]["8"]["batch_size"] == 8
    assert result["winners_by_batch_size"]["32"]["batch_size"] == 32
    assert result["fastest_quality_passing"]["batch_size"] == 32


def test_comparison_rejects_workload_contract_mismatch() -> None:
    first = attempt("dense", "dense_parent")
    second = attempt("packed", "optimized_physical")
    second["contract"]["workload_sha256"] = "different"  # type: ignore[index]

    with pytest.raises(ValueError, match="workload_sha256"):
        compare_attempts(
            [(Path("one.json"), first), (Path("two.json"), second)]
        )


def test_json_and_jsonl_receipts_load_without_dropping_failures(
    tmp_path: Path,
) -> None:
    passed = attempt("canonical", "canonical_physical")
    failed = attempt(
        "oom",
        "optimized_physical",
        status="failed",
    )
    json_path = tmp_path / "summary.json"
    json_path.write_text(json.dumps({"attempts": [passed, failed]}))
    jsonl_path = tmp_path / "attempts.jsonl"
    jsonl_path.write_text(
        json.dumps(passed) + "\n" + json.dumps(failed) + "\n"
    )

    assert [row["status"] for row in load_attempts(json_path)] == [
        "pass",
        "failed",
    ]
    assert [row["status"] for row in load_attempts(jsonl_path)] == [
        "pass",
        "failed",
    ]


def test_improvement_direction_is_consistent() -> None:
    assert percent_improvement(
        10.0,
        12.0,
        higher_is_better=True,
    ) == pytest.approx(20.0)
    assert percent_improvement(
        10.0,
        8.0,
        higher_is_better=False,
    ) == pytest.approx(20.0)
    assert percent_improvement(
        0.0,
        1.0,
        higher_is_better=True,
    ) is None
