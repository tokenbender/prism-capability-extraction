from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_bfcl_physical_throughput import (  # noqa: E402
    compare_reports,
    parse_candidate_spec,
    percent_improvement,
)


def stats(median: float, p95: float) -> dict[str, float]:
    return {"median": median, "p95": p95}


def benchmark_report(
    *,
    accepted: tuple[float, float] = (100.0, 120.0),
    elapsed: tuple[float, float] = (20.0, 24.0),
    latency: tuple[float, float] = (50.0, 70.0),
    allocated: tuple[float, float] = (1000.0, 1200.0),
    load_seconds: float = 4.0,
    repack_seconds: float = 2.0,
    compile_seconds: float | None = 3.0,
) -> dict[str, object]:
    report: dict[str, object] = {
        "status": "pass",
        "candidate": {
            "attention_implementation": "sdpa",
            "mlp_implementation": "packed_gate_up",
            "width_alignment": 128,
        },
        "contract": {
            "repeats": 5,
            "examples": 1007,
            "model_hash": "model-sha256",
            "pairs": "bfcl_pairs.jsonl",
        },
        "summary": {
            "accepted_generated_tokens_per_second": stats(*accepted),
            "generated_slots_per_second": stats(200.0, 220.0),
            "examples_per_second": stats(10.0, 12.0),
            "elapsed_seconds": stats(*elapsed),
            "peak_allocated_bytes": stats(*allocated),
            "peak_reserved_bytes": stats(1400.0, 1600.0),
        },
        "phase_summary": {
            "prefill_useful_tokens_per_second": stats(1000.0, 1100.0),
            "decode_tokens_per_second": stats(200.0, 220.0),
        },
        "batch_latency_summary": {
            "elapsed_milliseconds": stats(*latency),
        },
        "load_seconds": load_seconds,
        "warmup_measurements": [
            {"elapsed_seconds": 10.0},
            {"elapsed_seconds": 8.0},
            {"elapsed_seconds": 8.2},
        ],
        "load_receipt": {
            "timings_seconds": {"runtime_repack": repack_seconds},
        },
    }
    if compile_seconds is not None:
        report["compile_wrap_seconds"] = compile_seconds
    return report


def test_comparison_extracts_metrics_and_selects_fastest() -> None:
    baseline = benchmark_report()
    candidate = benchmark_report(
        accepted=(130.0, 144.0),
        elapsed=(16.0, 18.0),
        latency=(40.0, 56.0),
        allocated=(800.0, 900.0),
        load_seconds=3.0,
        repack_seconds=1.0,
        compile_seconds=None,
    )

    result = compare_reports(
        baseline_report=baseline,
        baseline_source=Path("baseline.json"),
        candidates=[("packed", Path("packed.json"), candidate)],
    )

    packed = result["candidates"][0]
    assert packed["setup"] == candidate["candidate"]
    assert packed["metrics"]["accepted_tokens_per_second"] == {
        "unit": "tokens/second",
        "higher_is_better": True,
        "median": 130.0,
        "p95": 144.0,
    }
    assert packed["metrics"]["load_seconds"]["median"] == 3.0
    assert packed["metrics"]["load_seconds"]["p95"] is None
    assert packed["metrics"]["outer_compile_wrap_seconds"] == {
        "unit": "seconds",
        "higher_is_better": False,
        "median": None,
        "p95": None,
    }
    assert packed["metrics"]["generation_first_warmup_seconds"]["median"] == 10.0
    assert packed["metrics"]["generation_steady_warmup_seconds"][
        "median"
    ] == pytest.approx(8.1)
    assert packed["metrics"]["generation_first_use_overhead_seconds"][
        "median"
    ] == pytest.approx(1.9)
    comparison = packed["vs_baseline"]
    assert comparison["accepted_tokens_per_second"] == {
        "median_improvement_percent": pytest.approx(30.0),
        "p95_improvement_percent": pytest.approx(20.0),
    }
    assert comparison["elapsed_seconds"] == {
        "median_improvement_percent": pytest.approx(20.0),
        "p95_improvement_percent": pytest.approx(25.0),
    }
    assert comparison["batch_latency_milliseconds"] == {
        "median_improvement_percent": pytest.approx(20.0),
        "p95_improvement_percent": pytest.approx(20.0),
    }
    assert comparison["peak_allocated_bytes"] == {
        "median_improvement_percent": pytest.approx(20.0),
        "p95_improvement_percent": pytest.approx(25.0),
    }
    assert comparison["runtime_repack_seconds"] == {
        "median_improvement_percent": pytest.approx(50.0),
        "p95_improvement_percent": None,
    }
    assert comparison["outer_compile_wrap_seconds"] == {
        "median_improvement_percent": None,
        "p95_improvement_percent": None,
    }
    assert result["fastest_by_accepted_tokens_per_second"] == {
        "label": "packed",
        "role": "candidate",
        "source": "packed.json",
        "median_accepted_tokens_per_second": 130.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeats", 6),
        ("examples", 1006),
        ("model_hash", "different-model"),
        ("pairs", "different-pairs.jsonl"),
    ],
)
def test_comparison_rejects_contract_mismatch(field: str, value: object) -> None:
    baseline = benchmark_report()
    candidate = benchmark_report()
    candidate["contract"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        compare_reports(
            baseline_report=baseline,
            baseline_source=Path("baseline.json"),
            candidates=[("candidate", Path("candidate.json"), candidate)],
        )


def test_optional_contract_and_metrics_remain_null_when_absent() -> None:
    baseline = benchmark_report()
    candidate = benchmark_report()
    candidate.pop("status")
    candidate["contract"].pop("model_hash")  # type: ignore[union-attr]
    candidate["summary"].pop(  # type: ignore[union-attr]
        "accepted_generated_tokens_per_second"
    )
    candidate.pop("warmup_measurements")

    result = compare_reports(
        baseline_report=baseline,
        baseline_source=Path("baseline.json"),
        candidates=[("partial", Path("partial.json"), candidate)],
    )

    partial = result["candidates"][0]
    assert partial["contract"]["status"] is None
    assert partial["contract"]["model_hash"] is None
    assert partial["metrics"]["accepted_tokens_per_second"]["median"] is None
    assert partial["metrics"]["generation_first_warmup_seconds"]["median"] is None
    assert result["validated_contract"]["model_hash"] == "model-sha256"
    assert result["fastest_by_accepted_tokens_per_second"]["label"] == "baseline"


def test_non_pass_report_is_rejected() -> None:
    candidate = benchmark_report()
    candidate["status"] = "failed"

    with pytest.raises(ValueError, match="non-pass status"):
        compare_reports(
            baseline_report=benchmark_report(),
            baseline_source=Path("baseline.json"),
            candidates=[("failed", Path("failed.json"), candidate)],
        )


def test_candidate_parser_and_improvement_direction() -> None:
    assert parse_candidate_spec("packed=reports/packed.json") == (
        "packed",
        Path("reports/packed.json"),
    )
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_candidate_spec("missing-separator")
    assert percent_improvement(10.0, 12.0, higher_is_better=True) == pytest.approx(
        20.0
    )
    assert percent_improvement(10.0, 8.0, higher_is_better=False) == pytest.approx(
        20.0
    )
    assert percent_improvement(0.0, 1.0, higher_is_better=True) is None


def test_compiled_decode_slots_take_precedence_over_manual_control() -> None:
    baseline = benchmark_report()
    candidate = benchmark_report()
    candidate["compiled_decode_summary"] = {
        "input_slots_per_second": stats(900.0, 950.0)
    }

    result = compare_reports(
        baseline_report=baseline,
        baseline_source=Path("baseline.json"),
        candidates=[("static", Path("static.json"), candidate)],
    )

    assert result["candidates"][0]["metrics"]["decode_slots_per_second"] == {
        "unit": "slots/second",
        "higher_is_better": True,
        "median": 900.0,
        "p95": 950.0,
    }
