from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_bfcl_silu_mul import (  # noqa: E402
    SUPPORTED_FORMAT,
    accuracy_summary,
    aggregate_results,
    align_width,
    percentile,
    plan_shapes,
    read_physical_widths,
    safe_ratio,
    summarize,
)


def write_metadata(bundle: Path, isolation: dict[str, object]) -> None:
    bundle.mkdir()
    (bundle / "substrate_metadata.json").write_text(
        json.dumps({"format": SUPPORTED_FORMAT, "isolation": isolation})
    )


def test_read_physical_widths_validates_layer_and_total_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write_metadata(
        bundle,
        {
            "layers": 3,
            "kept_total": 25,
            "kept_per_layer": {"0": 3, "1": 5, "2": 17},
        },
    )

    assert read_physical_widths(bundle) == [3, 5, 17]

    metadata_path = bundle / "substrate_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["isolation"]["kept_total"] = 24
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="kept_total mismatch"):
        read_physical_widths(bundle)

    metadata["isolation"]["kept_total"] = 25.0
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="kept_total must be an integer"):
        read_physical_widths(bundle)


def test_read_physical_widths_rejects_noncontiguous_layers(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    write_metadata(
        bundle,
        {
            "layers": 2,
            "kept_total": 8,
            "kept_per_layer": {"0": 3, "2": 5},
        },
    )

    with pytest.raises(ValueError, match="cover every layer exactly"):
        read_physical_widths(bundle)


def test_alignment_and_memory_bound_shape_planning() -> None:
    assert align_width(3, 16) == 16
    assert align_width(16, 16) == 16
    assert align_width(17, 16) == 32

    plans = plan_shapes(
        [3, 5, 17],
        [1, 10],
        alignment=16,
        max_elements=1500,
        triton_enabled=True,
    )

    assert plans[0] == {
        "rows": 1,
        "aligned_width": 16,
        "layer_indices": [0, 1],
        "active_widths": [3, 5],
        "input_elements": 32,
        "output_elements": 16,
        "estimated_peak_live_elements": 112,
        "max_elements": 1500,
        "status": "planned",
        "skip_reason": None,
    }
    assert plans[-1]["rows"] == 10
    assert plans[-1]["aligned_width"] == 32
    assert plans[-1]["estimated_peak_live_elements"] == 2240
    assert plans[-1]["status"] == "skipped"
    assert plans[-1]["skip_reason"] == (
        "estimated_peak_live_elements_exceed_max"
    )


def test_statistics_helpers_record_median_and_tail() -> None:
    values = [1.0, 2.0, 3.0, 10.0]

    assert percentile(values, 0.95) == pytest.approx(8.95)
    assert summarize(values) == {
        "count": 4,
        "mean": 4.0,
        "median": 2.5,
        "p95": pytest.approx(8.95),
        "min": 1.0,
        "max": 10.0,
    }


def test_accuracy_and_speedup_helpers_are_explicit() -> None:
    assert accuracy_summary(
        total_elements=8,
        exact_elements=6,
        max_abs_error=0.25,
        mean_abs_error=0.03125,
    ) == {
        "total_elements": 8,
        "exact_elements": 6,
        "exact_element_fraction": 0.75,
        "finite_error_elements": 8,
        "all_errors_finite": True,
        "max_abs_error": 0.25,
        "mean_abs_error": 0.03125,
    }
    assert safe_ratio(2.0, 1.0) == 2.0
    assert safe_ratio(2.0, 0.0) is None
    with pytest.raises(ValueError, match="exact_elements"):
        accuracy_summary(
            total_elements=8,
            exact_elements=9,
            max_abs_error=0.0,
            mean_abs_error=0.0,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        accuracy_summary(
            total_elements=8,
            exact_elements=8,
            max_abs_error=float("nan"),
            mean_abs_error=0.0,
        )


def test_aggregate_results_separates_measured_and_skipped_cases() -> None:
    results = [
        {
            "status": "measured",
            "torch": {"latency_milliseconds": {"median": 2.0}},
            "triton": {
                "status": "measured",
                "latency_milliseconds": {"median": 1.0},
            },
            "speedup": {"median": 2.0},
            "accuracy": {
                "total_elements": 4,
                "exact_elements": 3,
                "exact_element_fraction": 0.75,
                "finite_error_elements": 4,
                "all_errors_finite": True,
                "max_abs_error": 0.1,
                "mean_abs_error": 0.025,
            },
        },
        {
            "status": "torch_only",
            "torch": {"latency_milliseconds": {"median": 4.0}},
            "triton": None,
            "speedup": None,
            "accuracy": None,
        },
        {
            "status": "skipped",
            "skip_reason": "estimated_peak_live_elements_exceed_max",
            "torch": None,
            "triton": None,
            "speedup": None,
            "accuracy": None,
        },
    ]

    aggregate = aggregate_results(results)

    assert aggregate["planned_shape_dtype_cases"] == 3
    assert aggregate["torch_measured_cases"] == 2
    assert aggregate["triton_measured_cases"] == 1
    assert aggregate["skipped_cases"] == 1
    assert aggregate["skip_reasons"] == {
        "estimated_peak_live_elements_exceed_max": 1
    }
    assert aggregate["torch_shape_median_latency_milliseconds"]["median"] == 3.0
    assert aggregate["triton_speedup_across_shapes"]["median"] == 2.0
    assert aggregate["accuracy"]["exact_element_fraction"] == 0.75


def test_aggregate_results_does_not_hide_nonfinite_errors() -> None:
    aggregate = aggregate_results(
        [
            {
                "status": "measured",
                "torch": {"latency_milliseconds": {"median": 2.0}},
                "triton": {
                    "status": "measured",
                    "latency_milliseconds": {"median": 1.0},
                },
                "speedup": {"median": 2.0},
                "accuracy": {
                    "total_elements": 4,
                    "exact_elements": 2,
                    "exact_element_fraction": 0.5,
                    "finite_error_elements": 3,
                    "all_errors_finite": False,
                    "max_abs_error": None,
                    "mean_abs_error": None,
                },
            }
        ]
    )

    assert aggregate["accuracy"] == {
        "total_elements": 4,
        "exact_elements": 2,
        "exact_element_fraction": 0.5,
        "finite_error_elements": 3,
        "all_errors_finite": False,
        "max_abs_error": None,
        "mean_abs_error": None,
    }
