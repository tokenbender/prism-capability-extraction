#!/usr/bin/env python3
"""Compare one BFCL physical-throughput baseline with named candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Union


Number = Union[int, float]
PathParts = tuple[str, ...]


CONTRACT_PATHS: dict[str, tuple[PathParts, ...]] = {
    "repeats": (("contract", "repeats"), ("repeats",)),
    "examples": (("contract", "examples"), ("examples",)),
    "model_hash": (
        ("contract", "model_hash"),
        ("contract", "model_sha256"),
        ("model_hash",),
        ("model_sha256",),
        ("load_receipt", "model_hash"),
        ("load_receipt", "model_sha256"),
    ),
    "pairs": (("contract", "pairs"), ("pairs",)),
}


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "accepted_tokens_per_second": {
        "unit": "tokens/second",
        "higher_is_better": True,
        "paths": (
            ("summary", "accepted_generated_tokens_per_second"),
            ("accepted_generated_tokens_per_second",),
        ),
    },
    "generated_slots_per_second": {
        "unit": "slots/second",
        "higher_is_better": True,
        "paths": (
            ("summary", "generated_slots_per_second"),
            ("generated_slots_per_second",),
        ),
    },
    "examples_per_second": {
        "unit": "examples/second",
        "higher_is_better": True,
        "paths": (
            ("summary", "examples_per_second"),
            ("examples_per_second",),
        ),
    },
    "elapsed_seconds": {
        "unit": "seconds",
        "higher_is_better": False,
        "paths": (("summary", "elapsed_seconds"), ("elapsed_seconds",)),
    },
    "batch_latency_milliseconds": {
        "unit": "milliseconds",
        "higher_is_better": False,
        "paths": (
            ("batch_latency_summary", "elapsed_milliseconds"),
            ("summary", "batch_latency_milliseconds"),
            ("summary", "latency_milliseconds"),
            ("batch_latency_milliseconds",),
            ("latency_milliseconds",),
        ),
    },
    "prefill_tokens_per_second": {
        "unit": "tokens/second",
        "higher_is_better": True,
        "paths": (
            ("phase_summary", "prefill_useful_tokens_per_second"),
            ("phase_summary", "prefill_tokens_per_second"),
            ("summary", "prefill_tokens_per_second"),
            ("prefill_tokens_per_second",),
        ),
    },
    "decode_tokens_per_second": {
        "unit": "tokens/second",
        "higher_is_better": True,
        "paths": (
            ("phase_summary", "decode_tokens_per_second"),
            ("summary", "decode_tokens_per_second"),
            ("decode_tokens_per_second",),
        ),
    },
    "peak_allocated_bytes": {
        "unit": "bytes",
        "higher_is_better": False,
        "paths": (
            ("summary", "peak_allocated_bytes"),
            ("peak_allocated_bytes",),
        ),
    },
    "peak_reserved_bytes": {
        "unit": "bytes",
        "higher_is_better": False,
        "paths": (
            ("summary", "peak_reserved_bytes"),
            ("peak_reserved_bytes",),
        ),
    },
    "load_seconds": {
        "unit": "seconds",
        "higher_is_better": False,
        "paths": (
            ("setup_summary", "load_seconds"),
            ("summary", "load_seconds"),
            ("load_seconds",),
        ),
    },
    "runtime_repack_seconds": {
        "unit": "seconds",
        "higher_is_better": False,
        "paths": (
            ("setup_summary", "runtime_repack_seconds"),
            ("summary", "runtime_repack_seconds"),
            ("load_receipt", "timings_seconds", "runtime_repack"),
            ("runtime_repack_seconds",),
            ("repack_seconds",),
        ),
    },
    "compile_seconds": {
        "unit": "seconds",
        "higher_is_better": False,
        "paths": (
            ("setup_summary", "compile_seconds"),
            ("summary", "compile_seconds"),
            ("compile_wrap_seconds",),
            ("compile_seconds",),
        ),
    },
}


_MISSING = object()


def _at_path(value: Any, path: PathParts) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _as_number(value: Any, *, location: str) -> Number | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number or null")
    if not math.isfinite(float(value)):
        raise ValueError(f"{location} must be finite")
    return value


def _metric_stats(
    report: dict[str, Any],
    *,
    metric: str,
    paths: tuple[PathParts, ...],
) -> dict[str, Number | None]:
    """Return explicit median/p95 values without deriving missing statistics.

    A scalar setup timing is a single observation, so it is retained as the
    median observation while p95 remains null.
    """

    for path in paths:
        value = _at_path(report, path)
        if value is _MISSING or value is None:
            continue
        location = ".".join(path)
        if isinstance(value, dict):
            median_value = value.get("median", value.get("p50"))
            p95_value = value.get("p95")
            return {
                "median": _as_number(
                    median_value,
                    location=f"{metric} at {location}.median",
                ),
                "p95": _as_number(
                    p95_value,
                    location=f"{metric} at {location}.p95",
                ),
            }
        return {
            "median": _as_number(value, location=f"{metric} at {location}"),
            "p95": None,
        }
    return {"median": None, "p95": None}


def _contract_value(
    report: dict[str, Any],
    *,
    field: str,
    paths: tuple[PathParts, ...],
) -> Any:
    found: list[tuple[str, Any]] = []
    for path in paths:
        value = _at_path(report, path)
        if value is not _MISSING and value is not None:
            found.append((".".join(path), value))
    if not found:
        return None
    first_location, first_value = found[0]
    conflicts = {
        location: value for location, value in found[1:] if value != first_value
    }
    if conflicts:
        raise ValueError(
            f"conflicting {field} values inside one report: "
            f"{first_location}={first_value!r}, conflicts={conflicts!r}"
        )
    return first_value


def _report_contract(report: dict[str, Any], *, label: str) -> dict[str, Any]:
    status = report.get("status")
    if status is not None and status != "pass":
        raise ValueError(f"report {label!r} has non-pass status {status!r}")

    contract = {"status": status}
    for field, paths in CONTRACT_PATHS.items():
        contract[field] = _contract_value(report, field=field, paths=paths)

    for field in ("repeats", "examples"):
        value = contract[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(
                f"report {label!r} contract field {field!r} must be a positive integer"
            )
    model_hash = contract["model_hash"]
    if model_hash is not None and (
        not isinstance(model_hash, str) or not model_hash.strip()
    ):
        raise ValueError(
            f"report {label!r} contract field 'model_hash' must be a nonempty string"
        )
    pairs = contract["pairs"]
    if pairs is not None and isinstance(pairs, str) and not pairs.strip():
        raise ValueError(
            f"report {label!r} contract field 'pairs' must not be empty"
        )
    return contract


def _report_summary(
    *,
    label: str,
    source: Path,
    role: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    setup = report.get("candidate")
    if setup is not None and not isinstance(setup, dict):
        raise ValueError(f"report {label!r} candidate setup must be an object or null")

    metrics: dict[str, Any] = {}
    for metric, spec in METRIC_SPECS.items():
        metrics[metric] = {
            "unit": spec["unit"],
            "higher_is_better": spec["higher_is_better"],
            **_metric_stats(report, metric=metric, paths=spec["paths"]),
        }
    return {
        "label": label,
        "role": role,
        "source": str(source),
        "contract": _report_contract(report, label=label),
        "setup": setup,
        "metrics": metrics,
    }


def _validate_shared_contract(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    consensus: dict[str, Any] = {}
    for field in ("status", *CONTRACT_PATHS):
        present = [
            (report["label"], report["contract"][field])
            for report in reports
            if report["contract"][field] is not None
        ]
        if not present:
            consensus[field] = None
            continue
        reference_label, reference_value = present[0]
        mismatches = {
            label: value for label, value in present[1:] if value != reference_value
        }
        if mismatches:
            raise ValueError(
                f"benchmark contract field {field!r} differs: "
                f"{reference_label}={reference_value!r}, mismatches={mismatches!r}"
            )
        consensus[field] = reference_value
    return consensus


def percent_improvement(
    baseline: Number | None,
    candidate: Number | None,
    *,
    higher_is_better: bool,
) -> float | None:
    """Return signed improvement, where positive always means better."""

    if baseline is None or candidate is None or baseline == 0:
        return None
    ratio = float(candidate) / float(baseline)
    return 100.0 * (ratio - 1.0 if higher_is_better else 1.0 - ratio)


def _versus_baseline(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric, spec in METRIC_SPECS.items():
        baseline_stats = baseline["metrics"][metric]
        candidate_stats = candidate["metrics"][metric]
        comparison[metric] = {
            "median_improvement_percent": percent_improvement(
                baseline_stats["median"],
                candidate_stats["median"],
                higher_is_better=spec["higher_is_better"],
            ),
            "p95_improvement_percent": percent_improvement(
                baseline_stats["p95"],
                candidate_stats["p95"],
                higher_is_better=spec["higher_is_better"],
            ),
        }
    return comparison


def compare_reports(
    *,
    baseline_report: dict[str, Any],
    baseline_source: Path,
    candidates: list[tuple[str, Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Validate, normalize, and compare already-loaded benchmark reports."""

    if not candidates:
        raise ValueError("at least one candidate report is required")
    labels = [label for label, _, _ in candidates]
    if any(not label.strip() for label in labels):
        raise ValueError("candidate labels must not be empty")
    if "baseline" in labels:
        raise ValueError("candidate label 'baseline' is reserved")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")

    baseline = _report_summary(
        label="baseline",
        source=baseline_source,
        role="baseline",
        report=baseline_report,
    )
    normalized_candidates = [
        _report_summary(label=label, source=source, role="candidate", report=report)
        for label, source, report in candidates
    ]
    all_reports = [baseline, *normalized_candidates]
    validated_contract = _validate_shared_contract(all_reports)

    for candidate in normalized_candidates:
        candidate["vs_baseline"] = _versus_baseline(baseline, candidate)

    accepted_metric = "accepted_tokens_per_second"
    eligible = [
        report
        for report in all_reports
        if report["metrics"][accepted_metric]["median"] is not None
    ]
    fastest = None
    if eligible:
        winner = max(
            eligible,
            key=lambda report: report["metrics"][accepted_metric]["median"],
        )
        fastest = {
            "label": winner["label"],
            "role": winner["role"],
            "source": winner["source"],
            "median_accepted_tokens_per_second": winner["metrics"][accepted_metric][
                "median"
            ],
        }

    return {
        "status": "pass",
        "validated_contract": validated_contract,
        "baseline": baseline,
        "candidates": normalized_candidates,
        "fastest_by_accepted_tokens_per_second": fastest,
    }


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError(f"benchmark report {path} must contain a JSON object")
    return report


def parse_candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("candidate must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise ValueError("candidate label and path must both be nonempty")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="named candidate report; repeat for multiple candidates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        candidate_specs = [parse_candidate_spec(value) for value in args.candidate]
        comparison = compare_reports(
            baseline_report=load_report(args.baseline),
            baseline_source=args.baseline,
            candidates=[
                (label, path, load_report(path)) for label, path in candidate_specs
            ],
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
