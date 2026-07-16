#!/usr/bin/env python3
"""Compare quality-gated arithmetic throughput attempts across runtime roles.

Inputs may be benchmark summary JSON files, raw attempts JSONL files, JSON
lists, or individual attempt objects.  Failed, unsupported, and quality-failed
attempts remain in the output; only successful quality-passing attempts are
eligible to win.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


CONTRACT_FIELDS = (
    "workload_sha256",
    "examples",
    "prompt_contract",
    "quality_metric",
    "isolation",
    "generation_mode",
    "max_new_tokens",
    "generated_tokens_contract",
    "throughput_timing_contract",
)

METRICS: dict[str, dict[str, Any]] = {
    "correct_examples_per_second": {
        "path": ("summary", "correct_examples_per_second"),
        "higher_is_better": True,
        "unit": "correct examples/second",
    },
    "generated_tokens_per_second": {
        "path": ("summary", "generated_tokens_per_second"),
        "higher_is_better": True,
        "unit": "tokens/second",
    },
    "generated_token_slots_per_second": {
        "path": ("summary", "generated_token_slots_per_second"),
        "higher_is_better": True,
        "unit": "token slots/second",
    },
    "examples_per_second": {
        "path": ("summary", "examples_per_second"),
        "higher_is_better": True,
        "unit": "examples/second",
    },
    "elapsed_seconds": {
        "path": ("summary", "elapsed_seconds"),
        "higher_is_better": False,
        "unit": "seconds",
    },
    "accuracy": {
        "path": ("summary", "accuracy"),
        "higher_is_better": True,
        "unit": "fraction",
    },
    "batch_latency_milliseconds": {
        "path": ("batch_latency_summary", "batch_milliseconds"),
        "higher_is_better": False,
        "unit": "milliseconds",
    },
    "per_example_latency_milliseconds": {
        "path": ("batch_latency_summary", "per_example_milliseconds"),
        "higher_is_better": False,
        "unit": "milliseconds",
    },
    "ttft_milliseconds": {
        "path": ("ttft", "summary", "batch_milliseconds"),
        "higher_is_better": False,
        "unit": "milliseconds",
    },
    "peak_allocated_bytes": {
        "path": ("summary", "peak_allocated_bytes"),
        "higher_is_better": False,
        "unit": "bytes",
    },
    "peak_reserved_bytes": {
        "path": ("summary", "peak_reserved_bytes"),
        "higher_is_better": False,
        "unit": "bytes",
    },
    "after_load_allocated_bytes": {
        "path": ("setup_summary", "after_load_allocated_bytes"),
        "higher_is_better": False,
        "unit": "bytes",
        "scalar": True,
    },
    "after_load_reserved_bytes": {
        "path": ("setup_summary", "after_load_reserved_bytes"),
        "higher_is_better": False,
        "unit": "bytes",
        "scalar": True,
    },
    "load_seconds": {
        "path": ("setup_summary", "load_seconds"),
        "higher_is_better": False,
        "unit": "seconds",
        "scalar": True,
    },
    "outer_compile_wrap_seconds": {
        "path": ("setup_summary", "outer_compile_wrap_seconds"),
        "higher_is_better": False,
        "unit": "seconds",
        "scalar": True,
    },
}


_MISSING = object()


def _at_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _finite_number(value: Any, location: str) -> float | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number or null")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{location} must be finite")
    return normalized


def _metric_stats(
    attempt: dict[str, Any],
    *,
    metric: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    raw = _at_path(attempt, spec["path"])
    if raw is _MISSING or raw is None:
        median = p50 = p95 = None
    elif spec.get("scalar"):
        median = _finite_number(raw, metric)
        p50 = median
        p95 = None
    elif isinstance(raw, dict):
        median = _finite_number(
            raw.get("median", raw.get("p50")),
            f"{metric}.median",
        )
        p50 = _finite_number(raw.get("p50", median), f"{metric}.p50")
        p95 = _finite_number(raw.get("p95"), f"{metric}.p95")
    else:
        median = _finite_number(raw, metric)
        p50 = median
        p95 = None
    return {
        "unit": spec["unit"],
        "higher_is_better": spec["higher_is_better"],
        "median": median,
        "p50": p50,
        "p95": p95,
    }


def load_attempts(path: Path) -> list[dict[str, Any]]:
    """Load benchmark attempts from either durable receipt format."""

    if path.suffix.lower() == ".jsonl":
        attempts = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            attempts = payload
        elif isinstance(payload, dict) and isinstance(payload.get("attempts"), list):
            attempts = payload["attempts"]
        elif isinstance(payload, dict) and "candidate" in payload:
            attempts = [payload]
        else:
            raise ValueError(
                f"{path} must be a benchmark summary, attempt list, or attempt"
            )
    if not all(isinstance(attempt, dict) for attempt in attempts):
        raise ValueError(f"{path} contains a non-object attempt")
    return attempts


def validate_shared_contract(
    attempts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not attempts:
        raise ValueError("no benchmark attempts were supplied")
    consensus: dict[str, Any] = {}
    for field in CONTRACT_FIELDS:
        values = []
        for index, attempt in enumerate(attempts):
            contract = attempt.get("contract")
            if not isinstance(contract, dict):
                raise ValueError(f"attempt {index} has no contract object")
            if contract.get(field) is not None:
                values.append((index, contract[field]))
        if not values:
            raise ValueError(f"benchmark contract omits required field {field!r}")
        reference_index, reference_value = values[0]
        mismatches = {
            index: value for index, value in values[1:] if value != reference_value
        }
        if mismatches:
            raise ValueError(
                f"benchmark contract field {field!r} differs: "
                f"attempt {reference_index}={reference_value!r}, "
                f"mismatches={mismatches!r}"
            )
        consensus[field] = reference_value
    return consensus


def percent_improvement(
    baseline: float | None,
    candidate: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    """Return signed improvement, where positive always means better."""

    if baseline is None or candidate is None or baseline == 0.0:
        return None
    ratio = candidate / baseline
    return 100.0 * (ratio - 1.0 if higher_is_better else 1.0 - ratio)


def _eligible(normalized: dict[str, Any]) -> bool:
    return bool(
        normalized["status"] == "pass"
        and normalized["quality_gate_pass"] is True
        and normalized["metrics"]["correct_examples_per_second"]["median"]
        is not None
    )


def normalize_attempt(
    attempt: dict[str, Any],
    *,
    source: Path,
    source_index: int,
) -> dict[str, Any]:
    candidate = attempt.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{source} attempt {source_index} has no candidate object")
    name = candidate.get("name")
    role = candidate.get("role")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source} attempt {source_index} has no candidate name")
    if role not in {
        "dense_parent",
        "canonical_physical",
        "optimized_physical",
    }:
        raise ValueError(
            f"{source} attempt {source_index} has unsupported role {role!r}"
        )
    batch_size = attempt.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(
            f"{source} attempt {source_index} has invalid batch size"
        )
    status = attempt.get("status")
    if status not in {"pass", "failed", "unsupported"}:
        raise ValueError(
            f"{source} attempt {source_index} has invalid status {status!r}"
        )
    gate = attempt.get("quality_gate")
    gate_pass = gate.get("pass") if isinstance(gate, dict) else None
    normalized = {
        "source": str(source),
        "source_index": source_index,
        "name": name,
        "role": role,
        "batch_size": batch_size,
        "status": status,
        "quality_gate_pass": gate_pass,
        "quality_gate": gate,
        "candidate": candidate,
        "contract": attempt.get("contract"),
        "setup_summary": attempt.get("setup_summary"),
        "error": attempt.get("error"),
        "failure_stage": attempt.get("failure_stage"),
        "metrics": {
            metric: _metric_stats(attempt, metric=metric, spec=spec)
            for metric, spec in METRICS.items()
        },
    }
    normalized["eligible"] = _eligible(normalized)
    return normalized


def compare_metrics(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    result = {
        "baseline_name": baseline["name"],
        "baseline_role": baseline["role"],
        "baseline_batch_size": baseline["batch_size"],
        "metrics": {},
    }
    for metric, spec in METRICS.items():
        baseline_stats = baseline["metrics"][metric]
        candidate_stats = candidate["metrics"][metric]
        result["metrics"][metric] = {
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
    return result


def _best_for_role_and_batch(
    attempts: Sequence[dict[str, Any]],
    *,
    role: str,
    batch_size: int,
) -> dict[str, Any] | None:
    eligible = [
        attempt
        for attempt in attempts
        if attempt["role"] == role
        and attempt["batch_size"] == batch_size
        and attempt["eligible"]
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda attempt: attempt["metrics"][
            "correct_examples_per_second"
        ]["median"],
    )


def _winner_summary(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    if attempt is None:
        return None
    return {
        "name": attempt["name"],
        "role": attempt["role"],
        "batch_size": attempt["batch_size"],
        "source": attempt["source"],
        "source_index": attempt["source_index"],
        "median_correct_examples_per_second": attempt["metrics"][
            "correct_examples_per_second"
        ]["median"],
        "median_generated_tokens_per_second": attempt["metrics"][
            "generated_tokens_per_second"
        ]["median"],
        "median_generated_token_slots_per_second": attempt["metrics"][
            "generated_token_slots_per_second"
        ]["median"],
        "accuracy": attempt["metrics"]["accuracy"]["median"],
        "batch_latency_p50_milliseconds": attempt["metrics"][
            "batch_latency_milliseconds"
        ]["p50"],
        "batch_latency_p95_milliseconds": attempt["metrics"][
            "batch_latency_milliseconds"
        ]["p95"],
        "ttft_p50_milliseconds": attempt["metrics"]["ttft_milliseconds"]["p50"],
        "ttft_p95_milliseconds": attempt["metrics"]["ttft_milliseconds"]["p95"],
        "peak_allocated_bytes": attempt["metrics"]["peak_allocated_bytes"][
            "median"
        ],
    }


def compare_attempts(
    loaded: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    raw_attempts = [attempt for _, attempt in loaded]
    contract = validate_shared_contract(raw_attempts)
    normalized = [
        normalize_attempt(attempt, source=source, source_index=index)
        for index, (source, attempt) in enumerate(loaded)
    ]

    for attempt in normalized:
        dense = _best_for_role_and_batch(
            normalized,
            role="dense_parent",
            batch_size=attempt["batch_size"],
        )
        canonical = _best_for_role_and_batch(
            normalized,
            role="canonical_physical",
            batch_size=attempt["batch_size"],
        )
        attempt["versus_dense_parent"] = compare_metrics(dense, attempt)
        attempt["versus_canonical_physical"] = compare_metrics(
            canonical, attempt
        )

    eligible = [attempt for attempt in normalized if attempt["eligible"]]
    fastest = (
        max(
            eligible,
            key=lambda attempt: attempt["metrics"][
                "correct_examples_per_second"
            ]["median"],
        )
        if eligible
        else None
    )
    optimized = [
        attempt
        for attempt in eligible
        if attempt["role"] == "optimized_physical"
    ]
    fastest_optimized = (
        max(
            optimized,
            key=lambda attempt: attempt["metrics"][
                "correct_examples_per_second"
            ]["median"],
        )
        if optimized
        else None
    )
    winners_by_batch = {}
    for batch_size in sorted({attempt["batch_size"] for attempt in normalized}):
        batch_attempts = [
            attempt
            for attempt in eligible
            if attempt["batch_size"] == batch_size
        ]
        winner = (
            max(
                batch_attempts,
                key=lambda attempt: attempt["metrics"][
                    "correct_examples_per_second"
                ]["median"],
            )
            if batch_attempts
            else None
        )
        winners_by_batch[str(batch_size)] = _winner_summary(winner)

    return {
        "status": "pass" if eligible else "no_quality_passing_attempt",
        "validated_contract": contract,
        "attempts": normalized,
        "attempt_status_counts": {
            status: sum(attempt["status"] == status for attempt in normalized)
            for status in ("pass", "failed", "unsupported")
        },
        "quality_gate_failures": sum(
            attempt["status"] == "pass"
            and attempt["quality_gate_pass"] is False
            for attempt in normalized
        ),
        "fastest_quality_passing": _winner_summary(fastest),
        "fastest_optimized_physical": _winner_summary(fastest_optimized),
        "winners_by_batch_size": winners_by_batch,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        loaded = [
            (path, attempt)
            for path in args.input
            for attempt in load_attempts(path)
        ]
        result = compare_attempts(loaded)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
