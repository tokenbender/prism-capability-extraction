#!/usr/bin/env python3
"""Compare same-harness dense-parent and physical-bundle benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mean(report: dict[str, Any], key: str) -> float:
    return float(report["summary"][key]["mean"])


def phase_mean(report: dict[str, Any], key: str) -> float:
    return float(report["phase_summary"][key]["mean"])


def percent_change(before: float, after: float) -> float:
    return 100.0 * (after / before - 1.0)


def percent_reduction(before: float, after: float) -> float:
    return 100.0 * (1.0 - after / before)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dense = json.loads(args.dense.read_text())
    physical = json.loads(args.physical.read_text())
    contract_keys = (
        "gpu",
        "torch",
        "cuda",
        "dtype",
        "attention_implementation",
        "batch_size",
        "max_new_tokens",
        "warmup",
        "repeats",
        "decode_steps_per_batch",
    )
    mismatches = {
        key: {"dense": dense.get(key), "physical": physical.get(key)}
        for key in contract_keys
        if dense.get(key) != physical.get(key)
    }
    if mismatches:
        raise ValueError(f"benchmark contracts differ: {mismatches}")

    dense_load = float(dense["load_seconds"])
    physical_load = float(physical["load_seconds"])
    dense_after_allocated = float(dense["after_load_memory"]["allocated_bytes"])
    physical_after_allocated = float(physical["after_load_memory"]["allocated_bytes"])
    dense_after_reserved = float(dense["after_load_memory"]["reserved_bytes"])
    physical_after_reserved = float(physical["after_load_memory"]["reserved_bytes"])
    dense_peak_allocated = mean(dense, "peak_allocated_bytes")
    physical_peak_allocated = mean(physical, "peak_allocated_bytes")
    dense_peak_reserved = mean(dense, "peak_reserved_bytes")
    physical_peak_reserved = mean(physical, "peak_reserved_bytes")
    dense_throughput = mean(dense, "generated_tokens_per_second")
    physical_throughput = mean(physical, "generated_tokens_per_second")
    dense_prefill = phase_mean(dense, "prefill_tokens_per_second")
    physical_prefill = phase_mean(physical, "prefill_tokens_per_second")
    dense_decode = phase_mean(dense, "decode_tokens_per_second")
    physical_decode = phase_mean(physical, "decode_tokens_per_second")
    dense_decode_latency = phase_mean(dense, "decode_milliseconds_per_step")
    physical_decode_latency = phase_mean(physical, "decode_milliseconds_per_step")

    comparison = {
        "status": "pass",
        "contract": {key: dense.get(key) for key in contract_keys},
        "dense_report": str(args.dense),
        "physical_report": str(args.physical),
        "load_seconds": {
            "dense": dense_load,
            "physical": physical_load,
            "physical_change_percent": percent_change(dense_load, physical_load),
        },
        "after_load_allocated_bytes": {
            "dense": dense_after_allocated,
            "physical": physical_after_allocated,
            "physical_reduction_percent": percent_reduction(
                dense_after_allocated, physical_after_allocated
            ),
        },
        "after_load_reserved_bytes": {
            "dense": dense_after_reserved,
            "physical": physical_after_reserved,
            "physical_reduction_percent": percent_reduction(
                dense_after_reserved, physical_after_reserved
            ),
        },
        "peak_allocated_bytes": {
            "dense": dense_peak_allocated,
            "physical": physical_peak_allocated,
            "physical_reduction_percent": percent_reduction(
                dense_peak_allocated, physical_peak_allocated
            ),
        },
        "peak_reserved_bytes": {
            "dense": dense_peak_reserved,
            "physical": physical_peak_reserved,
            "physical_reduction_percent": percent_reduction(
                dense_peak_reserved, physical_peak_reserved
            ),
        },
        "generated_tokens_per_second": {
            "dense": dense_throughput,
            "physical": physical_throughput,
            "physical_change_percent": percent_change(dense_throughput, physical_throughput),
        },
        "prefill_tokens_per_second": {
            "dense": dense_prefill,
            "physical": physical_prefill,
            "physical_change_percent": percent_change(dense_prefill, physical_prefill),
        },
        "decode_tokens_per_second": {
            "dense": dense_decode,
            "physical": physical_decode,
            "physical_change_percent": percent_change(dense_decode, physical_decode),
        },
        "decode_milliseconds_per_step": {
            "dense": dense_decode_latency,
            "physical": physical_decode_latency,
            "physical_change_percent": percent_change(
                dense_decode_latency, physical_decode_latency
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
