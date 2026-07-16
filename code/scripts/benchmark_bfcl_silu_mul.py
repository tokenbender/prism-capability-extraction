#!/usr/bin/env python3
"""Benchmark Torch and optional Triton SiLU-multiply on physical BFCL widths."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence


SUPPORTED_FORMAT = "qwen_physical_mlp_substrate_v1"
SUPPORTED_ALIGNMENTS = (1, 16, 64, 128, 256)
DEFAULT_ROWS = (1, 8, 32, 128, 512, 2048)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p95": percentile(numeric, 0.95),
        "min": min(numeric),
        "max": max(numeric),
    }


def align_width(width: int, alignment: int) -> int:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
    ):
        raise ValueError("alignment must be a positive integer")
    return ((width + alignment - 1) // alignment) * alignment


def read_physical_widths(bundle: Path) -> list[int]:
    metadata_path = bundle / "substrate_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing bundle metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("substrate metadata must contain a JSON object")
    if metadata.get("format") != SUPPORTED_FORMAT:
        raise ValueError(
            f"unsupported bundle format {metadata.get('format')!r}; "
            f"expected {SUPPORTED_FORMAT!r}"
        )
    isolation = metadata.get("isolation")
    if not isinstance(isolation, dict):
        raise ValueError("substrate metadata is missing the isolation object")
    kept = isolation.get("kept_per_layer")
    if not isinstance(kept, dict) or not kept:
        raise ValueError("isolation.kept_per_layer must be a nonempty object")
    layers = isolation.get("layers", len(kept))
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError("isolation.layers must be a positive integer")
    expected_keys = {str(index) for index in range(layers)}
    if set(kept) != expected_keys:
        raise ValueError("kept_per_layer keys must cover every layer exactly")
    widths: list[int] = []
    for index in range(layers):
        value = kept[str(index)]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"layer {index} width must be a positive integer")
        widths.append(value)
    kept_total = isolation.get("kept_total")
    if kept_total is not None:
        if isinstance(kept_total, bool) or not isinstance(kept_total, int):
            raise ValueError("isolation.kept_total must be an integer")
        if kept_total != sum(widths):
            raise ValueError(
                f"isolation.kept_total mismatch: {kept_total!r} != {sum(widths)}"
            )
    return widths


def plan_shapes(
    widths: Sequence[int],
    row_counts: Sequence[int],
    *,
    alignment: int,
    max_elements: int,
    triton_enabled: bool,
) -> list[dict[str, Any]]:
    if not widths:
        raise ValueError("at least one physical width is required")
    if not row_counts:
        raise ValueError("at least one row count is required")
    if max_elements <= 0:
        raise ValueError("max_elements must be positive")
    groups: dict[int, dict[str, list[int]]] = {}
    for layer, width in enumerate(widths):
        aligned = align_width(width, alignment)
        group = groups.setdefault(aligned, {"layers": [], "active_widths": []})
        group["layers"].append(layer)
        if width not in group["active_widths"]:
            group["active_widths"].append(width)

    # With Triton enabled, accuracy can simultaneously hold the packed 2N
    # input, two N-element low-precision outputs, two N-element float casts,
    # and one N-element float subtraction. Torch-only execution peaks at the
    # packed input plus its SiLU intermediate and output (4N).
    live_factor = 7 if triton_enabled else 4
    plans: list[dict[str, Any]] = []
    for aligned, group in sorted(groups.items()):
        for rows in row_counts:
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise ValueError("row counts must be positive integers")
            output_elements = rows * aligned
            estimated_peak = live_factor * output_elements
            skipped = estimated_peak > max_elements
            plans.append(
                {
                    "rows": rows,
                    "aligned_width": aligned,
                    "layer_indices": group["layers"],
                    "active_widths": sorted(group["active_widths"]),
                    "input_elements": 2 * output_elements,
                    "output_elements": output_elements,
                    "estimated_peak_live_elements": estimated_peak,
                    "max_elements": max_elements,
                    "status": "skipped" if skipped else "planned",
                    "skip_reason": (
                        "estimated_peak_live_elements_exceed_max"
                        if skipped
                        else None
                    ),
                }
            )
    return plans


def accuracy_summary(
    *,
    total_elements: int,
    exact_elements: int,
    max_abs_error: float,
    mean_abs_error: float,
) -> dict[str, int | float]:
    if total_elements <= 0:
        raise ValueError("total_elements must be positive")
    if not 0 <= exact_elements <= total_elements:
        raise ValueError("exact_elements must fall within total_elements")
    if (
        not math.isfinite(max_abs_error)
        or not math.isfinite(mean_abs_error)
        or max_abs_error < 0.0
        or mean_abs_error < 0.0
    ):
        raise ValueError("absolute errors must be finite and nonnegative")
    return {
        "total_elements": total_elements,
        "exact_elements": exact_elements,
        "exact_element_fraction": exact_elements / total_elements,
        "finite_error_elements": total_elements,
        "all_errors_finite": True,
        "max_abs_error": float(max_abs_error),
        "mean_abs_error": float(mean_abs_error),
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator) / float(denominator)


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch_results = [
        result
        for result in results
        if result.get("status") in {"measured", "torch_only"}
        and isinstance(result.get("torch"), dict)
    ]
    triton_results = [
        result
        for result in results
        if isinstance(result.get("triton"), dict)
        and result["triton"].get("status") == "measured"
    ]
    speedups = [
        float(result["speedup"]["median"])
        for result in triton_results
        if result.get("speedup", {}).get("median") is not None
    ]
    accuracy_rows = [
        result["accuracy"]
        for result in triton_results
        if isinstance(result.get("accuracy"), dict)
    ]
    total_elements = sum(int(row["total_elements"]) for row in accuracy_rows)
    exact_elements = sum(int(row["exact_elements"]) for row in accuracy_rows)
    all_errors_finite = all(
        bool(row.get("all_errors_finite", True)) for row in accuracy_rows
    )
    finite_error_elements = sum(
        int(row.get("finite_error_elements", row["total_elements"]))
        for row in accuracy_rows
    )
    aggregate_accuracy = None
    if total_elements:
        aggregate_accuracy = {
            "total_elements": total_elements,
            "exact_elements": exact_elements,
            "exact_element_fraction": exact_elements / total_elements,
            "finite_error_elements": finite_error_elements,
            "all_errors_finite": all_errors_finite,
            "max_abs_error": None,
            "mean_abs_error": None,
        }
        if all_errors_finite:
            weighted_abs_error = sum(
                float(row["mean_abs_error"]) * int(row["total_elements"])
                for row in accuracy_rows
            )
            aggregate_accuracy["max_abs_error"] = max(
                float(row["max_abs_error"]) for row in accuracy_rows
            )
            aggregate_accuracy["mean_abs_error"] = (
                weighted_abs_error / total_elements
            )

    skip_reasons = Counter(
        str(result.get("skip_reason"))
        for result in results
        if result.get("status") == "skipped"
    )
    triton_statuses = Counter(
        str(result["triton"].get("status"))
        for result in results
        if isinstance(result.get("triton"), dict)
    )
    return {
        "planned_shape_dtype_cases": len(results),
        "torch_measured_cases": len(torch_results),
        "triton_measured_cases": len(triton_results),
        "skipped_cases": sum(1 for result in results if result.get("status") == "skipped"),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "triton_statuses": dict(sorted(triton_statuses.items())),
        "torch_shape_median_latency_milliseconds": (
            summarize(
                [
                    float(result["torch"]["latency_milliseconds"]["median"])
                    for result in torch_results
                ]
            )
            if torch_results
            else None
        ),
        "triton_shape_median_latency_milliseconds": (
            summarize(
                [
                    float(result["triton"]["latency_milliseconds"]["median"])
                    for result in triton_results
                ]
            )
            if triton_results
            else None
        ),
        "triton_speedup_across_shapes": summarize(speedups) if speedups else None,
        "accuracy": aggregate_accuracy,
    }


def _cuda_wall_call(
    torch: Any,
    device: Any,
    operation: Callable[[], Any],
) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = operation()
    torch.cuda.synchronize(device)
    return output, 1000.0 * (time.perf_counter() - started)


def _warm_cuda(
    torch: Any,
    device: Any,
    operation: Callable[[], Any],
    warmups: int,
) -> None:
    for _ in range(warmups):
        output = operation()
        del output
    torch.cuda.synchronize(device)


def _cuda_event_latencies(
    torch: Any,
    device: Any,
    operation: Callable[[], Any],
    repeats: int,
) -> list[float]:
    events: list[tuple[Any, Any]] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        del output
        events.append((start, end))
    torch.cuda.synchronize(device)
    return [float(start.elapsed_time(end)) for start, end in events]


def _tensor_accuracy(torch: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"accuracy tensors differ in shape: {reference.shape} != {candidate.shape}"
        )
    total = int(reference.numel())
    exact = int(torch.count_nonzero(reference == candidate).item())
    difference = (reference.float() - candidate.float()).abs()
    finite_elements = int(torch.count_nonzero(torch.isfinite(difference)).item())
    if finite_elements != total:
        return {
            "total_elements": total,
            "exact_elements": exact,
            "exact_element_fraction": exact / total,
            "finite_error_elements": finite_elements,
            "all_errors_finite": False,
            "max_abs_error": None,
            "mean_abs_error": None,
        }
    return accuracy_summary(
        total_elements=total,
        exact_elements=exact,
        max_abs_error=float(difference.max().item()),
        mean_abs_error=float(difference.mean().item()),
    )


def _load_triton_kernel(mode: str) -> tuple[Callable[[Any], Any] | None, dict[str, Any]]:
    if mode == "off":
        return None, {"status": "disabled", "version": None, "error": None}
    try:
        module = importlib.import_module("triton_silu_mul")
        kernel = getattr(module, "triton_silu_and_mul")
        version = getattr(getattr(module, "triton", None), "__version__", None)
        return kernel, {"status": "available", "version": version, "error": None}
    except Exception as exc:
        if mode == "required":
            raise RuntimeError("required Triton SiLU-multiply kernel is unavailable") from exc
        return None, {
            "status": "unavailable",
            "version": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _benchmark_shape(
    *,
    torch: Any,
    functional: Any,
    triton_kernel: Callable[[Any], Any] | None,
    triton_required: bool,
    device: Any,
    dtype: Any,
    dtype_name: str,
    plan: dict[str, Any],
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rows = int(plan["rows"])
    width = int(plan["aligned_width"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    gate_up = torch.randn(
        (rows, 2 * width),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    def torch_operation() -> Any:
        gate, up = gate_up.chunk(2, dim=-1)
        return functional.silu(gate) * up

    result = {
        **plan,
        "status": "measured" if triton_kernel is not None else "torch_only",
        "skip_reason": None,
        "dtype": dtype_name,
        "seed": seed,
        "torch": None,
        "triton": None,
        "speedup": None,
        "accuracy": None,
    }
    with torch.inference_mode():
        first_output, first_wall_ms = _cuda_wall_call(
            torch, device, torch_operation
        )
        del first_output
        _warm_cuda(torch, device, torch_operation, warmups)
        torch_latencies = _cuda_event_latencies(
            torch, device, torch_operation, repeats
        )
        result["torch"] = {
            "status": "measured",
            "first_call_wall_milliseconds": first_wall_ms,
            "latency_milliseconds": summarize(torch_latencies),
        }

        if triton_kernel is not None:

            def triton_operation() -> Any:
                return triton_kernel(gate_up)

            try:
                first_output, first_wall_ms = _cuda_wall_call(
                    torch, device, triton_operation
                )
                del first_output
                _warm_cuda(torch, device, triton_operation, warmups)
                triton_latencies = _cuda_event_latencies(
                    torch, device, triton_operation, repeats
                )
                reference = torch_operation()
                candidate = triton_operation()
                torch.cuda.synchronize(device)
                result["accuracy"] = _tensor_accuracy(
                    torch, reference, candidate
                )
                del reference, candidate
                result["triton"] = {
                    "status": "measured",
                    "first_call_wall_milliseconds": first_wall_ms,
                    "latency_milliseconds": summarize(triton_latencies),
                }
                result["speedup"] = {
                    "definition": "torch_latency_divided_by_triton_latency",
                    "median": safe_ratio(
                        float(result["torch"]["latency_milliseconds"]["median"]),
                        float(result["triton"]["latency_milliseconds"]["median"]),
                    ),
                    "p95": safe_ratio(
                        float(result["torch"]["latency_milliseconds"]["p95"]),
                        float(result["triton"]["latency_milliseconds"]["p95"]),
                    ),
                }
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                if triton_required:
                    raise
                result["status"] = "torch_only"
                result["triton"] = {
                    "status": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }

    result["observed_peak_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    del gate_up
    torch.cuda.empty_cache()
    return result


def _dtype_skip_result(
    plan: dict[str, Any], *, dtype_name: str, reason: str
) -> dict[str, Any]:
    return {
        **plan,
        "status": "skipped",
        "skip_reason": reason,
        "dtype": dtype_name,
        "seed": None,
        "torch": None,
        "triton": None,
        "speedup": None,
        "accuracy": None,
        "observed_peak_allocated_bytes": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("bf16", "fp16"),
        help="dtype to benchmark; repeat to select both (default: both)",
    )
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument(
        "--alignment", type=int, choices=SUPPORTED_ALIGNMENTS, default=128
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-elements", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--triton", choices=("auto", "required", "off"), default="auto"
    )
    args = parser.parse_args()

    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats must be positive")
    if args.max_elements <= 0:
        parser.error("max-elements must be positive")
    if any(rows <= 0 for rows in args.rows):
        parser.error("row counts must be positive")
    dtype_names = list(dict.fromkeys(args.dtype or ["bf16", "fp16"]))

    widths = read_physical_widths(args.bundle)
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("the benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.cuda.set_device(device)
    triton_kernel, triton_receipt = _load_triton_kernel(args.triton)
    plans = plan_shapes(
        widths,
        list(dict.fromkeys(args.rows)),
        alignment=args.alignment,
        max_elements=args.max_elements,
        triton_enabled=triton_kernel is not None,
    )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    results: list[dict[str, Any]] = []
    first_triton_call_receipt = None
    triton_runtime_errors: list[dict[str, Any]] = []
    case_index = 0

    for dtype_name in dtype_names:
        dtype_supported = not (
            dtype_name == "bf16" and not torch.cuda.is_bf16_supported()
        )
        for plan in plans:
            case_index += 1
            if plan["status"] == "skipped":
                results.append(
                    _dtype_skip_result(
                        plan,
                        dtype_name=dtype_name,
                        reason=str(plan["skip_reason"]),
                    )
                )
                continue
            if not dtype_supported:
                results.append(
                    _dtype_skip_result(
                        plan,
                        dtype_name=dtype_name,
                        reason="cuda_bf16_not_supported",
                    )
                )
                continue
            try:
                result = _benchmark_shape(
                    torch=torch,
                    functional=functional,
                    triton_kernel=triton_kernel,
                    triton_required=args.triton == "required",
                    device=device,
                    dtype=dtype_map[dtype_name],
                    dtype_name=dtype_name,
                    plan=plan,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    seed=args.seed + case_index,
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = _dtype_skip_result(
                    plan,
                    dtype_name=dtype_name,
                    reason=f"cuda_out_of_memory: {exc}",
                )
            results.append(result)
            if (
                first_triton_call_receipt is None
                and isinstance(result.get("triton"), dict)
                and result["triton"].get("status") == "measured"
            ):
                first_triton_call_receipt = {
                    "dtype": dtype_name,
                    "rows": plan["rows"],
                    "aligned_width": plan["aligned_width"],
                    "wall_milliseconds": result["triton"][
                        "first_call_wall_milliseconds"
                    ],
                    "scope": (
                        "first Triton invocation in this process; includes any "
                        "cache lookup or JIT, first kernel execution, and CUDA "
                        "synchronization"
                    ),
                }
            if (
                args.triton == "auto"
                and isinstance(result.get("triton"), dict)
                and result["triton"].get("status") == "error"
            ):
                triton_runtime_errors.append(
                    {
                        "dtype": dtype_name,
                        "rows": plan["rows"],
                        "aligned_width": plan["aligned_width"],
                        "error": result["triton"]["error"],
                    }
                )

    aggregate = aggregate_results(results)
    report = {
        "status": "complete" if aggregate["torch_measured_cases"] else "skipped",
        "benchmark": "torch_silu_mul_vs_optional_triton",
        "bundle": {
            "path": str(args.bundle),
            "metadata": str(args.bundle / "substrate_metadata.json"),
            "format": SUPPORTED_FORMAT,
            "layers": len(widths),
            "physical_widths": widths,
            "unique_aligned_widths": sorted(
                {align_width(width, args.alignment) for width in widths}
            ),
        },
        "configuration": {
            "device": str(device),
            "dtypes": dtype_names,
            "rows": list(dict.fromkeys(args.rows)),
            "alignment": args.alignment,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "max_elements": args.max_elements,
            "max_elements_scope": (
                "conservative estimated peak simultaneously live tensor elements "
                "per shape"
            ),
            "steady_state_timing": "one CUDA event pair per repeat",
            "accuracy_interpretation": (
                "raw Torch-versus-Triton elementwise comparison; no numerical "
                "pass/fail tolerance is applied"
            ),
            "seed": args.seed,
            "triton_mode": args.triton,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "device": str(device),
            "triton": triton_receipt.get("version"),
        },
        "triton": {
            **triton_receipt,
            "runtime_errors": triton_runtime_errors,
            "comparison_coverage": (
                "measured"
                if aggregate["triton_measured_cases"]
                else "not_measured"
            ),
            "first_call_jit_receipt": first_triton_call_receipt,
        },
        "results": results,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
