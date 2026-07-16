#!/usr/bin/env python3
"""Profile one dense or physical arithmetic generation configuration.

The report distinguishes synchronized wall time from profiler-derived
estimates.  ``torch.profiler(with_flops=True)`` only estimates FLOPs for
supported operators, and summed CUDA kernel time can double-count overlap.
Consequently, achieved TFLOP/s, active time, and MFU are estimate-only
diagnostics rather than hardware-counter measurements.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from benchmark_arithmetic_physical_throughput import COMPILE_MODES
from evaluate_arithmetic_standalone import (
    _special_token_ids,
    _trim_generated_ids,
    sha256_json,
)
from load_arithmetic_physical_bundle import load_arithmetic_physical_bundle
from load_bfcl_physical_bundle import (
    DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    SUPPORTED_ACTIVATION_IMPLEMENTATIONS,
    SUPPORTED_ATTENTION_IMPLEMENTATIONS,
    SUPPORTED_MLP_IMPLEMENTATIONS,
    SUPPORTED_WIDTH_ALIGNMENTS,
    add_generation_compile_arguments,
    build_generation_compile_settings,
    observe_generation_compile_state,
    validate_activation_runtime_settings,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
PROFILE_RANGE_NAME = "arithmetic_runtime_generate"


def _number(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is None:
            continue
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(normalized):
            return normalized
    return 0.0


def _is_cuda_event(event: Any) -> bool:
    return "cuda" in str(getattr(event, "device_type", "")).lower()


def _event_name(event: Any) -> str:
    return str(getattr(event, "name", getattr(event, "key", "unknown")))


def _is_profiler_annotation(name: str) -> bool:
    """Exclude profiler ranges and runtime markers from kernel counts."""

    return (
        name in {PROFILE_RANGE_NAME, "Command Buffer Full"}
        or name.startswith("## Call CompiledFxGraph ")
    )


def _is_transfer_or_profiler_event(name: str) -> bool:
    lowered = name.lower()
    return (
        "memcpy" in lowered
        or "memset" in lowered
        or name == "[memory]"
        or _is_profiler_annotation(name)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen_records(path: Path) -> list[dict[str, Any]]:
    """Read an exact JSON/JSONL record artifact without rerandomizing it."""

    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("records", "rows"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            else:
                raise ValueError(
                    f"{path} must contain a list or records/rows list"
                )
        else:
            raise ValueError(f"{path} must contain JSON records")
    if not rows:
        raise ValueError(f"no frozen records in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains a non-object record")
    normalized = [dict(row) for row in rows]
    ids = []
    for index, row in enumerate(normalized):
        prompt = row.get("generation_prompt", row.get("prompt"))
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"record {index} has no generation prompt")
        record_id = str(row.get("id", index))
        row["id"] = record_id
        ids.append(record_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"record IDs are not unique in {path}")
    return normalized


def select_profile_records(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    selected = [dict(row) for row in records[:batch_size]]
    if not selected:
        raise ValueError("profile selection is empty")
    return selected


def summarize_operator_events(
    events: Iterable[Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Summarize key averages and profiler-estimated supported-op FLOPs."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = []
    estimated_flops = 0.0
    flop_rows = 0
    for event in events:
        flops = max(0.0, _number(event, "flops"))
        estimated_flops += flops
        flop_rows += int(flops > 0.0)
        self_device_us = _number(
            event,
            "self_device_time_total",
            "self_cuda_time_total",
        )
        device_us = _number(
            event,
            "device_time_total",
            "cuda_time_total",
        )
        if self_device_us <= 0.0 and device_us <= 0.0 and flops <= 0.0:
            continue
        rows.append(
            {
                "name": _event_name(event),
                "count": int(getattr(event, "count", 0)),
                "self_device_time_total_us": self_device_us,
                "device_time_total_us": device_us,
                "profiler_estimated_flops": flops,
                "input_shapes": getattr(event, "input_shapes", None),
            }
        )
    by_device_time = sorted(
        rows,
        key=lambda row: row["self_device_time_total_us"],
        reverse=True,
    )
    by_flops = sorted(
        (row for row in rows if row["profiler_estimated_flops"] > 0.0),
        key=lambda row: row["profiler_estimated_flops"],
        reverse=True,
    )
    return {
        "profiler_estimated_executed_flops": estimated_flops,
        "operator_rows_with_flop_estimate": flop_rows,
        "top_operators_by_self_device_time": by_device_time[:top_k],
        "top_operators_by_profiler_estimated_flops": by_flops[:top_k],
    }


def summarize_cuda_kernels(
    events: Iterable[Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Count compute kernel events and sum their profiler device durations."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cuda_events = [event for event in events if _is_cuda_event(event)]
    totals: dict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {"count": 0, "device_time_total_us": 0.0}
    )
    for event in cuda_events:
        name = _event_name(event)
        if _is_transfer_or_profiler_event(name):
            continue
        duration = _number(
            event,
            "self_device_time_total",
            "device_time_total",
            "self_cuda_time_total",
            "cuda_time_total",
        )
        item = totals[name]
        item["count"] = int(item["count"]) + 1
        item["device_time_total_us"] = (
            float(item["device_time_total_us"]) + duration
        )
    kernels = [{"name": name, **values} for name, values in totals.items()]
    kernels.sort(key=lambda row: row["device_time_total_us"], reverse=True)
    active_us = sum(float(row["device_time_total_us"]) for row in kernels)
    return {
        "cuda_device_event_count": len(cuda_events),
        "kernel_launch_count": sum(int(row["count"]) for row in kernels),
        "profiler_estimated_summed_kernel_active_time_seconds": (
            active_us / 1_000_000.0
        ),
        "top_cuda_kernels_by_device_time": kernels[:top_k],
    }


def estimate_execution_metrics(
    *,
    profiler_estimated_flops: float,
    profiled_generation_seconds: float,
    hardware_peak_bf16_tflops: float,
    profiler_estimated_active_seconds: float | None = None,
) -> dict[str, Any]:
    """Convert profiler estimates into explicitly estimated throughput/MFU."""

    if profiler_estimated_flops < 0.0:
        raise ValueError("profiler estimated FLOPs must be nonnegative")
    if profiled_generation_seconds <= 0.0:
        raise ValueError("profiled generation seconds must be positive")
    if hardware_peak_bf16_tflops <= 0.0:
        raise ValueError("hardware peak BF16 TFLOP/s must be positive")
    achieved = (
        profiler_estimated_flops / profiled_generation_seconds / 1_000_000_000_000
    )
    result: dict[str, Any] = {
        "estimate_only": True,
        "profiler_estimated_executed_flops": profiler_estimated_flops,
        "synchronized_profiled_generation_seconds": (
            profiled_generation_seconds
        ),
        "estimated_achieved_tflops_per_second": achieved,
        "provided_hardware_peak_bf16_tflops_per_second": (
            hardware_peak_bf16_tflops
        ),
        "estimated_executed_mfu_fraction": (
            achieved / hardware_peak_bf16_tflops
        ),
        "estimated_executed_mfu_percent": (
            achieved / hardware_peak_bf16_tflops * 100.0
        ),
        "caveat": (
            "FLOPs come from torch.profiler with_flops=True and cover only "
            "supported operators; this is not a hardware-counter MFU."
        ),
    }
    if (
        profiler_estimated_active_seconds is not None
        and profiler_estimated_active_seconds > 0.0
    ):
        active_achieved = (
            profiler_estimated_flops
            / profiler_estimated_active_seconds
            / 1_000_000_000_000
        )
        result.update(
            {
                "profiler_estimated_summed_kernel_active_time_seconds": (
                    profiler_estimated_active_seconds
                ),
                "estimated_summed_kernel_active_to_wall_fraction": (
                    profiler_estimated_active_seconds
                    / profiled_generation_seconds
                ),
                "estimated_summed_kernel_active_to_wall_percent": (
                    profiler_estimated_active_seconds
                    / profiled_generation_seconds
                    * 100.0
                ),
                "estimated_active_time_tflops_per_second": active_achieved,
                "estimated_active_time_mfu_fraction": (
                    active_achieved / hardware_peak_bf16_tflops
                ),
                "estimated_active_time_mfu_percent": (
                    active_achieved / hardware_peak_bf16_tflops * 100.0
                ),
                "active_time_caveat": (
                    "summed profiler kernel durations may double-count overlap "
                    "and are a diagnostic estimate only"
                ),
            }
        )
    else:
        result.update(
            {
                "profiler_estimated_summed_kernel_active_time_seconds": (
                    profiler_estimated_active_seconds
                ),
                "estimated_summed_kernel_active_to_wall_fraction": None,
                "estimated_summed_kernel_active_to_wall_percent": None,
                "estimated_active_time_tflops_per_second": None,
                "estimated_active_time_mfu_fraction": None,
                "estimated_active_time_mfu_percent": None,
                "active_time_caveat": (
                    "no positive summed profiler kernel duration was available"
                ),
            }
        )
    return result


def count_generated_tokens(
    sequences: torch.Tensor,
    *,
    prompt_width: int,
    stop_ids: set[int],
    pad_token_id: int | None,
) -> dict[str, int]:
    if sequences.ndim != 2:
        raise TypeError("generated sequences must be rank two")
    suffix = sequences[:, prompt_width:]
    non_stop = sum(
        len(
            _trim_generated_ids(
                token_ids,
                stop_ids=stop_ids,
                pad_token_id=pad_token_id,
            )
        )
        for token_ids in suffix.detach().cpu().tolist()
    )
    return {
        "generated_token_slots": int(suffix.numel()),
        "non_stop_generated_tokens": int(non_stop),
    }


def resolve_generation_token_settings(
    model: Any,
    tokenizer: Any,
) -> tuple[set[int], int, Any]:
    eos_value = getattr(
        getattr(model, "generation_config", None),
        "eos_token_id",
        None,
    )
    if eos_value is None:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    stop_ids = _special_token_ids(eos_value)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        if not stop_ids:
            raise ValueError("tokenizer needs a pad token or model/tokenizer EOS")
        pad_token_id = min(stop_ids)
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = pad_token_id
    return stop_ids, int(pad_token_id), eos_value


def validate_outer_compile_settings(
    *,
    outer_compile_mode: str,
    cache_implementation: str,
    generation_disable_compile: bool,
) -> None:
    """Apply the same outer/Transformers compile ownership rule as the sweep."""

    if outer_compile_mode not in COMPILE_MODES:
        raise ValueError(
            f"unsupported outer compile mode {outer_compile_mode!r}"
        )
    if (
        outer_compile_mode != "none"
        and cache_implementation == "static"
        and not generation_disable_compile
    ):
        raise ValueError(
            "outer compilation and automatic StaticCache compilation cannot "
            "both own generation"
        )


def wrap_outer_compile(
    model: Any,
    *,
    mode: str,
    compile_factory: Callable[..., Any] | None = None,
) -> tuple[Any, float]:
    if mode not in COMPILE_MODES:
        raise ValueError(f"unsupported outer compile mode {mode!r}")
    if mode == "none":
        return model, 0.0
    if compile_factory is None:
        compile_factory = torch.compile
    started = time.perf_counter()
    compiled = compile_factory(model, mode=mode)
    return compiled, time.perf_counter() - started


def load_candidate(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], str, float]:
    load_started = time.perf_counter()
    if args.bundle is not None:
        kind = "physical"
        model, tokenizer, receipt = load_arithmetic_physical_bundle(
            args.bundle,
            device=str(device),
            attention_implementation=args.attention_implementation,
            mlp_implementation=args.mlp_implementation,
            activation_implementation=args.activation_implementation,
            hybrid_activation_threshold_rows=(
                args.hybrid_activation_threshold_rows
            ),
            width_alignment=args.width_alignment,
            allow_mlp_fallback=args.allow_mlp_fallback,
        )
    else:
        kind = "dense"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.dense_model,
            revision=args.dense_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.dense_model,
            revision=args.dense_revision,
            dtype=DTYPES[args.dtype],
            attn_implementation=args.attention_implementation,
        ).to(device).eval()
        receipt = {
            "status": "pass",
            "kind": "dense_parent",
            "model": args.dense_model,
            "revision": args.dense_revision,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_dtype": str(DTYPES[args.dtype]),
            "donor_model_loaded": False,
        }
    if tokenizer is None:
        raise RuntimeError("candidate loader did not restore a tokenizer")
    tokenizer.padding_side = "left"
    torch.cuda.synchronize(device)
    return model, tokenizer, receipt, kind, time.perf_counter() - load_started


def encode_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompts = [
        str(record.get("generation_prompt", record["prompt"]))
        for record in records
    ]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in dict(encoded).items()
        if isinstance(value, torch.Tensor)
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        args.batch_size <= 0
        or args.max_new_tokens <= 0
        or args.warmups <= 0
        or args.top_k <= 0
    ):
        parser.error(
            "batch size, max new tokens, warmups, and top-k must be positive"
        )
    if args.hardware_peak_bf16_tflops <= 0.0:
        parser.error("--hardware-peak-bf16-tflops must be positive")
    if args.dense_model is not None and (
        args.mlp_implementation != "separate"
        or args.activation_implementation != "torch"
        or args.width_alignment != 1
        or args.allow_mlp_fallback
    ):
        parser.error(
            "physical MLP/activation controls cannot be used with --dense-model"
        )
    try:
        validate_outer_compile_settings(
            outer_compile_mode=args.outer_compile_mode,
            cache_implementation=args.cache_implementation,
            generation_disable_compile=args.generation_disable_compile,
        )
        validate_activation_runtime_settings(
            activation_implementation=args.activation_implementation,
            hybrid_activation_threshold_rows=(
                args.hybrid_activation_threshold_rows
            ),
        )
        return build_generation_compile_settings(
            cache_implementation=args.cache_implementation,
            disable_compile=args.generation_disable_compile,
            compile_dynamic=args.generation_compile_dynamic,
        )
    except ValueError as error:
        parser.error(str(error))


def parse_args() -> tuple[
    argparse.ArgumentParser,
    argparse.Namespace,
    dict[str, Any],
    dict[str, Any],
]:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--bundle", type=Path)
    target.add_argument("--dense-model")
    parser.add_argument("--dense-revision")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--attention-implementation",
        choices=SUPPORTED_ATTENTION_IMPLEMENTATIONS,
        default="sdpa",
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=SUPPORTED_MLP_IMPLEMENTATIONS,
        default="separate",
    )
    parser.add_argument(
        "--activation-implementation",
        choices=SUPPORTED_ACTIVATION_IMPLEMENTATIONS,
        default="torch",
    )
    parser.add_argument(
        "--hybrid-activation-threshold-rows",
        type=int,
        default=DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    )
    parser.add_argument(
        "--width-alignment",
        type=int,
        choices=SUPPORTED_WIDTH_ALIGNMENTS,
        default=1,
    )
    parser.add_argument("--allow-mlp-fallback", action="store_true")
    parser.add_argument(
        "--outer-compile-mode",
        choices=COMPILE_MODES,
        default="none",
        help="wrap the loaded model with torch.compile before warmup",
    )
    add_generation_compile_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--hardware-peak-bf16-tflops",
        type=float,
        required=True,
        help=(
            "user-supplied applicable dense BF16 peak; never inferred from "
            "the GPU name"
        ),
    )
    args = parser.parse_args()
    generation_kwargs, generation_receipt = validate_args(parser, args)
    return parser, args, generation_kwargs, generation_receipt


def main() -> None:
    _, args, compile_kwargs, compile_receipt = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime profiling requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    records = read_frozen_records(args.records)
    selected = select_profile_records(records, batch_size=args.batch_size)
    model, tokenizer, load_receipt, kind, load_seconds = load_candidate(
        args,
        device=device,
    )
    model, outer_compile_wrap_seconds = wrap_outer_compile(
        model,
        mode=args.outer_compile_mode,
    )
    after_load_memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    stop_ids, pad_token_id, eos_value = resolve_generation_token_settings(
        model,
        tokenizer,
    )
    encoded = encode_records(selected, tokenizer, device=device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": False,
        "pad_token_id": pad_token_id,
        **compile_kwargs,
    }
    if eos_value is not None:
        generation_kwargs["eos_token_id"] = eos_value
    if args.cache_implementation != "dynamic":
        generation_kwargs["cache_implementation"] = args.cache_implementation

    warmup_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.warmups):
            model.generate(**encoded, **generation_kwargs)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    compile_observation = observe_generation_compile_state(model)
    after_warmup_memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }

    torch.cuda.reset_peak_memory_stats(device)
    profile_context_started = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as profiler:
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.inference_mode(), torch.profiler.record_function(
            PROFILE_RANGE_NAME
        ):
            profiled_output = model.generate(**encoded, **generation_kwargs)
        torch.cuda.synchronize(device)
        profiled_generation_seconds = time.perf_counter() - generation_started
    profile_context_seconds = time.perf_counter() - profile_context_started

    sequences = getattr(profiled_output, "sequences", profiled_output)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise TypeError("model.generate must return a rank-two token tensor")
    prompt_width = int(encoded["input_ids"].shape[-1])
    generated = count_generated_tokens(
        sequences,
        prompt_width=prompt_width,
        stop_ids=stop_ids,
        pad_token_id=pad_token_id,
    )

    operator_summary = summarize_operator_events(
        profiler.key_averages(group_by_input_shape=True),
        top_k=args.top_k,
    )
    kernel_summary = summarize_cuda_kernels(
        profiler.events(),
        top_k=args.top_k,
    )
    execution_estimates = estimate_execution_metrics(
        profiler_estimated_flops=operator_summary[
            "profiler_estimated_executed_flops"
        ],
        profiled_generation_seconds=profiled_generation_seconds,
        hardware_peak_bf16_tflops=args.hardware_peak_bf16_tflops,
        profiler_estimated_active_seconds=kernel_summary[
            "profiler_estimated_summed_kernel_active_time_seconds"
        ],
    )
    peak_memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    report = {
        "status": "pass",
        "estimate_policy": {
            "profiler_flops": "estimate_only_supported_operators",
            "summed_kernel_active_time": (
                "estimate_only_may_double_count_overlap"
            ),
            "mfu": "estimate_only_not_hardware_counter",
        },
        "candidate": {
            "kind": kind,
            "bundle": str(args.bundle) if args.bundle is not None else None,
            "dense_model": args.dense_model,
            "dense_revision": args.dense_revision,
            "dense_dtype": args.dtype if kind == "dense" else None,
            "attention_implementation": args.attention_implementation,
            "mlp_implementation": (
                args.mlp_implementation if kind == "physical" else None
            ),
            "activation_implementation": (
                args.activation_implementation if kind == "physical" else None
            ),
            "hybrid_activation_threshold_rows": (
                args.hybrid_activation_threshold_rows
                if kind == "physical"
                and args.activation_implementation == "hybrid"
                else None
            ),
            "width_alignment": (
                args.width_alignment if kind == "physical" else None
            ),
            "allow_mlp_fallback": (
                args.allow_mlp_fallback if kind == "physical" else None
            ),
            "cache_implementation": args.cache_implementation,
            "outer_compile_mode": args.outer_compile_mode,
            "generation_compile": compile_receipt,
            "batch_size_requested": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "warmups": args.warmups,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": {
            "records": str(args.records),
            "records_file_sha256": sha256_file(args.records),
            "all_frozen_records": len(records),
            "selected_records_sha256": sha256_json(selected),
            "examples": len(selected),
            "useful_prompt_tokens": int(
                encoded["attention_mask"].sum().item()
            ),
            "padded_prompt_tokens": int(encoded["input_ids"].numel()),
            "padded_prompt_width": prompt_width,
            **generated,
        },
        "timing": {
            "load_seconds": load_seconds,
            "outer_compile_wrap_seconds": outer_compile_wrap_seconds,
            "warmup_seconds": warmup_seconds,
            "synchronized_profiled_generation_seconds": (
                profiled_generation_seconds
            ),
            "profile_context_seconds": profile_context_seconds,
            "profile_timing_caveat": (
                "synchronized generation wall time includes profiler "
                "instrumentation overhead"
            ),
        },
        "memory": {
            "after_load": after_load_memory,
            "after_warmup": after_warmup_memory,
            "profiled": peak_memory,
        },
        "generation_compile_observation_after_warmup": (
            compile_observation
        ),
        "execution_estimates": execution_estimates,
        "profiler_operators": operator_summary,
        "profiler_cuda_kernels": kernel_summary,
        "load_receipt": load_receipt,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
