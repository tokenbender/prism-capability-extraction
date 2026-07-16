#!/usr/bin/env python3
"""Capture a compact CUDA-kernel profile for one physical BFCL runtime setup."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from bfcl_direct_qwen3 import messages_for_generation, read_records
from load_bfcl_physical_bundle import load_physical_bundle


def _number(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _is_cuda_event(event: Any) -> bool:
    return "cuda" in str(getattr(event, "device_type", "")).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=("separate", "packed_gate_up"),
        default="packed_gate_up",
    )
    parser.add_argument("--width-alignment", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--bfcl-canonicalization-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.top_k <= 0:
        parser.error("batch size, max new tokens, and top-k must be positive")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    load_started = time.perf_counter()
    model, tokenizer, load_receipt = load_physical_bundle(
        args.bundle,
        device=args.device,
        attention_implementation=args.attention_implementation,
        mlp_implementation=args.mlp_implementation,
        width_alignment=args.width_alignment,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    rows = read_records(args.pairs)[: args.batch_size]
    if not rows:
        raise ValueError("profile selection is empty")
    encoded_items = [
        tokenizer.apply_chat_template(
            messages_for_generation(
                row,
                bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
            ),
            tools=row.get("tools") or None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=args.enable_thinking,
        )
        for row in rows
    ]
    encoded = tokenizer.pad(
        encoded_items,
        padding=True,
        return_tensors="pt",
    ).to(model.get_input_embeddings().weight.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.inference_mode():
        model.generate(**encoded, **generation_kwargs)
    torch.cuda.synchronize()

    profile_started = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode(), torch.profiler.record_function(
            "physical_bfcl_generate"
        ):
            model.generate(**encoded, **generation_kwargs)
        torch.cuda.synchronize()
    profile_elapsed_seconds = time.perf_counter() - profile_started

    averaged = []
    for event in prof.key_averages(group_by_input_shape=True):
        self_device_us = _number(
            event, "self_device_time_total", "self_cuda_time_total"
        )
        if self_device_us <= 0:
            continue
        averaged.append(
            {
                "name": str(event.key),
                "count": int(event.count),
                "self_device_time_total_us": self_device_us,
                "device_time_total_us": _number(
                    event, "device_time_total", "cuda_time_total"
                ),
                "input_shapes": event.input_shapes,
            }
        )
    averaged.sort(key=lambda row: row["self_device_time_total_us"], reverse=True)

    raw_events = list(prof.events())
    cuda_events = [event for event in raw_events if _is_cuda_event(event)]
    report = {
        "status": "pass",
        "candidate": {
            "attention_implementation": args.attention_implementation,
            "mlp_implementation": args.mlp_implementation,
            "width_alignment": args.width_alignment,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input": {
            "examples": len(rows),
            "useful_prompt_tokens": int(encoded["attention_mask"].sum().item()),
            "padded_prompt_tokens": int(encoded["input_ids"].numel()),
            "padded_prompt_width": int(encoded["input_ids"].shape[-1]),
        },
        "load_seconds": load_seconds,
        "profile_elapsed_seconds": profile_elapsed_seconds,
        "kernel_launch_count": len(cuda_events),
        "top_cuda_operators_by_self_device_time": averaged[: args.top_k],
        "load_receipt": load_receipt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
