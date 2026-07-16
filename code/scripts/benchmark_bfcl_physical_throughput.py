#!/usr/bin/env python3
"""Issue #19 throughput and phase benchmark for the physical BFCL substrate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from bfcl_direct_qwen3 import messages_for_generation, read_records
from load_bfcl_physical_bundle import load_physical_bundle


COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 0.95),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def accepted_generation_tokens(
    sequences: torch.Tensor,
    *,
    prompt_width: int,
    eos_token_ids: set[int],
) -> int:
    """Count generated tokens through the first EOS, not padded decode slots."""

    suffix = sequences[:, prompt_width:].detach().cpu().tolist()
    total = 0
    for row in suffix:
        count = 0
        for token in row:
            count += 1
            if int(token) in eos_token_ids:
                break
        total += count
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=("separate", "packed_gate_up"),
        default="separate",
    )
    parser.add_argument("--width-alignment", type=int, default=1)
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, default="none")
    parser.add_argument(
        "--cache-implementation",
        choices=("dynamic", "static"),
        default="dynamic",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--bfcl-canonicalization-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.width_alignment not in (1, 16, 64, 128, 256):
        parser.error("--width-alignment must be one of 1, 16, 64, 128, 256")
    if args.mlp_implementation == "separate" and args.width_alignment != 1:
        parser.error("width alignment requires --mlp-implementation packed_gate_up")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max new tokens must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative and repeats must be positive")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

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
    after_load = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }

    compile_wrap_seconds = 0.0
    if args.compile_mode != "none":
        compile_started = time.perf_counter()
        model = torch.compile(model, mode=args.compile_mode)
        compile_wrap_seconds = time.perf_counter() - compile_started

    rows = read_records(args.pairs)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("benchmark selection is empty")

    encoded_batches: list[dict[str, torch.Tensor]] = []
    input_device = model.get_input_embeddings().weight.device
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
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
            for row in batch_rows
        ]
        encoded_batches.append(
            tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(
                input_device
            )
        )

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_token_ids = {eos_ids}
    else:
        eos_token_ids = {int(token) for token in eos_ids}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }
    if args.cache_implementation != "dynamic":
        generation_kwargs["cache_implementation"] = args.cache_implementation

    def run_generation_once() -> dict[str, float | int]:
        examples = 0
        useful_prompt_tokens = 0
        padded_prompt_tokens = 0
        generated_slots = 0
        accepted_tokens = 0
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            for encoded in encoded_batches:
                output = model.generate(**encoded, **generation_kwargs)
                sequences = output.sequences
                batch = int(sequences.shape[0])
                prompt_width = int(encoded["input_ids"].shape[-1])
                examples += batch
                useful_prompt_tokens += int(encoded["attention_mask"].sum().item())
                padded_prompt_tokens += int(encoded["input_ids"].numel())
                generated_slots += int((sequences.shape[-1] - prompt_width) * batch)
                accepted_tokens += accepted_generation_tokens(
                    sequences,
                    prompt_width=prompt_width,
                    eos_token_ids=eos_token_ids,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "elapsed_seconds": elapsed,
            "examples": examples,
            "useful_prompt_tokens": useful_prompt_tokens,
            "padded_prompt_tokens": padded_prompt_tokens,
            "generated_slots": generated_slots,
            "accepted_generated_tokens": accepted_tokens,
            "examples_per_second": examples / elapsed,
            "generated_slots_per_second": generated_slots / elapsed,
            "accepted_generated_tokens_per_second": accepted_tokens / elapsed,
            "useful_prompt_tokens_per_second": useful_prompt_tokens / elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    def run_phase_once() -> dict[str, float | int]:
        prefill_seconds = 0.0
        prefill_useful_tokens = 0
        prefill_padded_tokens = 0
        decode_seconds = 0.0
        decode_tokens = 0
        with torch.inference_mode():
            for encoded in encoded_batches:
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                batch, prompt_width = input_ids.shape
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                cache_position = torch.arange(prompt_width, device=input_ids.device)

                torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                prefill_seconds += time.perf_counter() - started
                prefill_useful_tokens += int(attention_mask.sum().item())
                prefill_padded_tokens += int(input_ids.numel())

                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                decode_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (batch, args.decode_steps),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                next_positions = attention_mask.sum(dim=-1, keepdim=True)

                torch.cuda.synchronize()
                started = time.perf_counter()
                for step in range(args.decode_steps):
                    output = model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        attention_mask=decode_mask[:, : prompt_width + step + 1],
                        position_ids=next_positions + step,
                        cache_position=torch.tensor(
                            [prompt_width + step], device=input_ids.device
                        ),
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                    past_key_values = output.past_key_values
                torch.cuda.synchronize()
                decode_seconds += time.perf_counter() - started
                decode_tokens += int(batch) * args.decode_steps

        return {
            "prefill_seconds": prefill_seconds,
            "prefill_useful_tokens": prefill_useful_tokens,
            "prefill_padded_tokens": prefill_padded_tokens,
            "prefill_useful_tokens_per_second": prefill_useful_tokens
            / prefill_seconds,
            "prefill_padded_tokens_per_second": prefill_padded_tokens
            / prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds,
            "decode_milliseconds_per_step": (
                1000.0 * decode_seconds / (len(encoded_batches) * args.decode_steps)
            ),
        }

    warmup_measurements = [run_generation_once() for _ in range(args.warmup)]
    generation_measurements = [run_generation_once() for _ in range(args.repeats)]
    run_phase_once()
    phase_measurements = [run_phase_once() for _ in range(args.repeats)]

    report = {
        "status": "pass",
        "candidate": {
            "attention_implementation": args.attention_implementation,
            "mlp_implementation": args.mlp_implementation,
            "width_alignment": args.width_alignment,
            "compile_mode": args.compile_mode,
            "cache_implementation": args.cache_implementation,
        },
        "contract": {
            "pairs": str(args.pairs),
            "examples": len(rows),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "decode_steps": args.decode_steps,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "enable_thinking": args.enable_thinking,
            "bfcl_canonicalization_prompt": args.bfcl_canonicalization_prompt,
            "tokenizer_fix_mistral_regex": False,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "load_receipt": load_receipt,
        "load_seconds": load_seconds,
        "compile_wrap_seconds": compile_wrap_seconds,
        "after_load_memory": after_load,
        "warmup_measurements": warmup_measurements,
        "generation_measurements": generation_measurements,
        "phase_measurements": phase_measurements,
        "summary": {
            key: summarize([float(row[key]) for row in generation_measurements])
            for key in (
                "elapsed_seconds",
                "examples_per_second",
                "generated_slots_per_second",
                "accepted_generated_tokens_per_second",
                "useful_prompt_tokens_per_second",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        },
        "phase_summary": {
            key: summarize([float(row[key]) for row in phase_measurements])
            for key in (
                "prefill_seconds",
                "prefill_useful_tokens_per_second",
                "prefill_padded_tokens_per_second",
                "decode_seconds",
                "decode_tokens_per_second",
                "decode_milliseconds_per_step",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
