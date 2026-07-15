#!/usr/bin/env python3
"""Same-GPU repeated generation benchmark for dense and physical BFCL models."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from bfcl_direct_qwen3 import messages_for_generation, read_records
from load_bfcl_physical_bundle import load_physical_bundle


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dense", "bundle"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    args = parser.parse_args()

    if args.mode == "dense" and args.model is None:
        parser.error("--mode dense requires --model")
    if args.mode == "bundle" and args.bundle is None:
        parser.error("--mode bundle requires --bundle")

    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    if args.mode == "dense":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            fix_mistral_regex=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map={"": args.device},
            attn_implementation="eager",
        )
        model.eval()
        isolation: dict[str, Any] = {"mode": "dense_merged_parent"}
    else:
        model, tokenizer, isolation = load_physical_bundle(
            args.bundle,
            device=args.device,
        )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    after_load = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }

    rows = read_records(args.pairs)
    batches = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batches.append(
            [
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
        )
    input_device = model.get_input_embeddings().weight.device

    def run_once() -> dict[str, float | int]:
        examples = 0
        prompt_tokens = 0
        generated_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            for encoded_items in batches:
                encoded = tokenizer.pad(encoded_items, padding=True, return_tensors="pt").to(input_device)
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                examples += int(output.shape[0])
                prompt_tokens += int(encoded["attention_mask"].sum().item())
                generated_tokens += int(
                    (output.shape[-1] - encoded["input_ids"].shape[-1]) * output.shape[0]
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {
            "elapsed_seconds": elapsed,
            "examples": examples,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "examples_per_second": examples / elapsed,
            "prompt_tokens_per_second": prompt_tokens / elapsed,
            "generated_tokens_per_second": generated_tokens / elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def run_prefill_decode_once() -> dict[str, float | int]:
        prefill_seconds = 0.0
        prefill_tokens = 0
        decode_seconds = 0.0
        decode_tokens = 0
        decode_steps = 0
        with torch.inference_mode():
            for encoded_items in batches:
                encoded = tokenizer.pad(
                    encoded_items,
                    padding=True,
                    return_tensors="pt",
                ).to(input_device)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                cache_position = torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                )

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
                prefill_tokens += int(attention_mask.sum().item())

                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                past_key_values = output.past_key_values
                running_ids = torch.cat([input_ids, next_token], dim=-1)
                running_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=-1,
                )

                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.decode_steps):
                    position_ids = running_mask.long().cumsum(-1)[:, -1:] - 1
                    cache_position = torch.tensor(
                        [past_key_values.get_seq_length()],
                        device=running_ids.device,
                    )
                    output = model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        attention_mask=running_mask,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        use_cache=True,
                        logits_to_keep=1,
                        return_dict=True,
                    )
                    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                    past_key_values = output.past_key_values
                    running_ids = torch.cat([running_ids, next_token], dim=-1)
                    running_mask = torch.cat(
                        [running_mask, torch.ones_like(next_token)],
                        dim=-1,
                    )
                torch.cuda.synchronize()
                decode_seconds += time.perf_counter() - started
                batch_decode_tokens = int(input_ids.shape[0]) * args.decode_steps
                decode_tokens += batch_decode_tokens
                decode_steps += args.decode_steps

        return {
            "prefill_seconds": prefill_seconds,
            "prefill_tokens": prefill_tokens,
            "prefill_tokens_per_second": prefill_tokens / prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds,
            "decode_steps": decode_steps,
            "decode_milliseconds_per_step": 1000.0 * decode_seconds / decode_steps,
        }

    for _ in range(args.warmup):
        run_once()
    measurements = [run_once() for _ in range(args.repeats)]
    run_prefill_decode_once()
    phase_measurements = [run_prefill_decode_once() for _ in range(args.repeats)]

    report = {
        "mode": args.mode,
        "model": str(args.model) if args.model else None,
        "bundle": str(args.bundle) if args.bundle else None,
        "pairs": str(args.pairs),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "attention_implementation": "eager",
        "tokenizer_fix_mistral_regex": False,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decode_steps_per_batch": args.decode_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "after_load_memory": after_load,
        "isolation": isolation,
        "measurements": measurements,
        "phase_measurements": phase_measurements,
        "summary": {
            "elapsed_seconds": summarize([float(row["elapsed_seconds"]) for row in measurements]),
            "examples_per_second": summarize(
                [float(row["examples_per_second"]) for row in measurements]
            ),
            "prompt_tokens_per_second": summarize(
                [float(row["prompt_tokens_per_second"]) for row in measurements]
            ),
            "generated_tokens_per_second": summarize(
                [float(row["generated_tokens_per_second"]) for row in measurements]
            ),
            "peak_allocated_bytes": summarize(
                [float(row["peak_allocated_bytes"]) for row in measurements]
            ),
            "peak_reserved_bytes": summarize(
                [float(row["peak_reserved_bytes"]) for row in measurements]
            ),
        },
        "phase_summary": {
            "prefill_seconds": summarize(
                [float(row["prefill_seconds"]) for row in phase_measurements]
            ),
            "prefill_tokens_per_second": summarize(
                [float(row["prefill_tokens_per_second"]) for row in phase_measurements]
            ),
            "decode_seconds": summarize(
                [float(row["decode_seconds"]) for row in phase_measurements]
            ),
            "decode_tokens_per_second": summarize(
                [float(row["decode_tokens_per_second"]) for row in phase_measurements]
            ),
            "decode_milliseconds_per_step": summarize(
                [float(row["decode_milliseconds_per_step"]) for row in phase_measurements]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
