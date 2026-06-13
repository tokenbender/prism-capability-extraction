#!/usr/bin/env python3
"""Collect per-query BFCL MLP activation heatmaps for issue #9.

This is a descriptive activation collector, not an attribution runner. It runs
the full model on prompt + gold tool-call continuations, hooks each MLP
``down_proj`` input, and immediately reduces token-level activations to compact
per-query statistics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bfcl_direct_qwen3 import encode_prompt, format_tool_call_target  # noqa: E402


SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def catalog_to_pair(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not messages:
        messages = [{"role": "user", "content": str(row.get("prompt", ""))}]
    refs = row.get("reference_calls") or []
    return {
        "id": row.get("eval_id", row.get("id")),
        "eval_id": row.get("eval_id", row.get("id")),
        "category": row.get("category"),
        "split_role": row.get("split_role"),
        "messages": messages,
        "tools": row.get("tools") or [],
        "target": refs,
        "reference_calls": refs,
    }


def row_eval_id(row: dict[str, Any]) -> str:
    return str(row.get("eval_id", row.get("id")))


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world, local_rank


def load_tokenized_rows(tokenizer, rows: list[dict[str, Any]], *, enable_thinking: bool) -> list[dict[str, Any]]:
    tokenized = []
    for global_index, row in rows:
        pair = catalog_to_pair(row)
        prompt = encode_prompt(tokenizer, pair, enable_thinking=enable_thinking)
        target_text = format_tool_call_target(pair)
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
        prompt_ids = prompt["input_ids"][0]
        input_ids = np.concatenate(
            [prompt_ids.cpu().numpy(), target_ids.cpu().numpy()]
        ).astype(np.int64)
        tokenized.append(
            {
                "global_index": global_index,
                "row": row,
                "pair": pair,
                "input_ids": input_ids,
                "prompt_tokens": int(prompt_ids.shape[0]),
                "target_tokens": int(target_ids.shape[0]),
                "full_tokens": int(input_ids.shape[0]),
            }
        )
    return tokenized


def make_query_manifest_row(
    item: dict[str, Any],
    *,
    shard_rank: int,
    shard_world: int,
    shard_position: int,
    source_catalog: Path,
    source_catalog_sha256: str,
) -> dict[str, Any]:
    row = item["row"]
    refs = row.get("reference_calls") or []
    return {
        "global_index": int(item["global_index"]),
        "shard_rank": int(shard_rank),
        "shard_world": int(shard_world),
        "shard_position": int(shard_position),
        "eval_id": row_eval_id(row),
        "split_role": row.get("split_role"),
        "category": row.get("category"),
        "prompt_hash": row.get("prompt_hash") or stable_hash(row.get("prompt", "")),
        "tools_hash": row.get("tools_hash") or stable_hash(row.get("tools") or []),
        "reference_calls_hash": row.get("reference_calls_hash") or stable_hash(refs),
        "prompt_tokens": int(item["prompt_tokens"]),
        "target_tokens": int(item["target_tokens"]),
        "full_tokens": int(item["full_tokens"]),
        "source_catalog": str(source_catalog),
        "source_catalog_sha256": source_catalog_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world, local_rank = distributed_context()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        gpu_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_name = "cpu"

    rows = list(enumerate(read_jsonl(args.catalog)))
    if args.limit is not None:
        rows = rows[: args.limit]
    assigned = [item for item in rows if item[0] % world == rank]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(args.catalog)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = load_tokenized_rows(
        tokenizer,
        assigned,
        enable_thinking=args.enable_thinking,
    )
    if args.sort_by_length:
        tokenized.sort(key=lambda item: item["full_tokens"])

    dtype = getattr(torch, args.dtype)
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": str(device)} if device.type == "cuda" else None,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    n_layers = int(model.config.num_hidden_layers)
    d_ffn = int(model.config.intermediate_size)
    expected_total = n_layers * d_ffn
    shard_count = len(tokenized)
    stats_shape = (shard_count, len(SEGMENTS), len(STATS), n_layers, d_ffn)
    stats_path = args.out_dir / f"shard_rank{rank:02d}_stats_float16.npy"
    stats_mm = np.lib.format.open_memmap(
        stats_path,
        mode="w+",
        dtype=np.float16,
        shape=stats_shape,
    )

    query_manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_time = time.time()
    processed = 0

    def batches(items: list[dict[str, Any]], size: int):
        for start in range(0, len(items), size):
            yield start, items[start : start + size]

    for batch_start, batch in batches(tokenized, max(args.batch_size, 1)):
        batch_size = len(batch)
        max_len = max(item["full_tokens"] for item in batch)
        input_ids = torch.full(
            (batch_size, max_len),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        lengths: list[tuple[int, int, int]] = []
        for b, item in enumerate(batch):
            ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
            input_ids[b, : ids.shape[0]] = ids
            attention_mask[b, : ids.shape[0]] = 1
            prompt_tokens = int(item["prompt_tokens"])
            full_tokens = int(item["full_tokens"])
            lengths.append((prompt_tokens, full_tokens - prompt_tokens, full_tokens))

        batch_stats = torch.empty(
            (batch_size, len(SEGMENTS), len(STATS), n_layers, d_ffn),
            dtype=torch.float32,
            device=device,
        )
        hooks = []

        def make_hook(layer_idx: int):
            def hook(module, hook_args, output):
                act = hook_args[0].detach().to(torch.float32)
                for b, (prompt_tokens, target_tokens, full_tokens) in enumerate(lengths):
                    spans = (
                        (0, prompt_tokens),
                        (prompt_tokens, prompt_tokens + target_tokens),
                        (0, full_tokens),
                    )
                    for seg_idx, (start, end) in enumerate(spans):
                        if end <= start:
                            batch_stats[b, seg_idx, :, layer_idx, :].zero_()
                            continue
                        seg = act[b, start:end, :]
                        abs_seg = seg.abs()
                        batch_stats[b, seg_idx, 0, layer_idx, :] = abs_seg.mean(dim=0)
                        batch_stats[b, seg_idx, 1, layer_idx, :] = torch.sqrt(
                            (seg * seg).mean(dim=0)
                        )
                        batch_stats[b, seg_idx, 2, layer_idx, :] = abs_seg.max(dim=0).values

            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_np = batch_stats.detach().to(torch.float16).cpu().numpy()
            stats_mm[batch_start : batch_start + batch_size] = batch_np
            stats_mm.flush()
            for offset, item in enumerate(batch):
                shard_position = batch_start + offset
                query_manifest.append(
                    make_query_manifest_row(
                        item,
                        shard_rank=rank,
                        shard_world=world,
                        shard_position=shard_position,
                        source_catalog=args.catalog,
                        source_catalog_sha256=source_sha,
                    )
                )
            processed += batch_size
        except Exception as exc:  # noqa: BLE001
            for item in batch:
                failures.append(
                    {
                        "global_index": int(item["global_index"]),
                        "eval_id": row_eval_id(item["row"]),
                        "error": repr(exc),
                    }
                )
            raise
        finally:
            for hook in hooks:
                hook.remove()
            del batch_stats
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if processed % max(args.log_every, 1) == 0 or processed == shard_count:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "processed": processed,
                        "assigned": shard_count,
                        "elapsed_sec": round(elapsed, 2),
                        "queries_per_sec": round(rate, 4),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    query_manifest.sort(key=lambda row: row["shard_position"])
    query_manifest_path = args.out_dir / f"shard_rank{rank:02d}_query_manifest.jsonl"
    write_jsonl(query_manifest_path, query_manifest)

    elapsed = time.time() - start_time
    shard_manifest = {
        "artifact": "bfcl_issue9_activation_atlas_shard",
        "model": args.model,
        "dtype": args.dtype,
        "device": str(device),
        "gpu_name": gpu_name,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "source_catalog": str(args.catalog),
        "source_catalog_sha256": source_sha,
        "rows_in_catalog_scope": len(rows),
        "assigned_queries": shard_count,
        "queries_processed": processed,
        "failed_queries": len(failures),
        "failures": failures,
        "segments": list(SEGMENTS),
        "stats": list(STATS),
        "array_order": ["query", "segment", "stat", "layer", "channel"],
        "stats_shape": list(stats_shape),
        "layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": expected_total,
        "hook_point": "model.model.layers[*].mlp.down_proj input",
        "teacher_forced_format": "prompt chat template + gold <tool_call> continuation",
        "batch_size": args.batch_size,
        "enable_thinking": args.enable_thinking,
        "sort_by_length": args.sort_by_length,
        "attn_implementation": args.attn_implementation,
        "elapsed_sec": elapsed,
        "queries_per_sec": processed / elapsed if elapsed > 0 else None,
        "split_counts": dict(Counter(row.get("split_role") for _, row in rows)),
        "category_counts": dict(Counter(row.get("category") for _, row in rows)),
        "stats_path": str(stats_path),
        "query_manifest_path": str(query_manifest_path),
    }
    manifest_path = args.out_dir / f"shard_rank{rank:02d}_manifest.json"
    manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(shard_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
