#!/usr/bin/env python3
"""Validate BFCL issue #9 activation atlas artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=1007)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-d-ffn", type=int, default=12288)
    args = parser.parse_args()

    manifest_path = args.atlas_dir / "activation_atlas_manifest.json"
    query_manifest_path = args.atlas_dir / "query_manifest.jsonl"
    stats_path = args.atlas_dir / "activation_stats_float16.npy"
    local_scores_path = args.atlas_dir / "activation_scores_local_uint8.npy"
    global_scores_path = args.atlas_dir / "activation_scores_global_uint8.npy"
    top_path = args.atlas_dir / "top_channels_per_query.jsonl"

    manifest = json.loads(manifest_path.read_text())
    query_rows = read_jsonl(query_manifest_path)
    eval_ids = [row["eval_id"] for row in query_rows]
    duplicates = [eval_id for eval_id, count in Counter(eval_ids).items() if count > 1]

    stats = np.load(stats_path, mmap_mode="r")
    local_scores = np.load(local_scores_path, mmap_mode="r")
    global_scores = np.load(global_scores_path, mmap_mode="r")

    expected_shape = (
        args.expected_queries,
        3,
        3,
        args.expected_layers,
        args.expected_d_ffn,
    )
    checks = {
        "manifest_exists": manifest_path.exists(),
        "query_manifest_exists": query_manifest_path.exists(),
        "query_count_ok": len(query_rows) == args.expected_queries,
        "duplicate_eval_ids": duplicates,
        "stats_shape": list(stats.shape),
        "stats_shape_ok": tuple(stats.shape) == expected_shape,
        "stats_dtype": str(stats.dtype),
        "stats_dtype_ok": str(stats.dtype) == "float16",
        "local_scores_shape": list(local_scores.shape),
        "local_scores_shape_ok": tuple(local_scores.shape) == expected_shape,
        "local_scores_dtype": str(local_scores.dtype),
        "local_scores_dtype_ok": str(local_scores.dtype) == "uint8",
        "global_scores_shape": list(global_scores.shape),
        "global_scores_shape_ok": tuple(global_scores.shape) == expected_shape,
        "global_scores_dtype": str(global_scores.dtype),
        "global_scores_dtype_ok": str(global_scores.dtype) == "uint8",
        "local_scores_min": int(local_scores.min()),
        "local_scores_max": int(local_scores.max()),
        "global_scores_min": int(global_scores.min()),
        "global_scores_max": int(global_scores.max()),
        "scores_range_ok": (
            int(local_scores.min()) >= 1
            and int(local_scores.max()) <= 10
            and int(global_scores.min()) >= 1
            and int(global_scores.max()) <= 10
        ),
        "manifest_queries_processed": manifest.get("queries_processed"),
        "manifest_failed_queries": manifest.get("failed_queries"),
        "manifest_layers": manifest.get("layers"),
        "manifest_d_ffn": manifest.get("d_ffn"),
        "manifest_total_mlp_channels": manifest.get("total_mlp_channels"),
        "split_counts": dict(Counter(row.get("split_role") for row in query_rows)),
        "category_counts": dict(Counter(row.get("category") for row in query_rows)),
        "top_channels_file_exists": top_path.exists(),
    }
    if top_path.exists():
        with top_path.open("r", encoding="utf-8") as handle:
            first_top = json.loads(handle.readline())
        checks["sample_top_channels"] = first_top

    ok = (
        checks["query_count_ok"]
        and not duplicates
        and checks["stats_shape_ok"]
        and checks["stats_dtype_ok"]
        and checks["local_scores_shape_ok"]
        and checks["local_scores_dtype_ok"]
        and checks["global_scores_shape_ok"]
        and checks["global_scores_dtype_ok"]
        and checks["scores_range_ok"]
        and checks["manifest_queries_processed"] == args.expected_queries
        and checks["manifest_failed_queries"] == 0
        and checks["manifest_layers"] == args.expected_layers
        and checks["manifest_d_ffn"] == args.expected_d_ffn
        and checks["manifest_total_mlp_channels"] == args.expected_layers * args.expected_d_ffn
        and checks["top_channels_file_exists"]
    )
    checks["ok"] = ok
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
