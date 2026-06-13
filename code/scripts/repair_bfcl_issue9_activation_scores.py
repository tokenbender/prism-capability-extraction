#!/usr/bin/env python3
"""Repair BFCL issue #9 uint8 activation score arrays from raw stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEGMENTS = ("prompt", "target", "full")
STATS = ("mean_abs", "rms", "max_abs")


def decile_thresholds(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return np.zeros(9, dtype=np.float32)
    return np.quantile(finite, np.arange(0.1, 1.0, 0.1)).astype(np.float32)


def score(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out = 1 + (flat[:, None] > thresholds[None, :]).sum(axis=1)
    return np.clip(out, 1, 10).astype(np.uint8).reshape(values.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--plane-index", type=int, required=True, help="0..8 for segment/stat")
    args = parser.parse_args()

    seg_idx = args.plane_index // len(STATS)
    stat_idx = args.plane_index % len(STATS)
    segment = SEGMENTS[seg_idx]
    stat = STATS[stat_idx]

    stats = np.load(args.atlas_dir / "activation_stats_float16.npy", mmap_mode="r")
    local_scores = np.load(args.atlas_dir / "activation_scores_local_uint8.npy", mmap_mode="r+")
    global_scores = np.load(args.atlas_dir / "activation_scores_global_uint8.npy", mmap_mode="r+")

    thresholds = decile_thresholds(stats[:, seg_idx, stat_idx])
    n_queries = stats.shape[0]
    for qidx in range(n_queries):
        values = stats[qidx, seg_idx, stat_idx]
        local_t = decile_thresholds(values)
        local_scores[qidx, seg_idx, stat_idx] = score(values, local_t)
        global_scores[qidx, seg_idx, stat_idx] = score(values, thresholds)
    local_scores.flush()
    global_scores.flush()

    out_dir = args.atlas_dir / "score_repair_thresholds"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / f"plane_{args.plane_index:02d}_{segment}_{stat}.npz",
        global_thresholds=thresholds,
        segment=segment,
        stat=stat,
    )
    print(
        json.dumps(
            {
                "plane_index": args.plane_index,
                "segment": segment,
                "stat": stat,
                "queries": n_queries,
                "local_min": int(local_scores[:, seg_idx, stat_idx].min()),
                "local_max": int(local_scores[:, seg_idx, stat_idx].max()),
                "global_min": int(global_scores[:, seg_idx, stat_idx].min()),
                "global_max": int(global_scores[:, seg_idx, stat_idx].max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
