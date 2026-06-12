#!/usr/bin/env python3
"""Build candidate BFCL issue #8 attribution masks from bucket attributions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


CORE_BUCKETS = [
    "r0_value_recovery",
    "r0_function_selection",
    "r0_schema_completion",
    "r0_sql_domain_control",
    "r0_normalization_pilots",
    "r0_misc_failure",
]

CATEGORY_CONTROLS = [
    "r0_category_control_exec_simple",
    "r0_category_control_java",
    "r0_category_control_javascript",
    "r0_category_control_live_simple",
    "r0_category_control_simple",
    "r0_category_control_sql",
]

DEFAULT_BUDGETS = [80000, 120000, 160000, 200000, 240000]


def load_scores(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["mlp_scores"], dtype=np.float32)


def write_scores(path: Path, scores: np.ndarray, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mlp_scores=scores.astype(np.float32), **metadata)


def normalize(scores: np.ndarray) -> np.ndarray:
    values = scores.astype(np.float32, copy=True)
    positive = values[values > 0]
    if positive.size == 0:
        return values
    scale = float(np.percentile(positive, 99.9))
    if scale <= 0:
        scale = float(positive.max())
    if scale <= 0:
        return values
    return np.clip(values / scale, 0.0, 10.0)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    flat = scores.reshape(-1)
    k = min(k, flat.size)
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(flat, -k)[-k:]
    return idx[np.argsort(flat[idx])[::-1]]


def rank_encoded_scores(shape: tuple[int, int], selected: list[int]) -> np.ndarray:
    scores = np.zeros(shape[0] * shape[1], dtype=np.float32)
    high = float(len(selected) + 1)
    for rank, idx in enumerate(selected):
        if scores[idx] == 0:
            scores[idx] = high - rank
    return scores.reshape(shape)


def weighted_sum(score_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for run_id, weight in weights.items():
        if run_id not in score_map:
            continue
        value = normalize(score_map[run_id]) * float(weight)
        out = value if out is None else out + value
    if out is None:
        first = next(iter(score_map.values()))
        out = np.zeros_like(first)
    return out.astype(np.float32)


def max_scores(score_map: dict[str, np.ndarray], run_ids: list[str]) -> np.ndarray:
    arrays = [normalize(score_map[run_id]) for run_id in run_ids if run_id in score_map]
    if not arrays:
        first = next(iter(score_map.values()))
        return np.zeros_like(first)
    return np.maximum.reduce(arrays).astype(np.float32)


def union_scores(
    score_map: dict[str, np.ndarray],
    *,
    run_ids: list[str],
    budget: int,
    allocation: dict[str, float],
    fill_run: str,
) -> np.ndarray:
    shape = next(iter(score_map.values())).shape
    selected: list[int] = []
    selected_set: set[int] = set()
    for run_id in run_ids:
        if run_id not in score_map:
            continue
        quota = max(1, int(round(budget * allocation.get(run_id, 0.0))))
        for idx in topk_indices(score_map[run_id], quota):
            item = int(idx)
            if item not in selected_set:
                selected.append(item)
                selected_set.add(item)
            if len(selected) >= budget:
                break
        if len(selected) >= budget:
            break
    if len(selected) < budget and fill_run in score_map:
        for idx in topk_indices(score_map[fill_run], budget):
            item = int(idx)
            if item not in selected_set:
                selected.append(item)
                selected_set.add(item)
            if len(selected) >= budget:
                break
    return rank_encoded_scores(shape, selected[:budget])


def copy_candidate(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def jaccard_for(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> dict[str, Any]:
    a = set(int(idx) for idx in topk_indices(scores_a, k))
    b = set(int(idx) for idx in topk_indices(scores_b, k))
    inter = len(a & b)
    union = len(a | b)
    return {
        "k": k,
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relp-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", default=[])
    args = parser.parse_args()

    budgets = args.budget or DEFAULT_BUDGETS
    relp_paths: dict[str, Path] = {}
    for relp_dir in args.relp_dir:
        for path in relp_dir.glob("*_relp.npz"):
            run_id = path.name.removesuffix("_relp.npz")
            relp_paths[run_id] = path

    score_map = {run_id: load_scores(path) for run_id, path in sorted(relp_paths.items())}
    if "r0_global_decision_eligible" not in score_map:
        raise ValueError("missing r0_global_decision_eligible attribution")

    candidate_dir = args.out_dir / "candidate_attributions"
    candidates: list[dict[str, Any]] = []

    for run_id, path in sorted(relp_paths.items()):
        if run_id == "r0_global_decision_eligible" or run_id in CORE_BUCKETS or run_id in CATEGORY_CONTROLS:
            candidate_id = run_id.removeprefix("r0_")
            dst = candidate_dir / f"{candidate_id}.npz"
            copy_candidate(path, dst)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "kind": "single_attribution",
                    "path": str(dst),
                    "source_runs": [run_id],
                }
            )

    weighted_core = {
        "r0_global_decision_eligible": 0.75,
        "r0_value_recovery": 1.0,
        "r0_function_selection": 1.0,
        "r0_schema_completion": 1.0,
        "r0_sql_domain_control": 0.5,
        "r0_normalization_pilots": 0.35,
        "r0_misc_failure": 0.35,
    }
    failure_only = {
        "r0_value_recovery": 1.0,
        "r0_function_selection": 1.0,
        "r0_schema_completion": 1.0,
        "r0_sql_domain_control": 0.6,
        "r0_normalization_pilots": 0.4,
        "r0_misc_failure": 0.5,
    }
    category_weights = {run_id: 1.0 for run_id in CATEGORY_CONTROLS}

    combos = {
        "weighted_core": weighted_sum(score_map, weighted_core),
        "failure_weighted_no_global": weighted_sum(score_map, failure_only),
        "failure_max_no_global": max_scores(score_map, CORE_BUCKETS),
        "category_weighted_control": weighted_sum(score_map, category_weights),
    }
    combo_sources = {
        "weighted_core": list(weighted_core),
        "failure_weighted_no_global": list(failure_only),
        "failure_max_no_global": CORE_BUCKETS,
        "category_weighted_control": CATEGORY_CONTROLS,
    }
    for candidate_id, scores in combos.items():
        dst = candidate_dir / f"{candidate_id}.npz"
        write_scores(dst, scores, candidate_id=candidate_id)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "kind": "combined_scores",
                "path": str(dst),
                "source_runs": combo_sources[candidate_id],
            }
        )

    even_allocation = {run_id: 1.0 / len(CORE_BUCKETS) for run_id in CORE_BUCKETS}
    weighted_allocation = {
        "r0_value_recovery": 0.26,
        "r0_function_selection": 0.24,
        "r0_schema_completion": 0.24,
        "r0_sql_domain_control": 0.10,
        "r0_normalization_pilots": 0.06,
        "r0_misc_failure": 0.10,
    }
    for budget in budgets:
        for label, allocation in (
            ("bucket_union_even", even_allocation),
            ("bucket_union_weighted", weighted_allocation),
        ):
            candidate_id = f"{label}_k{budget}"
            scores = union_scores(
                score_map,
                run_ids=CORE_BUCKETS,
                budget=budget,
                allocation=allocation,
                fill_run="r0_global_decision_eligible",
            )
            dst = candidate_dir / f"{candidate_id}.npz"
            write_scores(dst, scores, candidate_id=candidate_id, budget=budget)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "kind": "rank_encoded_union",
                    "path": str(dst),
                    "source_runs": CORE_BUCKETS,
                    "budget": budget,
                    "allocation": allocation,
                }
            )

    overlap_rows = []
    compare_runs = ["r0_global_decision_eligible"] + CORE_BUCKETS + CATEGORY_CONTROLS
    for k in budgets:
        for i, left in enumerate(compare_runs):
            if left not in score_map:
                continue
            for right in compare_runs[i + 1 :]:
                if right not in score_map:
                    continue
                item = jaccard_for(score_map[left], score_map[right], k)
                item.update({"left": left, "right": right})
                overlap_rows.append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "overlap_topk.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "left", "right", "intersection", "union", "jaccard"])
        writer.writeheader()
        writer.writerows(overlap_rows)

    manifest = {
        "budgets": budgets,
        "source_attributions": {run_id: str(path) for run_id, path in sorted(relp_paths.items())},
        "candidates": candidates,
        "overlap_csv": str(args.out_dir / "overlap_topk.csv"),
    }
    (args.out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = ["# BFCL Issue #8 Candidate Mask Summary", ""]
    lines.append(f"Source attributions: `{len(score_map)}`")
    lines.append(f"Candidates: `{len(candidates)}`")
    lines.append("")
    lines.append("| Candidate | Kind | Source runs |")
    lines.append("|---|---|---:|")
    for candidate in candidates:
        lines.append(
            f"| `{candidate['candidate_id']}` | {candidate['kind']} | {len(candidate.get('source_runs', []))} |"
        )
    (args.out_dir / "candidate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
