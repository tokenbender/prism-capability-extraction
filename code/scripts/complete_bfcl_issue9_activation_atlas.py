#!/usr/bin/env python3
"""Complete BFCL issue #9 atlas artifacts from existing stats/scores arrays.

Use this after the large activation arrays already exist. It avoids recomputing
the expensive decile score arrays and focuses on metadata, thresholds,
top-channel inspection rows, summary heatmaps, report, and checksums.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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


def decile_thresholds(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return np.zeros(9, dtype=np.float32)
    return np.quantile(finite, np.arange(0.1, 1.0, 0.1)).astype(np.float32)


def load_failure_metadata(path: Path | None) -> dict[str, dict[str, list[str]]]:
    if path is None or not path.exists():
        return {}
    meta: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"repair_buckets": set(), "primary_failure_types": set()}
    )
    for row in read_jsonl(path):
        eval_id = str(row.get("eval_id"))
        if not eval_id:
            continue
        meta[eval_id]["repair_buckets"].update(row.get("repair_buckets") or [])
        if row.get("primary_failure_type"):
            meta[eval_id]["primary_failure_types"].add(str(row["primary_failure_type"]))
    return {
        eval_id: {key: sorted(values) for key, values in fields.items()}
        for eval_id, fields in meta.items()
    }


def group_specs(
    query_rows: list[dict[str, Any]],
    failure_meta: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(query_rows):
        if row.get("split_role"):
            groups[f"split:{row['split_role']}"].add(idx)
        if row.get("category"):
            groups[f"category:{row['category']}"].add(idx)
        fields = failure_meta.get(row["eval_id"], {})
        for bucket in fields.get("repair_buckets", []):
            groups[f"repair_bucket:{bucket}"].add(idx)
        for failure_type in fields.get("primary_failure_types", []):
            groups[f"primary_failure_type:{failure_type}"].add(idx)
    return [
        {"group": name, "query_indices": sorted(indices), "query_count": len(indices)}
        for name, indices in sorted(groups.items())
        if indices
    ]


def top_rows_for_range(
    *,
    stats: np.ndarray,
    local_scores: np.ndarray,
    global_scores: np.ndarray,
    query_rows: list[dict[str, Any]],
    d_ffn: int,
    start: int,
    end: int,
    top_n: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for qidx in range(start, end):
        row = query_rows[qidx]
        for seg_idx, segment in enumerate(SEGMENTS):
            for stat_idx, stat in enumerate(STATS):
                values = np.asarray(stats[qidx, seg_idx, stat_idx], dtype=np.float32).reshape(-1)
                n = min(top_n, values.size)
                idxs = np.argpartition(values, -n)[-n:]
                idxs = idxs[np.argsort(values[idxs])[::-1]]
                local_flat = local_scores[qidx, seg_idx, stat_idx].reshape(-1)
                global_flat = global_scores[qidx, seg_idx, stat_idx].reshape(-1)
                out.append(
                    {
                        "eval_id": row["eval_id"],
                        "global_index": int(row["global_index"]),
                        "segment": segment,
                        "stat": stat,
                        "top": [
                            {
                                "rank": rank + 1,
                                "layer": int(idx // d_ffn),
                                "channel": int(idx % d_ffn),
                                "value": float(values[idx]),
                                "local_score": int(local_flat[idx]),
                                "global_score": int(global_flat[idx]),
                            }
                            for rank, idx in enumerate(idxs)
                        ],
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--failure-matrix", type=Path)
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--top-shard", type=int)
    parser.add_argument("--top-num-shards", type=int, default=1)
    parser.add_argument("--skip-top", action="store_true")
    parser.add_argument("--only-top", action="store_true")
    parser.add_argument("--merge-top-shards", action="store_true")
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()

    manifest_paths = sorted(args.shard_dir.glob("shard_rank*_manifest.json"))
    shard_infos = [json.loads(path.read_text()) for path in manifest_paths]
    if not shard_infos:
        raise FileNotFoundError(f"no shard manifests in {args.shard_dir}")

    query_rows = read_jsonl(args.atlas_dir / "query_manifest.jsonl")
    query_rows.sort(key=lambda row: int(row["global_index"]))
    n_queries = len(query_rows)
    n_layers = int(shard_infos[0]["layers"])
    d_ffn = int(shard_infos[0]["d_ffn"])
    full_shape = (n_queries, len(SEGMENTS), len(STATS), n_layers, d_ffn)

    stats = np.load(args.atlas_dir / "activation_stats_float16.npy", mmap_mode="r")
    local_scores = np.load(args.atlas_dir / "activation_scores_local_uint8.npy", mmap_mode="r")
    global_scores = np.load(args.atlas_dir / "activation_scores_global_uint8.npy", mmap_mode="r")
    if tuple(stats.shape) != full_shape:
        raise ValueError(f"stats shape {stats.shape} != {full_shape}")
    if tuple(local_scores.shape) != full_shape or tuple(global_scores.shape) != full_shape:
        raise ValueError("score array shapes do not match stats")

    threshold_path = args.atlas_dir / "activation_decile_thresholds.npz"
    if not threshold_path.exists():
        global_thresholds = np.zeros((len(SEGMENTS), len(STATS), 9), dtype=np.float32)
        local_thresholds_sample = np.zeros((min(n_queries, 32), len(SEGMENTS), len(STATS), 9), dtype=np.float32)
        for seg_idx in range(len(SEGMENTS)):
            for stat_idx in range(len(STATS)):
                global_thresholds[seg_idx, stat_idx] = decile_thresholds(stats[:, seg_idx, stat_idx])
                for qidx in range(local_thresholds_sample.shape[0]):
                    local_thresholds_sample[qidx, seg_idx, stat_idx] = decile_thresholds(
                        stats[qidx, seg_idx, stat_idx]
                    )
        np.savez_compressed(
            threshold_path,
            global_thresholds=global_thresholds,
            local_thresholds_sample_first32=local_thresholds_sample,
            segments=np.array(SEGMENTS),
            stats=np.array(STATS),
            note="Full local thresholds are implicit in activation_scores_local_uint8; first 32 are stored as an audit sample.",
        )

    if not args.skip_top:
        top_dir = args.atlas_dir / "top_shards"
        top_dir.mkdir(exist_ok=True)
        shard = args.top_shard if args.top_shard is not None else 0
        total = max(args.top_num_shards, 1)
        start = (n_queries * shard) // total
        end = (n_queries * (shard + 1)) // total
        rows = top_rows_for_range(
            stats=stats,
            local_scores=local_scores,
            global_scores=global_scores,
            query_rows=query_rows,
            d_ffn=d_ffn,
            start=start,
            end=end,
            top_n=args.top_n,
        )
        write_jsonl(top_dir / f"top_channels_shard{shard:02d}_of{total:02d}.jsonl", rows)
        if args.only_top:
            print(
                json.dumps(
                    {
                        "top_shard": shard,
                        "top_num_shards": total,
                        "start": start,
                        "end": end,
                        "rows": len(rows),
                        "output": str(top_dir / f"top_channels_shard{shard:02d}_of{total:02d}.jsonl"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

    if args.merge_top_shards:
        top_files = sorted((args.atlas_dir / "top_shards").glob("top_channels_shard*_of*.jsonl"))
        with (args.atlas_dir / "top_channels_per_query.jsonl").open("w", encoding="utf-8") as out:
            for path in top_files:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        out.write(line)

    failure_meta = load_failure_metadata(args.failure_matrix)
    enriched = []
    for row in query_rows:
        merged = dict(row)
        merged.update(failure_meta.get(row["eval_id"], {}))
        enriched.append(merged)
    write_jsonl(args.atlas_dir / "query_manifest_with_failure_metadata.jsonl", enriched)

    specs = group_specs(query_rows, failure_meta)
    group_stats = np.zeros((len(specs), len(SEGMENTS), len(STATS), n_layers, d_ffn), dtype=np.float16)
    for group_idx, spec in enumerate(specs):
        group_stats[group_idx] = stats[spec["query_indices"]].mean(axis=0, dtype=np.float32).astype(np.float16)
    np.savez_compressed(
        args.atlas_dir / "bucket_summary_heatmaps.npz",
        group_stats_mean_float16=group_stats,
        group_names=np.array([spec["group"] for spec in specs]),
        group_query_counts=np.array([spec["query_count"] for spec in specs], dtype=np.int32),
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
    )
    (args.atlas_dir / "bucket_summary_manifest.json").write_text(
        json.dumps({"groups": specs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    split_counts = Counter(row.get("split_role") for row in query_rows)
    category_counts = Counter(row.get("category") for row in query_rows)
    source_catalog = shard_infos[0]["source_catalog"]
    source_catalog_sha = shard_infos[0]["source_catalog_sha256"]
    manifest = {
        "artifact": "bfcl_issue9_activation_atlas",
        "source_catalog": source_catalog,
        "source_catalog_sha256": source_catalog_sha,
        "queries_processed": n_queries,
        "failed_queries": int(sum(info.get("failed_queries", 0) for info in shard_infos)),
        "segments": list(SEGMENTS),
        "stats": list(STATS),
        "array_order": ["query", "segment", "stat", "layer", "channel"],
        "stats_shape": list(full_shape),
        "score_shape": list(full_shape),
        "layers": n_layers,
        "d_ffn": d_ffn,
        "total_mlp_channels": n_layers * d_ffn,
        "hook_point": "model.model.layers[*].mlp.down_proj input",
        "teacher_forced_format": "prompt chat template + gold <tool_call> continuation",
        "score_dtype": "uint8",
        "stats_dtype": "float16",
        "local_score_rule": "1 + count(value > per-query decile thresholds); ties stay in the lower bin; all-equal arrays score 1",
        "global_score_rule": "1 + count(value > corpus decile thresholds for matching segment/stat); ties stay in the lower bin; all-equal arrays score 1",
        "split_counts": dict(split_counts),
        "category_counts": dict(category_counts),
        "shard_manifests": [str(path) for path in manifest_paths],
    }
    (args.atlas_dir / "activation_atlas_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    file_sizes = {
        str(path.relative_to(args.atlas_dir)): path.stat().st_size
        for path in sorted(args.atlas_dir.rglob("*"))
        if path.is_file()
    }
    report = [
        "# BFCL Issue #9 Activation Atlas",
        "",
        "This artifact is a descriptive per-query MLP activation atlas for the BFCL single-call slice.",
        "It is intended as the visual/analysis board for later sparse-circuit search.",
        "",
        "## Boundary",
        "",
        "This is activation, not attribution. It records which MLP channels light up under the full original model; it does not by itself prove which channels causally determine the answer.",
        "",
        "This is not training, not collimation, and not final mask selection.",
        "",
        "This atlas includes train, calibration, validation, and heldout rows for visualization. It is therefore descriptive data, not a sealed heldout benchmark surface.",
        "",
        "## Shape",
        "",
        f"- queries: `{n_queries}`",
        f"- shape: `{full_shape}` with order `query, segment, stat, layer, channel`",
        f"- MLP channels: `{n_layers} x {d_ffn} = {n_layers * d_ffn}`",
        f"- source catalog sha256: `{source_catalog_sha}`",
        f"- split counts: `{dict(split_counts)}`",
        f"- category counts: `{dict(category_counts)}`",
        "",
        "Segments: " + ", ".join(f"`{x}`" for x in SEGMENTS) + ".",
        "Stats: " + ", ".join(f"`{x}`" for x in STATS) + ".",
        "",
        "## Files",
        "",
        "| File | Meaning |",
        "| --- | --- |",
        "| `query_manifest.jsonl` | one row per query with eval ID, split/category, token counts, and source hashes |",
        "| `query_manifest_with_failure_metadata.jsonl` | query manifest enriched with #8 failure buckets where available |",
        "| `activation_stats_float16.npy` | raw aggregated activation stats, shape `query x segment x stat x layer x channel` |",
        "| `activation_scores_local_uint8.npy` | per-query decile heatmap scores in `[1, 10]` |",
        "| `activation_scores_global_uint8.npy` | corpus-global decile heatmap scores in `[1, 10]` |",
        "| `activation_decile_thresholds.npz` | corpus-global thresholds plus local-threshold audit sample |",
        "| `top_channels_per_query.jsonl` | top hot channels per query/stat/segment for inspection |",
        "| `bucket_summary_heatmaps.npz` | split/category/failure-bucket aggregate heatmaps |",
        "| `activation_atlas_manifest.json` | machine-readable provenance, shapes, rules, and file paths |",
        "| `checksums.sha256` | checksums for preserved artifacts |",
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "import numpy as np",
        "manifest = json.load(open('activation_atlas_manifest.json'))",
        "stats = np.load('activation_stats_float16.npy', mmap_mode='r')",
        "local_scores = np.load('activation_scores_local_uint8.npy', mmap_mode='r')",
        "global_scores = np.load('activation_scores_global_uint8.npy', mmap_mode='r')",
        "print(stats.shape, stats.dtype)",
        "print(local_scores.min(), local_scores.max())",
        "```",
        "",
        "Array indices are documented in `activation_atlas_manifest.json`: `query, segment, stat, layer, channel`.",
        "",
        "## Score Rules",
        "",
        "- Local scores use per-query deciles for the matching segment/stat.",
        "- Global scores use corpus-wide deciles for the matching segment/stat.",
        "- Scores are `uint8` integers in `[1, 10]`.",
        "- Ties stay in the lower bin; all-equal arrays score `1`.",
        "",
        "## Future Use",
        "",
        "A later mask-search issue can use this as a board for candidate generation, visualization, clustering, or failure-bucket comparison. It should define a fresh leakage policy before using heldout-derived heatmaps for any selection claim.",
        "",
        "## File Sizes",
        "",
        "```json",
        json.dumps(file_sizes, indent=2, sort_keys=True),
        "```",
    ]
    text = "\n".join(report) + "\n"
    (args.atlas_dir / "final_report.md").write_text(text, encoding="utf-8")
    (args.atlas_dir / "README.md").write_text(text, encoding="utf-8")

    if args.write_checksums:
        lines = []
        for path in sorted(p for p in args.atlas_dir.rglob("*") if p.is_file()):
            if path.name == "checksums.sha256":
                continue
            lines.append(f"{sha256_file(path)}  {path.relative_to(args.atlas_dir)}")
        (args.atlas_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
