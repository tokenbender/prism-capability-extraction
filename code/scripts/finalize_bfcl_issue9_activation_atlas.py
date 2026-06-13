#!/usr/bin/env python3
"""Finalize BFCL issue #9 activation shards into atlas artifacts."""

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


def decile_scores(values: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        if thresholds is None:
            thresholds = np.zeros(9, dtype=np.float32)
        return np.ones(flat.shape, dtype=np.uint8).reshape(values.shape), thresholds
    if thresholds is None:
        thresholds = np.quantile(finite, np.arange(0.1, 1.0, 0.1)).astype(np.float32)
    scores = 1 + (flat[:, None] > thresholds[None, :]).sum(axis=1)
    return np.clip(scores, 1, 10).astype(np.uint8).reshape(values.shape), thresholds


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


def build_group_specs(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--failure-matrix", type=Path)
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_manifests = sorted(args.shard_dir.glob("shard_rank*_manifest.json"))
    if not shard_manifests:
        raise FileNotFoundError(f"no shard manifests found in {args.shard_dir}")

    shard_infos = [json.loads(path.read_text()) for path in shard_manifests]
    n_layers = int(shard_infos[0]["layers"])
    d_ffn = int(shard_infos[0]["d_ffn"])
    source_catalog = shard_infos[0]["source_catalog"]
    source_catalog_sha = shard_infos[0]["source_catalog_sha256"]

    query_rows: list[dict[str, Any]] = []
    shard_arrays: dict[int, np.ndarray] = {}
    for info in shard_infos:
        rank = int(info["rank"])
        stats_path = Path(info["stats_path"])
        if not stats_path.is_absolute():
            stats_path = args.shard_dir / stats_path.name
        shard_arrays[rank] = np.load(stats_path, mmap_mode="r")
        qpath = Path(info["query_manifest_path"])
        if not qpath.is_absolute():
            qpath = args.shard_dir / qpath.name
        query_rows.extend(read_jsonl(qpath))

    query_rows.sort(key=lambda row: int(row["global_index"]))
    n_queries = len(query_rows)
    if n_queries == 0:
        raise ValueError("no query rows found")
    if len({row["eval_id"] for row in query_rows}) != n_queries:
        raise ValueError("duplicate eval_id entries found in query manifest")

    full_shape = (n_queries, len(SEGMENTS), len(STATS), n_layers, d_ffn)
    stats_out = args.out_dir / "activation_stats_float16.npy"
    stats_mm = np.lib.format.open_memmap(
        stats_out,
        mode="w+",
        dtype=np.float16,
        shape=full_shape,
    )

    for out_idx, row in enumerate(query_rows):
        rank = int(row["shard_rank"])
        shard_position = int(row["shard_position"])
        stats_mm[out_idx] = shard_arrays[rank][shard_position]
    stats_mm.flush()

    query_manifest = args.out_dir / "query_manifest.jsonl"
    write_jsonl(query_manifest, query_rows)

    local_scores_path = args.out_dir / "activation_scores_local_uint8.npy"
    global_scores_path = args.out_dir / "activation_scores_global_uint8.npy"
    local_scores = np.lib.format.open_memmap(
        local_scores_path,
        mode="w+",
        dtype=np.uint8,
        shape=full_shape,
    )
    global_scores = np.lib.format.open_memmap(
        global_scores_path,
        mode="w+",
        dtype=np.uint8,
        shape=full_shape,
    )

    local_thresholds = np.zeros((n_queries, len(SEGMENTS), len(STATS), 9), dtype=np.float32)
    global_thresholds = np.zeros((len(SEGMENTS), len(STATS), 9), dtype=np.float32)

    for seg_idx in range(len(SEGMENTS)):
        for stat_idx in range(len(STATS)):
            thresholds = decile_thresholds(stats_mm[:, seg_idx, stat_idx, :, :])
            global_thresholds[seg_idx, stat_idx] = thresholds
            for qidx in range(n_queries):
                local, local_t = decile_scores(stats_mm[qidx, seg_idx, stat_idx, :, :])
                glob, _ = decile_scores(
                    stats_mm[qidx, seg_idx, stat_idx, :, :],
                    thresholds=thresholds,
                )
                local_scores[qidx, seg_idx, stat_idx] = local
                global_scores[qidx, seg_idx, stat_idx] = glob
                local_thresholds[qidx, seg_idx, stat_idx] = local_t
            local_scores.flush()
            global_scores.flush()

    np.savez_compressed(
        args.out_dir / "activation_decile_thresholds.npz",
        local_thresholds=local_thresholds,
        global_thresholds=global_thresholds,
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
    )

    top_rows = []
    for qidx, row in enumerate(query_rows):
        for seg_idx, segment in enumerate(SEGMENTS):
            for stat_idx, stat in enumerate(STATS):
                values = np.asarray(stats_mm[qidx, seg_idx, stat_idx]).reshape(-1)
                n = min(args.top_n, values.size)
                top_idx = np.argpartition(values, -n)[-n:]
                top_idx = top_idx[np.argsort(values[top_idx])[::-1]]
                top_rows.append(
                    {
                        "eval_id": row["eval_id"],
                        "global_index": int(row["global_index"]),
                        "segment": segment,
                        "stat": stat,
                        "top": [
                            {
                                "rank": int(rank + 1),
                                "layer": int(idx // d_ffn),
                                "channel": int(idx % d_ffn),
                                "value": float(values[idx]),
                                "local_score": int(local_scores[qidx, seg_idx, stat_idx].reshape(-1)[idx]),
                                "global_score": int(global_scores[qidx, seg_idx, stat_idx].reshape(-1)[idx]),
                            }
                            for rank, idx in enumerate(top_idx)
                        ],
                    }
                )
    write_jsonl(args.out_dir / "top_channels_per_query.jsonl", top_rows)

    failure_meta = load_failure_metadata(args.failure_matrix)
    enriched_rows = []
    for row in query_rows:
        enriched = dict(row)
        enriched.update(failure_meta.get(row["eval_id"], {}))
        enriched_rows.append(enriched)
    write_jsonl(args.out_dir / "query_manifest_with_failure_metadata.jsonl", enriched_rows)

    group_specs = build_group_specs(query_rows, failure_meta)
    group_stats = np.zeros((len(group_specs), len(SEGMENTS), len(STATS), n_layers, d_ffn), dtype=np.float16)
    for group_idx, spec in enumerate(group_specs):
        group_stats[group_idx] = stats_mm[spec["query_indices"]].mean(axis=0, dtype=np.float32).astype(np.float16)
    np.savez_compressed(
        args.out_dir / "bucket_summary_heatmaps.npz",
        group_stats_mean_float16=group_stats,
        group_names=np.array([spec["group"] for spec in group_specs]),
        group_query_counts=np.array([spec["query_count"] for spec in group_specs], dtype=np.int32),
        segments=np.array(SEGMENTS),
        stats=np.array(STATS),
    )
    (args.out_dir / "bucket_summary_manifest.json").write_text(
        json.dumps({"groups": group_specs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    split_counts = Counter(row.get("split_role") for row in query_rows)
    category_counts = Counter(row.get("category") for row in query_rows)
    manifest = {
        "artifact": "bfcl_issue9_activation_atlas",
        "source_catalog": source_catalog,
        "source_catalog_sha256": source_catalog_sha,
        "queries_processed": n_queries,
        "failed_queries": 0,
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
        "shard_manifests": [str(path) for path in shard_manifests],
        "files": {
            "query_manifest": str(query_manifest),
            "query_manifest_with_failure_metadata": str(args.out_dir / "query_manifest_with_failure_metadata.jsonl"),
            "activation_stats_float16": str(stats_out),
            "activation_scores_local_uint8": str(local_scores_path),
            "activation_scores_global_uint8": str(global_scores_path),
            "activation_decile_thresholds": str(args.out_dir / "activation_decile_thresholds.npz"),
            "top_channels_per_query": str(args.out_dir / "top_channels_per_query.jsonl"),
            "bucket_summary_heatmaps": str(args.out_dir / "bucket_summary_heatmaps.npz"),
            "bucket_summary_manifest": str(args.out_dir / "bucket_summary_manifest.json"),
        },
    }
    manifest_path = args.out_dir / "activation_atlas_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    file_sizes = {
        str(path.relative_to(args.out_dir)): path.stat().st_size
        for path in sorted(args.out_dir.rglob("*"))
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
        "| `activation_decile_thresholds.npz` | local and global score thresholds |",
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
        "",
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
        "",
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if args.write_checksums:
        checksum_lines = []
        for path in sorted(p for p in args.out_dir.rglob("*") if p.is_file()):
            if path.name == "checksums.sha256":
                continue
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(args.out_dir)}")
        (args.out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
