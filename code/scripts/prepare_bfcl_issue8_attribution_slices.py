#!/usr/bin/env python3
"""Prepare BFCL issue #8 attribution pair slices.

The issue #8 corpus is keyed by stable BFCL eval IDs. This script turns the
published catalog plus failure matrix into runnable pair files for
``bfcl_direct_qwen3.py relp-attribute`` / ``eval-mask`` while preserving the
heldout split boundary.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DECISION_SPLITS = {"train", "calibration", "validation"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def catalog_to_pair(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row.get("prompt")
    messages = row.get("messages")
    if not messages:
        messages = [{"role": "user", "content": prompt}]
    reference_calls = row.get("reference_calls") or []
    return {
        "id": row["eval_id"],
        "category": row.get("category"),
        "split_role": row.get("split_role"),
        "messages": messages,
        "tools": row.get("tools") or [],
        "target": reference_calls,
        "reference_calls": reference_calls,
    }


def has_any(values: list[str], wanted: set[str]) -> bool:
    return bool(set(values or []) & wanted)


def build_run_filters(categories: set[str]) -> dict[str, Callable[[dict[str, Any]], bool]]:
    filters: dict[str, Callable[[dict[str, Any]], bool]] = {
        "r0_global_decision_eligible": lambda row: True,
        "r0_value_recovery": lambda row: has_any(
            row.get("repair_buckets", []),
            {"arg_value_exactness", "live_slot_values"},
        ),
        "r0_function_selection": lambda row: (
            row.get("primary_failure_type") == "wrong_function"
            or has_any(row.get("repair_buckets", []), {"function_name_disambiguation"})
        ),
        "r0_schema_completion": lambda row: (
            row.get("primary_failure_type")
            in {"missing_arg", "extra_arg", "arg_key_mismatch"}
            or has_any(row.get("repair_buckets", []), {"schema_completion"})
        ),
        "r0_sql_domain_control": lambda row: (
            row.get("category") == "sql"
            or has_any(row.get("repair_buckets", []), {"sql_schema_discipline"})
        ),
        "r0_misc_failure": lambda row: has_any(row.get("repair_buckets", []), {"misc_failure"}),
        "r0_normalization_pilots": lambda row: has_any(
            row.get("repair_buckets", []),
            {
                "time_normalization",
                "unit_default_normalization",
                "formula_normalization",
            },
        ),
        "r0_time_normalization": lambda row: has_any(
            row.get("repair_buckets", []), {"time_normalization"}
        ),
        "r0_unit_default_normalization": lambda row: has_any(
            row.get("repair_buckets", []), {"unit_default_normalization"}
        ),
        "r0_formula_normalization": lambda row: has_any(
            row.get("repair_buckets", []), {"formula_normalization"}
        ),
    }
    for category in sorted(categories):
        safe = category.replace("-", "_").replace("/", "_")
        filters[f"r0_category_control_{safe}"] = (
            lambda row, _category=category: row.get("category") == _category
        )
    return filters


def sorted_pairs_for_ids(catalog: dict[str, dict[str, Any]], eval_ids: set[str]) -> list[dict[str, Any]]:
    return [catalog_to_pair(catalog[eval_id]) for eval_id in sorted(eval_ids) if eval_id in catalog]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--failure-matrix", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke-limit", type=int, default=8)
    parser.add_argument("--max-per-run", type=int)
    args = parser.parse_args()

    catalog_rows = read_jsonl(args.catalog)
    failure_rows = read_jsonl(args.failure_matrix)
    catalog = {row["eval_id"]: row for row in catalog_rows}
    decision_ids = {
        row["eval_id"]
        for row in catalog_rows
        if row.get("split_role") in DECISION_SPLITS
    }
    heldout_ids = {
        row["eval_id"]
        for row in catalog_rows
        if row.get("split_role") == "heldout"
    }
    decision_failures = [
        row for row in failure_rows if row.get("eval_id") in decision_ids
    ]
    categories = {row.get("category") for row in catalog_rows if row.get("category")}
    filters = build_run_filters(categories)

    pairs_dir = args.out_dir / "pairs"
    write_jsonl(
        pairs_dir / "all_catalog.jsonl",
        sorted_pairs_for_ids(catalog, set(catalog)),
    )
    write_jsonl(
        pairs_dir / "decision_eligible_all.jsonl",
        sorted_pairs_for_ids(catalog, decision_ids),
    )
    write_jsonl(
        pairs_dir / "heldout_audit_only.jsonl",
        sorted_pairs_for_ids(catalog, heldout_ids),
    )

    specs = []
    ids_by_run: dict[str, set[str]] = defaultdict(set)
    for run_id, predicate in filters.items():
        for row in decision_failures:
            if predicate(row):
                ids_by_run[run_id].add(row["eval_id"])

    for run_id in sorted(ids_by_run):
        eval_ids = ids_by_run[run_id]
        rows = sorted_pairs_for_ids(catalog, eval_ids)
        if args.max_per_run is not None:
            rows = rows[: args.max_per_run]
        pair_path = pairs_dir / f"{run_id}.jsonl"
        smoke_path = pairs_dir / f"{run_id}.smoke.jsonl"
        write_jsonl(pair_path, rows)
        write_jsonl(smoke_path, rows[: args.smoke_limit])
        specs.append(
            {
                "run_id": run_id,
                "pair_path": str(pair_path),
                "smoke_pair_path": str(smoke_path),
                "rows": len(rows),
                "unique_eval_ids": len({row["id"] for row in rows}),
                "split_boundary": "train+calibration+validation only; heldout excluded",
            }
        )

    summary = {
        "catalog_rows": len(catalog_rows),
        "failure_observations": len(failure_rows),
        "decision_eval_ids": len(decision_ids),
        "heldout_eval_ids": len(heldout_ids),
        "smoke_limit": args.smoke_limit,
        "max_per_run": args.max_per_run,
        "runs": specs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "attribution_slice_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
