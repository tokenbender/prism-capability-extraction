#!/usr/bin/env python3
"""Build the issue #8 BFCL failure-conditioned decomposition corpus.

The script reads published #4/#5/#6 failure-bucket artifacts from the
TokenBender/circuit-discovery Hugging Face dataset, creates a deterministic
row-level failure matrix, assigns leakage-safe split roles by stable BFCL eval
ID, and writes a bucket report plus launch recommendations for the first
failure-conditioned attribution runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


MATRIX_VERSION = "issue8_failure_conditioned_decomposition_v1"
DEFAULT_REPO_ID = "TokenBender/circuit-discovery"
DEFAULT_OUT_DIR = Path("results/bfcl/issue8_failure_conditioned_decomposition")
TOTAL_MLP_CHANNELS = 36 * 12288
SPLIT_SEED = "issue8_failure_conditioned_decomposition_v1"
DECISION_SPLITS = {"train", "calibration", "validation"}

ISSUE_ROOTS = {
    4: "bfcl/issue4_edge_collimation_v1",
    5: "bfcl/issue5_nearmiss_loop_v1",
    6: "bfcl/issue6_tree_search_v1",
}

PRIMARY_FAILURE_TYPES = [
    "wrong_arg_value",
    "wrong_function",
    "missing_arg",
    "arg_key_mismatch",
    "multi_call_or_extra_call",
    "extra_arg",
    "no_parse_or_no_call",
]

REPAIR_BUCKETS = [
    "arg_value_exactness",
    "live_slot_values",
    "function_name_disambiguation",
    "schema_completion",
    "sql_schema_discipline",
    "unit_default_normalization",
    "misc_failure",
    "time_normalization",
    "formula_normalization",
]

BRANCH_PROFILES = {
    "b001": "conservative_nearmiss",
    "b002": "bucket_balanced",
    "b003": "teacher_ranked",
    "b004": "schema_stratified",
    "b005": "compression_biased",
    "b006": "hardcase_replay",
    "b007": "epsilon_repair",
    "b008": "pareto_trim",
    "b009": "compression_biased",
    "b010": "teacher_ranked",
    "b011": "bucket_balanced",
    "b012": "schema_stratified",
    "b013": "conservative_nearmiss",
    "b014": "hardcase_replay",
    "b015": "epsilon_repair",
    "b016": "pareto_trim",
    "b017": "compression_biased",
    "b018": "teacher_ranked",
    "b019": "bucket_balanced",
    "b020": "schema_stratified",
}

BRANCH_PARENTS = {
    "b001": "issue6_r0",
    "b002": "issue6_r0",
    "b003": "issue6_r0",
    "b004": "issue6_r0",
    "b005": "issue6_r0",
    "b006": "issue6_r0",
    "b007": "b005",
    "b008": "b005",
    "b009": "b005",
    "b010": "b005",
    "b011": "b005",
    "b012": "b005",
    "b013": "b005",
    "b014": "b005",
    "b015": "b007",
    "b016": "b007",
    "b017": "b007",
    "b018": "b007",
    "b019": "b007",
    "b020": "b007",
}

SELECTED_NEAR_FRONTIER_UNITS = {
    "issue4_full",
    "issue5_r3",
    "issue6_b007",
    "issue6_b014",
    "issue6_b016",
}

FRONTIER_MISS_POINTS = {
    ("issue5_r3", 160000),
    ("issue5_r3", 200000),
    ("issue5_r3", 240000),
    ("issue6_b014", 120000),
    ("issue6_b014", 140000),
    ("issue6_b014", 160000),
    ("issue6_b007", 160000),
    ("issue6_b007", 180000),
    ("issue6_b007", 200000),
    ("issue6_b007", 220000),
    ("issue6_b007", 240000),
    ("issue6_b016", 140000),
    ("issue6_b016", 160000),
    ("issue6_b016", 180000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--split-seed", default=SPLIT_SEED)
    return parser.parse_args()


def hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text().strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def request_json(url: str, *, headers: dict[str, str], timeout: float, retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - exercised in live runs
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to fetch JSON after {retries} attempts: {url}") from last_error


def request_text(url: str, *, headers: dict[str, str], timeout: float, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:  # pragma: no cover - exercised in live runs
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to fetch text after {retries} attempts: {url}") from last_error


def parse_next_link(link_header: str) -> str | None:
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    return match.group(1) if match else None


def list_hf_tree(
    *,
    repo_id: str,
    revision: str,
    root: str,
    headers: dict[str, str],
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{root}"
    params = "?recursive=true&expand=false&limit=1000"
    out: list[dict[str, Any]] = []
    next_url: str | None = url + params
    while next_url:
        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(retries):
            try:
                response = requests.get(next_url, headers=headers, timeout=timeout)
                response.raise_for_status()
                break
            except Exception as exc:  # pragma: no cover - exercised in live runs
                last_error = exc
                time.sleep(min(2**attempt, 10))
        if response is None:
            raise RuntimeError(f"failed to list tree after {retries} attempts: {next_url}") from last_error
        out.extend(response.json())
        next_url = parse_next_link(response.headers.get("Link", ""))
    return out


def raw_url(repo_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_role(eval_id: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{eval_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % 10000
    if bucket < 6000:
        return "train"
    if bucket < 7500:
        return "calibration"
    if bucket < 9000:
        return "validation"
    return "heldout"


def call_name(call: Any) -> str | None:
    return call.get("name") if isinstance(call, dict) else None


def call_args(call: Any) -> dict[str, Any]:
    if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
        return call["arguments"]
    return {}


def union_keys(calls: list[Any]) -> list[str]:
    keys: set[str] = set()
    for call in calls:
        keys.update(call_args(call))
    return sorted(keys)


def parse_source_path(path: str) -> dict[str, Any]:
    k_match = re.search(r"failure_buckets_k(\d+)/all_failures\.jsonl$", path)
    if not k_match:
        raise ValueError(f"could not parse k budget from {path}")
    k_budget = int(k_match.group(1))
    issue_match = re.search(r"bfcl/issue(\d+)_", path)
    if not issue_match:
        raise ValueError(f"could not parse issue number from {path}")
    issue = int(issue_match.group(1))

    round_id: str | None = None
    branch_id: str | None = None
    branch_profile: str | None = None
    parent_unit: str | None = None
    unit_id = f"issue{issue}_unknown"

    if issue == 4:
        round_id = "full"
        unit_id = "issue4_full"
        branch_profile = "generic_edge_curriculum"
    elif issue == 5:
        match = re.search(r"/run/(r\d+)/failure_buckets_", path)
        if not match:
            raise ValueError(f"could not parse issue5 round from {path}")
        round_id = match.group(1)
        unit_id = f"issue5_{round_id}"
        branch_profile = "r0_reproduce_issue2" if round_id == "r0" else "targeted_nearmiss"
        parent_unit = "issue2" if round_id == "r0" else f"issue5_r{int(round_id[1:]) - 1}"
    elif issue == 6:
        if "/run/r0/" in path:
            round_id = "r0"
            unit_id = "issue6_r0"
            branch_profile = "imported_issue5_r0_root"
            parent_unit = "issue5_r0"
        else:
            match = re.search(r"/run/branches/(b\d+)/failure_buckets_", path)
            if not match:
                raise ValueError(f"could not parse issue6 branch from {path}")
            branch_id = match.group(1)
            round_id = "branch"
            unit_id = f"issue6_{branch_id}"
            branch_profile = BRANCH_PROFILES.get(branch_id, "unknown")
            parent_unit = BRANCH_PARENTS.get(branch_id)

    return {
        "source_issue": issue,
        "round_id": round_id,
        "branch_id": branch_id,
        "branch_profile": branch_profile,
        "parent_unit": parent_unit,
        "source_unit": unit_id,
        "k_budget": k_budget,
        "selected_mlp_channels": k_budget,
        "selected_mlp_percent": round(100.0 * k_budget / TOTAL_MLP_CHANNELS, 4),
    }


def gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_record(
    *,
    row: dict[str, Any],
    source_path: str,
    source_meta: dict[str, Any],
    line_index: int,
    split: str,
) -> dict[str, Any]:
    prediction_calls = row.get("prediction_calls") or []
    reference_calls = row.get("reference_calls") or []
    detail = row.get("detail") or {}
    prompt = row.get("prompt") or ""
    tools = row.get("tools") or []
    pred_count = len(prediction_calls)
    parse_status = "no_call" if pred_count == 0 else "single_call" if pred_count == 1 else "multi_call"
    pred_name = call_name(prediction_calls[0]) if prediction_calls else None
    ref_names = sorted({name for name in (call_name(call) for call in reference_calls) if name})
    eval_id = str(row["id"])
    observation_id = stable_hash([source_path, line_index, eval_id, row.get("failure_type")])

    pred_keys = detail.get("pred_keys")
    ref_keys = detail.get("ref_keys")
    if pred_keys is None:
        pred_keys = union_keys(prediction_calls)
    if ref_keys is None:
        ref_keys = union_keys(reference_calls)

    record = {
        "matrix_version": MATRIX_VERSION,
        "observation_id": observation_id,
        "eval_id": eval_id,
        "split_role": split,
        "category": row.get("category", "unknown"),
        "primary_failure_type": row.get("failure_type", "unknown"),
        "repair_buckets": sorted(row.get("repair_buckets") or ["misc_failure"]),
        "prediction_parse_status": parse_status,
        "prediction_call_count": pred_count,
        "reference_call_count": len(reference_calls),
        "predicted_function_name": pred_name,
        "reference_function_names": ref_names,
        "predicted_argument_keys": sorted(pred_keys),
        "reference_argument_keys": sorted(ref_keys),
        "missing_argument_keys": sorted(detail.get("missing_keys") or []),
        "extra_argument_keys": sorted(detail.get("extra_keys") or []),
        "wrong_value_keys": sorted(detail.get("wrong_value_keys") or []),
        "prediction_calls_hash": stable_hash(prediction_calls),
        "reference_calls_hash": stable_hash(reference_calls),
        "predicted_arguments_hash": stable_hash([call_args(call) for call in prediction_calls]),
        "reference_arguments_hash": stable_hash([call_args(call) for call in reference_calls]),
        "prediction_text_hash": text_hash(row.get("prediction_text") or ""),
        "prompt_hash": text_hash(prompt),
        "tools_hash": stable_hash(tools),
        "source_artifact_path": source_path,
        "source_line_index": line_index,
        "is_selected_near_frontier_subset": source_meta["source_unit"] in SELECTED_NEAR_FRONTIER_UNITS,
        **source_meta,
    }
    return record


def make_catalog_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    prompt = row.get("prompt") or ""
    tools = row.get("tools") or []
    reference_calls = row.get("reference_calls") or []
    return {
        "matrix_version": MATRIX_VERSION,
        "eval_id": str(row["id"]),
        "split_role": split,
        "category": row.get("category", "unknown"),
        "prompt": prompt,
        "prompt_hash": text_hash(prompt),
        "tools": tools,
        "tools_hash": stable_hash(tools),
        "reference_calls": reference_calls,
        "reference_calls_hash": stable_hash(reference_calls),
        "do_not_train_if_split_role_heldout": split == "heldout",
    }


def add_count(counter: Counter[str], key: Any, inc: int = 1) -> None:
    counter[str(key)] += inc


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def observe_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    failure_type_obs: Counter[str] = Counter()
    repair_obs: Counter[str] = Counter()
    category_obs: Counter[str] = Counter()
    issue_obs: Counter[str] = Counter()
    split_obs: Counter[str] = Counter()
    failure_type_ids: defaultdict[str, set[str]] = defaultdict(set)
    repair_ids: defaultdict[str, set[str]] = defaultdict(set)
    category_ids: defaultdict[str, set[str]] = defaultdict(set)
    issue_ids: defaultdict[str, set[str]] = defaultdict(set)
    split_ids: defaultdict[str, set[str]] = defaultdict(set)

    for record in records:
        eval_id = record["eval_id"]
        ft = record["primary_failure_type"]
        add_count(failure_type_obs, ft)
        failure_type_ids[ft].add(eval_id)
        add_count(category_obs, record["category"])
        category_ids[record["category"]].add(eval_id)
        add_count(issue_obs, f"issue{record['source_issue']}")
        issue_ids[f"issue{record['source_issue']}"].add(eval_id)
        add_count(split_obs, record["split_role"])
        split_ids[record["split_role"]].add(eval_id)
        for bucket in record["repair_buckets"]:
            add_count(repair_obs, bucket)
            repair_ids[bucket].add(eval_id)

    return {
        "failure_type_observations": sorted_counter(failure_type_obs),
        "failure_type_unique_eval_ids": dict(sorted((k, len(v)) for k, v in failure_type_ids.items())),
        "repair_bucket_observations": sorted_counter(repair_obs),
        "repair_bucket_unique_eval_ids": dict(sorted((k, len(v)) for k, v in repair_ids.items())),
        "category_observations": sorted_counter(category_obs),
        "category_unique_eval_ids": dict(sorted((k, len(v)) for k, v in category_ids.items())),
        "issue_observations": sorted_counter(issue_obs),
        "issue_unique_eval_ids": dict(sorted((k, len(v)) for k, v in issue_ids.items())),
        "split_observations": sorted_counter(split_obs),
        "split_unique_eval_ids": dict(sorted((k, len(v)) for k, v in split_ids.items())),
    }


def cross_counts(records: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    unique_ids: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        left_value = str(record[left])
        right_value = str(record[right])
        key = (left_value, right_value)
        counts[key] += 1
        unique_ids[key].add(record["eval_id"])
    rows = []
    for (left_value, right_value), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append({
            left: left_value,
            right: right_value,
            "failure_observations": count,
            "unique_eval_ids": len(unique_ids[(left_value, right_value)]),
        })
    return rows


def bucket_overlap(records: list[dict[str, Any]], buckets: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids_by_bucket: dict[str, set[str]] = {bucket: set() for bucket in buckets}
    for record in records:
        for bucket in record["repair_buckets"]:
            ids_by_bucket.setdefault(bucket, set()).add(record["eval_id"])

    matrix_rows = []
    pair_rows = []
    active = [bucket for bucket in buckets if ids_by_bucket.get(bucket)]
    for left in active:
        row: dict[str, Any] = {"bucket": left}
        for right in active:
            a = ids_by_bucket[left]
            b = ids_by_bucket[right]
            union = a | b
            value = len(a & b) / len(union) if union else 0.0
            row[right] = round(value, 4)
            if left < right:
                pair_rows.append({
                    "bucket_a": left,
                    "bucket_b": right,
                    "intersection_unique_eval_ids": len(a & b),
                    "union_unique_eval_ids": len(union),
                    "jaccard": round(value, 4),
                })
        matrix_rows.append(row)
    pair_rows.sort(key=lambda item: (-item["jaccard"], -item["intersection_unique_eval_ids"], item["bucket_a"], item["bucket_b"]))
    return matrix_rows, pair_rows


def build_persistence_tags(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "ks": set(),
        "units": set(),
        "branches": set(),
        "branches_high_k": set(),
        "failure_types": set(),
        "repair_buckets": set(),
        "near_frontier_failure_types": set(),
        "near_frontier_repair_buckets": set(),
        "issue5_r0_ks": set(),
        "issue5_r3_ks": set(),
        "issue6_r0_ks": set(),
        "issue6_branch_overlap_ks": set(),
    })
    for record in records:
        eval_id = record["eval_id"]
        item = stats[eval_id]
        item["ks"].add(record["k_budget"])
        item["units"].add(record["source_unit"])
        if record.get("branch_id"):
            item["branches"].add(record["branch_id"])
            if record["k_budget"] >= 160000:
                item["branches_high_k"].add(record["branch_id"])
        item["failure_types"].add(record["primary_failure_type"])
        item["repair_buckets"].update(record["repair_buckets"])
        if record["source_unit"] in SELECTED_NEAR_FRONTIER_UNITS:
            item["near_frontier_failure_types"].add(record["primary_failure_type"])
            item["near_frontier_repair_buckets"].update(record["repair_buckets"])
        if record["source_unit"] == "issue5_r0":
            item["issue5_r0_ks"].add(record["k_budget"])
        if record["source_unit"] == "issue5_r3":
            item["issue5_r3_ks"].add(record["k_budget"])
        if record["source_unit"] == "issue6_r0":
            item["issue6_r0_ks"].add(record["k_budget"])
        if record["source_issue"] == 6 and record.get("branch_id") and record["k_budget"] in {80000, 120000, 160000, 200000, 240000}:
            item["issue6_branch_overlap_ks"].add(record["k_budget"])

    tags_by_id: dict[str, set[str]] = {}
    for eval_id, item in stats.items():
        tags: set[str] = set()
        if len(item["branches_high_k"]) >= 5:
            tags.add("persistent_across_branches")
        if len(item["ks"]) >= 5 or (min(item["ks"]) <= 80000 and max(item["ks"]) >= 180000):
            tags.add("persistent_across_k")
        if item["ks"] and max(item["ks"]) <= 120000:
            tags.add("low_k_only")
        if item["issue5_r0_ks"] and item["issue5_r0_ks"] - item["issue5_r3_ks"]:
            tags.add("fixed_by_r3_or_later")
        if item["issue6_branch_overlap_ks"] - item["issue6_r0_ks"]:
            tags.add("regressed_in_tree_branch")
        if len(item["near_frontier_failure_types"]) > 1 or len(item["near_frontier_repair_buckets"]) > 2:
            tags.add("bucket_conflict_or_ambiguous")
        tags_by_id[eval_id] = tags
    return tags_by_id


def add_row_level_tags(records: list[dict[str, Any]], tags_by_id: dict[str, set[str]]) -> None:
    for record in records:
        tags = set(tags_by_id.get(record["eval_id"], set()))
        if (record["source_unit"], record["k_budget"]) in FRONTIER_MISS_POINTS:
            tags.add("near_frontier_miss")
        record["persistence_tags"] = sorted(tags)


def persistence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    obs_counts: Counter[str] = Counter()
    id_sets: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        for tag in record.get("persistence_tags", []):
            obs_counts[tag] += 1
            id_sets[tag].add(record["eval_id"])
    return {
        "tag_observations": sorted_counter(obs_counts),
        "tag_unique_eval_ids": dict(sorted((tag, len(ids)) for tag, ids in id_sets.items())),
    }


def selected_subset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [record for record in records if record["is_selected_near_frontier_subset"]]
    by_unit: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in subset:
        by_unit[record["source_unit"]].append(record)
    return {
        "description": "issue4 all k + issue5 r3 all k + issue6 b014/b007/b016 all k",
        "failure_observations": len(subset),
        "unique_eval_ids": len({record["eval_id"] for record in subset}),
        "by_unit": {
            unit: {
                "failure_observations": len(rows),
                "unique_eval_ids": len({record["eval_id"] for record in rows}),
                "counts": observe_counts(rows),
            }
            for unit, rows in sorted(by_unit.items())
        },
        "counts": observe_counts(subset),
    }


def enough_bucket_label(unique_ids: int, observations: int) -> str:
    if unique_ids >= 250 and observations >= 1000:
        return "primary_launch"
    if unique_ids >= 100 and observations >= 500:
        return "pilot_or_nested"
    return "group_or_defer"


def make_recommendations(decision_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = observe_counts(decision_records)
    repair_obs = counts["repair_bucket_observations"]
    repair_ids = counts["repair_bucket_unique_eval_ids"]
    failure_obs = counts["failure_type_observations"]
    failure_ids = counts["failure_type_unique_eval_ids"]

    recommendations = [
        {
            "run_id": "r0_global_decision_eligible",
            "mode": "global attribution",
            "selection": "all train+calibration+validation failure observations; heldout excluded from selection",
            "purpose": "matched baseline for bucket-conditioned attribution",
            "priority": 0,
        },
        {
            "run_id": "r0_value_recovery",
            "mode": "repair-bucket attribution",
            "selection": "arg_value_exactness plus live_slot_values, prioritize persistent and near-frontier misses",
            "decision_observations": int(repair_obs.get("arg_value_exactness", 0) + repair_obs.get("live_slot_values", 0)),
            "decision_unique_eval_ids": len(set()),
            "purpose": "largest recoverable mass: exact values and dynamic slot extraction",
            "priority": 1,
        },
        {
            "run_id": "r0_function_selection",
            "mode": "primary-type / repair-bucket attribution",
            "selection": "wrong_function and function_name_disambiguation",
            "decision_observations": int(max(failure_obs.get("wrong_function", 0), repair_obs.get("function_name_disambiguation", 0))),
            "decision_unique_eval_ids": int(max(failure_ids.get("wrong_function", 0), repair_ids.get("function_name_disambiguation", 0))),
            "purpose": "test whether tool-name choice has a separable substrate",
            "priority": 2,
        },
        {
            "run_id": "r0_schema_completion",
            "mode": "primary-type / repair-bucket attribution",
            "selection": "missing_arg + extra_arg + arg_key_mismatch + schema_completion",
            "decision_observations": int(repair_obs.get("schema_completion", 0)),
            "decision_unique_eval_ids": int(repair_ids.get("schema_completion", 0)),
            "purpose": "argument-key and required-schema construction",
            "priority": 3,
        },
        {
            "run_id": "r0_sql_domain_control",
            "mode": "repair-bucket and category-conditioned attribution",
            "selection": "sql_schema_discipline plus category=sql",
            "decision_observations": int(repair_obs.get("sql_schema_discipline", 0)),
            "decision_unique_eval_ids": int(repair_ids.get("sql_schema_discipline", 0)),
            "purpose": "domain-specific control to separate error buckets from ordinary category slicing",
            "priority": 4,
        },
        {
            "run_id": "r0_category_controls",
            "mode": "category-conditioned attribution",
            "selection": "simple, live_simple, sql, java, javascript, exec_simple decision-eligible rows",
            "purpose": "prove failure-conditioned slices add signal beyond category-conditioned slices",
            "priority": 5,
        },
        {
            "run_id": "r0_small_normalization_pilots",
            "mode": "nested repair-bucket pilots",
            "selection": "time_normalization, unit_default_normalization, formula_normalization",
            "purpose": "run only as pilots or nested under value/domain branches unless compute is abundant",
            "priority": 6,
        },
    ]

    value_ids: set[str] = set()
    for record in decision_records:
        if {"arg_value_exactness", "live_slot_values"} & set(record["repair_buckets"]):
            value_ids.add(record["eval_id"])
    recommendations[1]["decision_unique_eval_ids"] = len(value_ids)

    bucket_readiness = []
    for bucket in REPAIR_BUCKETS:
        bucket_readiness.append({
            "bucket": bucket,
            "decision_failure_observations": int(repair_obs.get(bucket, 0)),
            "decision_unique_eval_ids": int(repair_ids.get(bucket, 0)),
            "recommendation": enough_bucket_label(int(repair_ids.get(bucket, 0)), int(repair_obs.get(bucket, 0))),
        })
    recommendations.append({
        "run_id": "bucket_readiness_table",
        "mode": "planning metadata",
        "bucket_readiness": bucket_readiness,
        "priority": 99,
    })
    return recommendations


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}%" if denominator else "n/a"


def write_markdown_report(
    *,
    path: Path,
    summary: dict[str, Any],
    selected_subset: dict[str, Any],
    recommendations: list[dict[str, Any]],
    overlap_pairs: list[dict[str, Any]],
) -> None:
    total = summary["total_failure_observations"]
    counts = summary["all_counts"]
    decision_counts = summary["decision_eligible_counts"]
    lines: list[str] = []
    lines.append("# BFCL Issue #8 Failure-Conditioned Decomposition Bucket Report")
    lines.append("")
    lines.append(f"Generated: `{summary['generated_at']}`")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append(markdown_table(
        ["Metric", "Value"],
        [
            ["source issues", "#4, #5, #6"],
            ["source all_failures files", summary["source_file_count"]],
            ["failure observations", total],
            ["unique eval IDs failed at least once", summary["unique_eval_ids_failed_at_least_once"]],
            ["deterministic split seed", summary["split_policy"]["seed"]],
            ["artifact prefix", summary["artifact_prefix"]],
        ],
    ))
    lines.append("")
    lines.append("Observation-level counts are repeated over issue, branch, round, and selected-channel budgets. Unique eval-ID counts collapse repeated observations of the same BFCL row.")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    split_rows = []
    for role in ["train", "calibration", "validation", "heldout"]:
        split_rows.append([
            role,
            counts["split_unique_eval_ids"].get(role, 0),
            counts["split_observations"].get(role, 0),
        ])
    lines.append(markdown_table(["Split", "Unique eval IDs", "Failure observations"], split_rows))
    lines.append("")
    lines.append("Launch decisions below use train + calibration + validation summaries. Heldout rows are retained in the artifact for final auditing but excluded from method choice.")
    lines.append("")
    lines.append("## By Issue")
    lines.append("")
    issue_rows = []
    for issue, obs in counts["issue_observations"].items():
        issue_rows.append([issue, obs, counts["issue_unique_eval_ids"].get(issue, 0)])
    lines.append(markdown_table(["Issue", "Failure observations", "Unique eval IDs"], issue_rows))
    lines.append("")
    lines.append("## Primary Failure Types")
    lines.append("")
    failure_rows = []
    for name, obs in counts["failure_type_observations"].items():
        failure_rows.append([
            name,
            obs,
            pct(obs, total),
            counts["failure_type_unique_eval_ids"].get(name, 0),
            decision_counts["failure_type_observations"].get(name, 0),
            decision_counts["failure_type_unique_eval_ids"].get(name, 0),
        ])
    lines.append(markdown_table(
        ["Failure type", "All obs", "All share", "All unique IDs", "Decision obs", "Decision unique IDs"],
        failure_rows,
    ))
    lines.append("")
    lines.append("## Repair Buckets")
    lines.append("")
    repair_rows = []
    for name, obs in counts["repair_bucket_observations"].items():
        repair_rows.append([
            name,
            obs,
            pct(obs, total),
            counts["repair_bucket_unique_eval_ids"].get(name, 0),
            decision_counts["repair_bucket_observations"].get(name, 0),
            decision_counts["repair_bucket_unique_eval_ids"].get(name, 0),
        ])
    lines.append(markdown_table(
        ["Repair bucket", "All obs", "All share", "All unique IDs", "Decision obs", "Decision unique IDs"],
        repair_rows,
    ))
    lines.append("")
    lines.append("## Selected Near-Frontier Subset")
    lines.append("")
    unit_rows = []
    for unit, data in selected_subset["by_unit"].items():
        unit_rows.append([unit, data["failure_observations"], data["unique_eval_ids"]])
    lines.append(markdown_table(["Unit", "Failure observations", "Unique eval IDs"], unit_rows))
    lines.append("")
    lines.append(f"Selected subset total: `{selected_subset['failure_observations']}` observations over `{selected_subset['unique_eval_ids']}` unique eval IDs.")
    lines.append("")
    lines.append("## Persistence Tags")
    lines.append("")
    tag_rows = []
    tag_obs = summary["persistence"]["tag_observations"]
    tag_ids = summary["persistence"]["tag_unique_eval_ids"]
    for tag, obs in tag_obs.items():
        tag_rows.append([tag, obs, tag_ids.get(tag, 0)])
    lines.append(markdown_table(["Tag", "Failure observations", "Unique eval IDs"], tag_rows))
    lines.append("")
    lines.append("## Decomposition Map")
    lines.append("")
    lines.append("```text")
    lines.append("BFCL failures")
    lines.append("  |")
    lines.append("  +-- tool selection")
    lines.append("  |     +-- wrong_function")
    lines.append("  |")
    lines.append("  +-- schema construction")
    lines.append("  |     +-- missing_arg")
    lines.append("  |     +-- extra_arg")
    lines.append("  |     +-- arg_key_mismatch")
    lines.append("  |")
    lines.append("  +-- value recovery")
    lines.append("  |     +-- wrong_arg_value")
    lines.append("  |     +-- live_slot_values")
    lines.append("  |     +-- time/unit/formula normalization")
    lines.append("  |")
    lines.append("  +-- domain discipline")
    lines.append("  |     +-- sql_schema_discipline")
    lines.append("  |     +-- code/category-specific failures")
    lines.append("  |")
    lines.append("  +-- output protocol")
    lines.append("        +-- multi_call_or_extra_call")
    lines.append("        +-- no_parse_or_no_call")
    lines.append("```")
    lines.append("")
    lines.append("## Highest Bucket Overlaps")
    lines.append("")
    overlap_rows = []
    for row in overlap_pairs[:12]:
        overlap_rows.append([
            row["bucket_a"],
            row["bucket_b"],
            row["intersection_unique_eval_ids"],
            row["union_unique_eval_ids"],
            row["jaccard"],
        ])
    lines.append(markdown_table(["Bucket A", "Bucket B", "Intersection IDs", "Union IDs", "Jaccard"], overlap_rows))
    lines.append("")
    lines.append("## Attribution Launch Recommendation")
    lines.append("")
    rec_rows = []
    for rec in recommendations:
        if "bucket_readiness" in rec:
            continue
        rec_rows.append([
            rec["priority"],
            rec["run_id"],
            rec["mode"],
            rec.get("decision_observations", ""),
            rec.get("decision_unique_eval_ids", ""),
            rec["purpose"],
        ])
    lines.append(markdown_table(["Priority", "Run", "Mode", "Decision obs", "Decision unique IDs", "Purpose"], rec_rows))
    lines.append("")
    lines.append("## Artifact Files")
    lines.append("")
    for file_info in summary["files"]:
        lines.append(f"- `{file_info['path']}` sha256 `{file_info['sha256']}` bytes `{file_info['bytes']}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    headers = hf_headers()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory: dict[str, Any] = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "roots": ISSUE_ROOTS,
        "root_file_counts": {},
        "all_failure_paths": [],
    }

    all_failure_paths: list[str] = []
    for issue, root in ISSUE_ROOTS.items():
        items = list_hf_tree(
            repo_id=args.repo_id,
            revision=args.revision,
            root=root,
            headers=headers,
            timeout=args.timeout,
            retries=args.retries,
        )
        files = [item["path"] for item in items if item.get("type") == "file"]
        failure_paths = sorted(path for path in files if path.endswith("all_failures.jsonl"))
        manifest_paths = sorted(path for path in files if path.endswith("manifest.json"))
        inventory["root_file_counts"][f"issue{issue}"] = {
            "total_items": len(items),
            "files": len(files),
            "all_failures_jsonl": len(failure_paths),
            "manifest_json": len(manifest_paths),
        }
        all_failure_paths.extend(failure_paths)
    all_failure_paths = sorted(all_failure_paths)
    inventory["all_failure_paths"] = all_failure_paths

    records: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}

    for source_path in all_failure_paths:
        source_meta = parse_source_path(source_path)
        text = request_text(
            raw_url(args.repo_id, args.revision, source_path),
            headers=headers,
            timeout=args.timeout,
            retries=args.retries,
        )
        for line_index, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            eval_id = str(row["id"])
            role = split_role(eval_id, args.split_seed)
            if eval_id not in catalog:
                catalog[eval_id] = make_catalog_row(row, role)
            else:
                catalog[eval_id].setdefault("categories_seen", set())
                if isinstance(catalog[eval_id]["categories_seen"], set):
                    catalog[eval_id]["categories_seen"].add(row.get("category", "unknown"))
            records.append(
                make_record(
                    row=row,
                    source_path=source_path,
                    source_meta=source_meta,
                    line_index=line_index,
                    split=role,
                )
            )

    for item in catalog.values():
        if isinstance(item.get("categories_seen"), set):
            item["categories_seen"] = sorted(item["categories_seen"])

    tags_by_id = build_persistence_tags(records)
    add_row_level_tags(records, tags_by_id)
    records.sort(key=lambda row: (row["source_issue"], row["source_unit"], row["k_budget"], row["eval_id"], row["observation_id"]))
    catalog_rows = sorted(catalog.values(), key=lambda row: row["eval_id"])

    decision_records = [record for record in records if record["split_role"] in DECISION_SPLITS]
    heldout_records = [record for record in records if record["split_role"] == "heldout"]
    selected_subset = selected_subset_summary(records)
    repair_overlap_matrix, repair_overlap_pairs = bucket_overlap(decision_records, REPAIR_BUCKETS)
    recommendations = make_recommendations(decision_records)

    category_failure = cross_counts(records, "category", "primary_failure_type")
    k_failure = cross_counts(records, "k_budget", "primary_failure_type")
    unit_failure = cross_counts(records, "source_unit", "primary_failure_type")
    split_failure = cross_counts(records, "split_role", "primary_failure_type")

    artifact_prefix = "bfcl/issue8_failure_conditioned_decomposition_v1"
    summary: dict[str, Any] = {
        "matrix_version": MATRIX_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "artifact_prefix": artifact_prefix,
        "source_file_count": len(all_failure_paths),
        "source_inventory": inventory,
        "total_failure_observations": len(records),
        "unique_eval_ids_failed_at_least_once": len(catalog_rows),
        "decision_eligible_failure_observations": len(decision_records),
        "heldout_failure_observations": len(heldout_records),
        "split_policy": {
            "seed": args.split_seed,
            "unit": "stable BFCL eval ID",
            "train": "hash bucket < 6000 / 10000",
            "calibration": "6000 <= hash bucket < 7500 / 10000",
            "validation": "7500 <= hash bucket < 9000 / 10000",
            "heldout": "9000 <= hash bucket < 10000 / 10000",
            "decision_eligible": sorted(DECISION_SPLITS),
        },
        "all_counts": observe_counts(records),
        "decision_eligible_counts": observe_counts(decision_records),
        "heldout_counts": observe_counts(heldout_records),
        "selected_near_frontier_subset": selected_subset,
        "persistence": persistence_summary(records),
        "tables": {
            "category_x_failure_type": category_failure,
            "k_budget_x_failure_type": k_failure,
            "source_unit_x_failure_type": unit_failure,
            "split_x_failure_type": split_failure,
        },
        "repair_bucket_overlap_jaccard_decision_eligible": {
            "matrix": repair_overlap_matrix,
            "pairs_sorted": repair_overlap_pairs,
        },
        "attribution_launch_recommendations": recommendations,
        "files": [],
    }

    gzip_jsonl(args.out_dir / "failure_matrix.jsonl.gz", records)
    gzip_jsonl(args.out_dir / "eval_id_catalog.jsonl.gz", catalog_rows)
    write_json(args.out_dir / "artifact_inventory.json", inventory)
    write_json(args.out_dir / "bucket_report.json", summary)
    write_json(args.out_dir / "launch_recommendations.json", recommendations)
    write_json(args.out_dir / "split_manifest.json", {
        "matrix_version": MATRIX_VERSION,
        "split_policy": summary["split_policy"],
        "splits": {
            role: sorted(row["eval_id"] for row in catalog_rows if row["split_role"] == role)
            for role in ["train", "calibration", "validation", "heldout"]
        },
    })
    write_csv(
        args.out_dir / "repair_bucket_overlap_jaccard.csv",
        repair_overlap_matrix,
        ["bucket"] + [row["bucket"] for row in repair_overlap_matrix],
    )
    write_csv(
        args.out_dir / "category_x_failure_type.csv",
        category_failure,
        ["category", "primary_failure_type", "failure_observations", "unique_eval_ids"],
    )
    write_csv(
        args.out_dir / "k_budget_x_failure_type.csv",
        k_failure,
        ["k_budget", "primary_failure_type", "failure_observations", "unique_eval_ids"],
    )
    write_csv(
        args.out_dir / "source_unit_x_failure_type.csv",
        unit_failure,
        ["source_unit", "primary_failure_type", "failure_observations", "unique_eval_ids"],
    )

    files = []
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file() and path.name not in {"bucket_report.json", "bucket_report.md", "checksums.sha256"}:
            files.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            })
    summary["files"] = files
    write_json(args.out_dir / "bucket_report.json", summary)
    write_markdown_report(
        path=args.out_dir / "bucket_report.md",
        summary=summary,
        selected_subset=selected_subset,
        recommendations=recommendations,
        overlap_pairs=repair_overlap_pairs,
    )
    files = []
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            files.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            })
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files)
    )

    print(json.dumps({
        "out_dir": str(args.out_dir),
        "source_file_count": len(all_failure_paths),
        "failure_observations": len(records),
        "unique_eval_ids": len(catalog_rows),
        "decision_eligible_failure_observations": len(decision_records),
        "heldout_failure_observations": len(heldout_records),
        "all_failure_type_counts": summary["all_counts"]["failure_type_observations"],
        "all_repair_bucket_counts": summary["all_counts"]["repair_bucket_observations"],
        "selected_subset_failure_observations": selected_subset["failure_observations"],
        "recommended_runs": [rec["run_id"] for rec in recommendations if rec.get("priority", 99) < 10],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
