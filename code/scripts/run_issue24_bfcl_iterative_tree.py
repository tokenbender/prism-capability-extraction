#!/usr/bin/env python3
"""Run Issue #24's twenty-branch zero-isolated BFCL repair-SFT tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "prism_bfcl_issue24_iterative_tree_v1"
CONFIG_SCHEMA_VERSION = "prism_bfcl_issue24_iterative_tree_config_v1"
CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = CODE_ROOT / "scripts"
BFCL = SCRIPTS / "bfcl_direct_qwen3.py"
TRAINER = SCRIPTS / "train_bfcl_zero_masked_lora.py"
CURRICULUM = SCRIPTS / "build_bfcl_tree_branch_curriculum.py"
LEAK_AUDIT = SCRIPTS / "audit_bfcl_train_eval_overlap.py"
FAILURE_BUCKETS = SCRIPTS / "build_bfcl_failure_buckets.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--base-train-jsonl", type=Path)
    parser.add_argument("--root-model", type=Path)
    parser.add_argument("--root-mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema: {config.get('schema_version')}")
    contract = config["contract"]
    if contract["trained_branch_attempts"] != 20:
        raise ValueError("Issue #24 requires exactly twenty trained branch attempts")
    if not contract["continue_after_exhausted_wave"]:
        raise ValueError("tree must continue after a wave has no accepted child")
    if not contract["terminal_scored_only_after_tree_frozen"]:
        raise ValueError("terminal holdout must remain sealed until tree freeze")
    if config["search"]["parallel_branches"] < 1 or config["search"]["beam_width"] < 1:
        raise ValueError("parallel_branches and beam_width must be positive")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(output_dir: Path, event: Mapping[str, Any]) -> None:
    path = output_dir / "events.jsonl"
    payload = {"time": time.time(), **dict(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_command(command: Sequence[str], *, output_dir: Path, env: Mapping[str, str] | None = None) -> None:
    append_event(output_dir, {"stage": "command", "command": list(command)})
    subprocess.run(list(command), check=True, env=dict(env) if env is not None else None)


def stable_key(seed: int, row_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{row_id}".encode()).hexdigest()


def stratified_split(rows: Sequence[Mapping[str, Any]], *, seed: int, terminal_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        by_category[str(row.get("category", "unknown"))].append(row)
    development: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for category, category_rows in sorted(by_category.items()):
        ordered = sorted(category_rows, key=lambda row: stable_key(seed, str(row["id"]), "terminal"))
        terminal_count = max(1, round(len(ordered) * terminal_fraction))
        terminal.extend(ordered[:terminal_count])
        development.extend(ordered[terminal_count:])
    development.sort(key=lambda row: str(row["id"]))
    terminal.sort(key=lambda row: str(row["id"]))
    return development, terminal


def stratified_subset(rows: Sequence[Mapping[str, Any]], *, count: int, seed: int, purpose: str) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: stable_key(seed, str(row["id"]), purpose))
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_category[str(row.get("category", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(selected) < min(count, len(ordered)):
        progressed = False
        for category in categories:
            if by_category[category] and len(selected) < count:
                selected.append(by_category[category].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def load_mask(path: Path, budget: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.load(path)["mlp_scores"].astype(np.float32, copy=False)
    if scores.ndim != 2:
        raise ValueError(f"expected 2D mlp_scores in {path}, got {scores.shape}")
    flat = scores.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:budget].astype(np.int32)
    if len(np.unique(order)) != budget:
        raise ValueError("mask selection contains duplicate indices")
    return scores, order


def selected_sha1(order: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(order, dtype=np.int32).tobytes()).hexdigest()


def selected_sha256(order: np.ndarray, total: int) -> str:
    bits = np.zeros(total, dtype=np.bool_)
    bits[order] = True
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()


def write_mask(path: Path, order: Sequence[int], *, shape: tuple[int, int]) -> dict[str, Any]:
    unique = list(dict.fromkeys(int(value) for value in order))
    scores = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for rank, index in enumerate(unique):
        scores[index] = float(len(unique) - rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mlp_scores=scores.reshape(shape))
    array = np.asarray(unique, dtype=np.int32)
    return {
        "path": str(path),
        "budget": len(unique),
        "selected_sha1": selected_sha1(array),
        "selected_sha256": selected_sha256(array, scores.size),
    }


def generate_candidates(parent_mask: Path, parent_budget: int, attribution: Path, branch_dir: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    parent_scores, parent_order = load_mask(parent_mask, parent_budget)
    attribution_scores = np.load(attribution)["mlp_scores"].astype(np.float32, copy=False)
    if attribution_scores.shape != parent_scores.shape:
        raise ValueError(f"attribution shape {attribution_scores.shape} != mask shape {parent_scores.shape}")
    flat_attr = attribution_scores.reshape(-1)
    parent_set = set(int(value) for value in parent_order)
    selected_ranked = sorted(parent_set, key=lambda index: (-float(flat_attr[index]), index))
    unselected_ranked = sorted((index for index in range(flat_attr.size) if index not in parent_set), key=lambda index: (-float(flat_attr[index]), index))
    minimum_budget = int(config["search"]["minimum_budget"])
    proposals: list[tuple[str, str, list[int], dict[str, Any]]] = [
        ("parent_control", "parent_control", list(parent_order), {"removed": 0, "added": 0})
    ]
    for delta in config["search"]["shrink_deltas"]:
        budget = max(parent_budget - int(delta), minimum_budget)
        if budget < parent_budget:
            proposals.append((f"shrink_{parent_budget - budget}", "parent_shrink", selected_ranked[:budget], {"removed": parent_budget - budget, "added": 0}))
    for count_raw in config["search"]["swap_counts"]:
        count = min(int(count_raw), parent_budget, len(unselected_ranked))
        kept = selected_ranked[: parent_budget - count]
        proposals.append((f"swap_{count}", "bounded_swap", kept + unselected_ranked[:count], {"removed": count, "added": count}))
    swap_shrink = config["search"]["swap_shrink"]
    add_count = min(int(swap_shrink["add"]), len(unselected_ranked))
    remove_count = min(int(swap_shrink["remove"]), parent_budget)
    kept = selected_ranked[: parent_budget - remove_count]
    proposals.append((f"swap_add{add_count}_remove{remove_count}", "bounded_swap_shrink", kept + unselected_ranked[:add_count], {"removed": remove_count, "added": add_count}))

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    mask_dir = branch_dir / "candidate_masks"
    for candidate_id, operator, order, lineage in proposals:
        receipt = write_mask(mask_dir / f"{candidate_id}.npz", order, shape=parent_scores.shape)
        if receipt["selected_sha256"] in seen:
            continue
        seen.add(receipt["selected_sha256"])
        candidates.append({"candidate_id": candidate_id, "operator": operator, **receipt, "lineage": lineage})
    write_json(branch_dir / "candidate_manifest.json", {"parent_budget": parent_budget, "candidates": candidates})
    return candidates


def evaluation_command(*, pairs: Path, output: Path, model: Path, config: Mapping[str, Any], mask: Path | None = None, budget: int | None = None) -> list[str]:
    evaluation = config["evaluation"]
    command = [
        sys.executable,
        str(BFCL),
        "eval-mask",
        "--pairs",
        str(pairs),
        "--output",
        str(output),
        "--model",
        str(model),
        "--dtype",
        str(evaluation["dtype"]),
        "--device-map",
        "auto",
        "--max-new-tokens",
        str(evaluation["max_new_tokens"]),
        "--batch-size",
        str(evaluation["batch_size"]),
        "--normalized",
    ]
    if evaluation["bfcl_canonicalization_prompt"]:
        command.append("--bfcl-canonicalization-prompt")
    if mask is not None:
        if budget is None:
            raise ValueError("masked evaluation requires budget")
        command.extend(["--attribution", str(mask), "--topk", str(budget)])
    return command


def evaluate(*, pairs: Path, output: Path, model: Path, config: Mapping[str, Any], event_dir: Path, mask: Path | None = None, budget: int | None = None) -> dict[str, Any]:
    summary_path = output.with_suffix(".summary.json")
    if not summary_path.exists():
        run_command(evaluation_command(pairs=pairs, output=output, model=model, config=config, mask=mask, budget=budget), output_dir=event_dir)
    return json.loads(summary_path.read_text())


def category_metrics(predictions: Path, pairs: Path) -> dict[str, Any]:
    pair_by_id = {str(row["id"]): row for row in read_jsonl(pairs)}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "n": 0})
    for prediction in read_jsonl(predictions):
        category = str(pair_by_id[str(prediction["id"])].get("category", "unknown"))
        totals[category]["n"] += 1
        totals[category]["correct"] += int(bool(prediction["correct"]))
    return {category: {**row, "accuracy": row["correct"] / row["n"] if row["n"] else None} for category, row in sorted(totals.items())}


def recovery(masked: Mapping[str, Any], dense: Mapping[str, Any]) -> float:
    denominator = int(dense["normalized_exact_correct"])
    return int(masked["normalized_exact_correct"]) / denominator if denominator else 0.0


def category_recovery(masked: Mapping[str, Any], dense: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for category, dense_row in dense.items():
        denominator = int(dense_row["correct"])
        if denominator:
            result[category] = int(masked.get(category, {}).get("correct", 0)) / denominator
    return result


def build_failure_buckets(*, predictions: Path, pairs: Path, out_dir: Path, run_name: str, event_dir: Path) -> None:
    if out_dir.exists():
        return
    run_command([
        sys.executable,
        str(FAILURE_BUCKETS),
        "--eval-jsonl",
        str(predictions),
        "--pairs-jsonl",
        str(pairs),
        "--out-dir",
        str(out_dir),
        "--run-name",
        run_name,
    ], output_dir=event_dir)


def prepare_splits(pairs: Path, output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    split_dir = output_dir / "splits"
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    rows = read_jsonl(pairs)
    contract = config["contract"]
    if len(rows) != int(contract["eval_rows"]):
        raise ValueError(f"expected {contract['eval_rows']} BFCL rows, got {len(rows)}")
    development, terminal = stratified_split(rows, seed=int(contract["split_seed"]), terminal_fraction=float(contract["terminal_fraction"]))
    calibration = stratified_subset(development, count=int(config["search"]["calibration_rows"]), seed=int(contract["split_seed"]), purpose="calibration")
    attribution = stratified_subset(development, count=int(config["search"]["attribution_rows"]), seed=int(contract["split_seed"]), purpose="attribution")
    paths = {
        "development": split_dir / "development.jsonl",
        "terminal": split_dir / "terminal.jsonl",
        "calibration": split_dir / "calibration.jsonl",
        "attribution": split_dir / "attribution.jsonl",
    }
    write_jsonl(paths["development"], development)
    write_jsonl(paths["terminal"], terminal)
    write_jsonl(paths["calibration"], calibration)
    write_jsonl(paths["attribution"], attribution)
    development_ids = {str(row["id"]) for row in development}
    terminal_ids = {str(row["id"]) for row in terminal}
    if development_ids & terminal_ids:
        raise ValueError("development and terminal rows overlap")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": contract["split_seed"],
        "source_pairs": str(pairs),
        "source_pairs_sha256": sha256_file(pairs),
        "counts": {name: sum(1 for _ in read_jsonl(path)) for name, path in paths.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "development_terminal_disjoint": True,
        "terminal_scored": False,
        "terminal_ids_sha256": hashlib.sha256("\n".join(sorted(terminal_ids)).encode()).hexdigest(),
    }
    write_json(manifest_path, manifest)
    return manifest


def prepare_root(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir
    root_summary_path = output_dir / "root" / "checkpoint.json"
    if root_summary_path.exists():
        return json.loads(root_summary_path.read_text())
    if args.pairs is None or args.base_train_jsonl is None or args.root_model is None or args.root_mask is None:
        raise ValueError("run requires pairs, base training data, root model, and root mask")
    split = prepare_splits(args.pairs, output_dir, config)
    contract = config["contract"]
    mask_scores, root_order = load_mask(args.root_mask, int(contract["root_mask_budget"]))
    del mask_scores
    observed_sha1 = selected_sha1(root_order)
    if observed_sha1 != contract["root_mask_selected_sha1"]:
        raise ValueError(f"root mask SHA-1 mismatch: {observed_sha1} != {contract['root_mask_selected_sha1']}")
    root_dir = output_dir / "root"
    development = Path(split["paths"]["development"])
    calibration = Path(split["paths"]["calibration"])
    dev_dense_output = root_dir / "development_dense.jsonl"
    dev_masked_output = root_dir / "development_masked.jsonl"
    cal_dense_output = root_dir / "calibration_dense.jsonl"
    cal_masked_output = root_dir / "calibration_masked.jsonl"
    dev_dense = evaluate(pairs=development, output=dev_dense_output, model=args.root_model, config=config, event_dir=root_dir)
    dev_masked = evaluate(pairs=development, output=dev_masked_output, model=args.root_model, config=config, event_dir=root_dir, mask=args.root_mask, budget=int(contract["root_mask_budget"]))
    cal_dense = evaluate(pairs=calibration, output=cal_dense_output, model=args.root_model, config=config, event_dir=root_dir)
    cal_masked = evaluate(pairs=calibration, output=cal_masked_output, model=args.root_model, config=config, event_dir=root_dir, mask=args.root_mask, budget=int(contract["root_mask_budget"]))
    dev_dense_categories = category_metrics(dev_dense_output, development)
    dev_masked_categories = category_metrics(dev_masked_output, development)
    failure_dir = root_dir / "failure_buckets"
    build_failure_buckets(predictions=dev_masked_output, pairs=development, out_dir=failure_dir, run_name="issue24_root", event_dir=root_dir)
    summary = {
        "checkpoint_id": "root",
        "accepted": True,
        "branch_index": 0,
        "depth": 0,
        "model_path": str(args.root_model),
        "mask_path": str(args.root_mask),
        "budget": int(contract["root_mask_budget"]),
        "selected_sha1": observed_sha1,
        "selected_sha256": selected_sha256(root_order, int(contract["total_mlp_channels"])),
        "train_jsonl": str(args.base_train_jsonl),
        "failure_bucket_dir": str(failure_dir),
        "development_dense": dev_dense,
        "development_masked": dev_masked,
        "development_recovery": recovery(dev_masked, dev_dense),
        "development_dense_categories": dev_dense_categories,
        "development_masked_categories": dev_masked_categories,
        "development_category_recovery": category_recovery(dev_masked_categories, dev_dense_categories),
        "calibration_dense": cal_dense,
        "calibration_masked": cal_masked,
        "calibration_recovery": recovery(cal_masked, cal_dense),
    }
    write_json(root_summary_path, summary)
    append_event(output_dir, {"stage": "root_prepared", "budget": summary["budget"], "development_recovery": summary["development_recovery"]})
    return summary


def checkpoint_key(checkpoint: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(checkpoint["budget"]),
        -int(checkpoint["development_masked"]["normalized_exact_correct"]),
        -float(checkpoint["development_recovery"]),
        int(checkpoint.get("branch_index", 0)),
    )


def branch_summaries(output_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((output_dir / "branches").glob("b*/branch_summary.json"))]


def eligible_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    root = json.loads((output_dir / "root" / "checkpoint.json").read_text())
    accepted = [summary["checkpoint"] for summary in branch_summaries(output_dir) if summary.get("accepted")]
    return sorted([root, *accepted], key=checkpoint_key)


def plan_wave(output_dir: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    completed = branch_summaries(output_dir)
    target = int(config["contract"]["trained_branch_attempts"])
    remaining = target - len(completed)
    if remaining <= 0:
        return []
    parallel = min(int(config["search"]["parallel_branches"]), remaining)
    beam = eligible_checkpoints(output_dir)[: int(config["search"]["beam_width"])]
    profiles = list(config["search"]["profiles"])
    specs: list[dict[str, Any]] = []
    next_index = len(completed) + 1
    for offset in range(parallel):
        parent = beam[offset % len(beam)]
        profile = profiles[(next_index + offset - 1) % len(profiles)]
        specs.append({
            "branch_id": f"b{next_index + offset:03d}",
            "branch_index": next_index + offset,
            "parent_id": parent["checkpoint_id"],
            "parent_checkpoint": parent,
            "depth": int(parent["depth"]) + 1,
            "profile": profile,
            "seed": int(config["training"]["seed_base"]) + (next_index + offset) * 1009,
        })
    return specs


def candidate_receipt(*, candidate: Mapping[str, Any], dense_summary: Mapping[str, Any], dense_categories: Mapping[str, Any], masked_summary: Mapping[str, Any], masked_categories: Mapping[str, Any], parent: Mapping[str, Any], root: Mapping[str, Any], config: Mapping[str, Any], calibration: bool) -> dict[str, Any]:
    candidate_recovery = recovery(masked_summary, dense_summary)
    per_category = category_recovery(masked_categories, dense_categories)
    gates = config["gates"]
    if calibration:
        gate = candidate_recovery >= float(gates["minimum_calibration_recovery"])
        reasons = [] if gate else ["calibration_recovery"]
    else:
        reasons: list[str] = []
        if candidate_recovery < float(gates["minimum_development_recovery"]):
            reasons.append("development_recovery")
        root_dense_correct = int(root["development_dense"]["normalized_exact_correct"])
        dense_correct = int(dense_summary["normalized_exact_correct"])
        if dense_correct < float(gates["minimum_dense_retention_of_root"]) * root_dense_correct:
            reasons.append("dense_retention")
        root_dense_categories = root["development_dense_categories"]
        minimum_category_dense_retention = float(gates.get("minimum_category_dense_retention_of_root", 0.95))
        for category, root_dense_row in root_dense_categories.items():
            minimum = minimum_category_dense_retention * int(root_dense_row["correct"])
            if int(dense_categories.get(category, {}).get("correct", 0)) < minimum:
                reasons.append(f"dense_category:{category}")
        root_recovery = root["development_category_recovery"]
        parent_recovery = parent["development_category_recovery"]
        for category in sorted(set(root_recovery) | set(parent_recovery)):
            floor = max(
                float(root_recovery.get(category, 0.0)) - float(gates["category_recovery_tolerance_below_root"]),
                float(parent_recovery.get(category, 0.0)) - float(gates["category_recovery_tolerance_below_parent"]),
            )
            if float(per_category.get(category, 0.0)) < floor:
                reasons.append(f"category_recovery:{category}")
        if int(candidate["budget"]) > int(parent["budget"]):
            reasons.append("budget_growth")
        if int(candidate["budget"]) == int(parent["budget"]):
            gain = int(masked_summary["normalized_exact_correct"]) - int(parent["development_masked"]["normalized_exact_correct"])
            if gain < int(gates["same_budget_minimum_masked_correct_gain"]):
                reasons.append("same_budget_no_gain")
        gate = not reasons
    parent_order = set(int(value) for value in load_mask(Path(parent["mask_path"]), int(parent["budget"]))[1])
    candidate_order = set(int(value) for value in load_mask(Path(candidate["path"]), int(candidate["budget"]))[1])
    return {
        **dict(candidate),
        "masked": dict(masked_summary),
        "masked_categories": dict(masked_categories),
        "recovery": candidate_recovery,
        "category_recovery": per_category,
        "parent_retention": len(parent_order & candidate_order) / len(parent_order),
        "quality_gate": gate,
        "gate_failures": reasons,
    }


def run_worker(args: argparse.Namespace, config: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    branch_id = str(spec["branch_id"])
    branch_dir = args.output_dir / "branches" / branch_id
    summary_path = branch_dir / "branch_summary.json"
    if summary_path.exists():
        return
    branch_dir.mkdir(parents=True, exist_ok=True)
    write_json(branch_dir / "spec.json", spec)
    parent = spec["parent_checkpoint"]
    split = json.loads((args.output_dir / "splits" / "manifest.json").read_text())
    development = Path(split["paths"]["development"])
    calibration = Path(split["paths"]["calibration"])
    attribution_pairs = Path(split["paths"]["attribution"])
    full_pairs = Path(split["source_pairs"])
    data_dir = branch_dir / "data"
    train_jsonl = data_dir / "train_mixed.jsonl"
    run_command([
        sys.executable,
        str(CURRICULUM),
        "--base-train-jsonl",
        str(parent["train_jsonl"]),
        "--eval-jsonl",
        str(development),
        "--failure-bucket-dirs",
        str(parent["failure_bucket_dir"]),
        "--edge-output",
        str(data_dir / "edge.jsonl"),
        "--mixed-output",
        str(train_jsonl),
        "--manifest",
        str(data_dir / "manifest.json"),
        "--branch-id",
        branch_id,
        "--parent-id",
        str(parent["checkpoint_id"]),
        "--branch-profile",
        str(spec["profile"]),
        "--seed",
        str(spec["seed"]),
        "--fail-on-leak",
    ], output_dir=branch_dir)
    run_command([
        sys.executable,
        str(LEAK_AUDIT),
        "--train-jsonl",
        str(train_jsonl),
        "--eval-jsonl",
        str(full_pairs),
        "--output",
        str(branch_dir / "leak_audit.json"),
        "--near-threshold",
        "0.85",
        "--fail-on-overlap",
    ], output_dir=branch_dir)
    training = config["training"]
    train_dir = branch_dir / "train"
    run_command([
        sys.executable,
        str(TRAINER),
        "--model",
        str(parent["model_path"]),
        "--train-jsonl",
        str(train_jsonl),
        "--attribution",
        str(parent["mask_path"]),
        "--topk",
        str(parent["budget"]),
        "--out-dir",
        str(train_dir),
        "--device",
        "cuda",
        "--dtype",
        str(config["evaluation"]["dtype"]),
        "--seed",
        str(spec["seed"]),
        "--max-seq-length",
        str(training["max_seq_length"]),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--grad-accum",
        str(training["grad_accum"]),
        "--lr",
        str(training["learning_rate"]),
        "--lora-r",
        str(training["lora_rank"]),
        "--lora-alpha",
        str(training["lora_alpha"]),
        "--lora-dropout",
        str(training["lora_dropout"]),
        "--masked-kl-beta",
        str(training["masked_kl_beta"]),
        "--ce-beta",
        str(training["ce_beta"]),
        "--unmasked-kl-beta",
        str(training["unmasked_kl_beta"]),
        "--save-merged",
    ], output_dir=branch_dir)
    merged_model = train_dir / "merged"
    attribution = branch_dir / "relp_attribution.npz"
    run_command([
        sys.executable,
        str(BFCL),
        "relp-attribute",
        "--pairs",
        str(attribution_pairs),
        "--output",
        str(attribution),
        "--model",
        str(merged_model),
        "--dtype",
        str(config["evaluation"]["dtype"]),
        "--device-map",
        "auto",
        "--log-every",
        "16",
        "--report-topk",
        "100",
    ], output_dir=branch_dir)
    candidates = generate_candidates(Path(parent["mask_path"]), int(parent["budget"]), attribution, branch_dir, config)

    calibration_dense_output = branch_dir / "calibration_dense.jsonl"
    calibration_dense = evaluate(pairs=calibration, output=calibration_dense_output, model=merged_model, config=config, event_dir=branch_dir)
    calibration_dense_categories = category_metrics(calibration_dense_output, calibration)
    calibration_receipts: list[dict[str, Any]] = []
    for candidate in candidates:
        output = branch_dir / "calibration" / f"{candidate['candidate_id']}.jsonl"
        masked = evaluate(pairs=calibration, output=output, model=merged_model, config=config, event_dir=branch_dir, mask=Path(candidate["path"]), budget=int(candidate["budget"]))
        masked_categories = category_metrics(output, calibration)
        calibration_receipts.append(candidate_receipt(candidate=candidate, dense_summary=calibration_dense, dense_categories=calibration_dense_categories, masked_summary=masked, masked_categories=masked_categories, parent=parent, root=json.loads((args.output_dir / "root" / "checkpoint.json").read_text()), config=config, calibration=True))
    calibration_receipts.sort(key=lambda row: (not row["quality_gate"], int(row["budget"]), -int(row["masked"]["normalized_exact_correct"]), row["candidate_id"]))
    promoted = calibration_receipts[: int(config["search"]["calibration_promotions"])]

    development_dense_output = branch_dir / "development_dense.jsonl"
    development_dense = evaluate(pairs=development, output=development_dense_output, model=merged_model, config=config, event_dir=branch_dir)
    development_dense_categories = category_metrics(development_dense_output, development)
    root = json.loads((args.output_dir / "root" / "checkpoint.json").read_text())
    development_receipts: list[dict[str, Any]] = []
    for candidate in promoted:
        output = branch_dir / "development" / f"{candidate['candidate_id']}.jsonl"
        masked = evaluate(pairs=development, output=output, model=merged_model, config=config, event_dir=branch_dir, mask=Path(candidate["path"]), budget=int(candidate["budget"]))
        masked_categories = category_metrics(output, development)
        receipt = candidate_receipt(candidate=candidate, dense_summary=development_dense, dense_categories=development_dense_categories, masked_summary=masked, masked_categories=masked_categories, parent=parent, root=root, config=config, calibration=False)
        receipt["predictions"] = str(output)
        development_receipts.append(receipt)
    passing = [row for row in development_receipts if row["quality_gate"]]
    passing.sort(key=lambda row: (int(row["budget"]), -int(row["masked"]["normalized_exact_correct"]), -float(row["recovery"]), -float(row["parent_retention"]), row["candidate_id"]))
    accepted = bool(passing)
    selected = passing[0] if accepted else None
    checkpoint: dict[str, Any] | None = None
    if selected is not None:
        failure_dir = branch_dir / "failure_buckets"
        build_failure_buckets(predictions=Path(selected["predictions"]), pairs=development, out_dir=failure_dir, run_name=f"issue24_{branch_id}", event_dir=branch_dir)
        checkpoint = {
            "checkpoint_id": branch_id,
            "accepted": True,
            "branch_index": int(spec["branch_index"]),
            "depth": int(spec["depth"]),
            "model_path": str(merged_model),
            "mask_path": str(selected["path"]),
            "budget": int(selected["budget"]),
            "selected_sha1": selected["selected_sha1"],
            "selected_sha256": selected["selected_sha256"],
            "train_jsonl": str(train_jsonl),
            "failure_bucket_dir": str(failure_dir),
            "development_dense": development_dense,
            "development_masked": selected["masked"],
            "development_recovery": selected["recovery"],
            "development_dense_categories": development_dense_categories,
            "development_masked_categories": selected["masked_categories"],
            "development_category_recovery": selected["category_recovery"],
            "calibration_dense": calibration_dense,
            "calibration_masked": next(row["masked"] for row in calibration_receipts if row["candidate_id"] == selected["candidate_id"]),
            "calibration_recovery": next(row["recovery"] for row in calibration_receipts if row["candidate_id"] == selected["candidate_id"]),
        }
    uniform_target = float(config["gates"]["category_uniform_target"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "branch_id": branch_id,
        "branch_index": int(spec["branch_index"]),
        "parent_id": parent["checkpoint_id"],
        "depth": int(spec["depth"]),
        "profile": spec["profile"],
        "seed": spec["seed"],
        "accepted": accepted,
        "checkpoint": checkpoint,
        "train_summary": str(train_dir / "train_summary.json"),
        "adapter_path": str(train_dir / "adapter"),
        "attribution": str(attribution),
        "calibration_candidates": calibration_receipts,
        "development_candidates": development_receipts,
        "passing_candidates": len(passing),
        "selected_candidate": selected,
        "category_uniform_target_pass": bool(selected) and all(float(value) >= uniform_target for value in selected["category_recovery"].values()),
    }
    write_json(summary_path, summary)
    append_event(args.output_dir, {"stage": "branch_complete", "branch_id": branch_id, "accepted": accepted, "budget": checkpoint["budget"] if checkpoint else parent["budget"]})


def terminal_guard(output_dir: Path, config: Mapping[str, Any]) -> None:
    completed = len(branch_summaries(output_dir))
    required = int(config["contract"]["trained_branch_attempts"])
    if completed != required:
        raise RuntimeError(f"terminal holdout sealed: {completed}/{required} branches complete")


def finalize(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    terminal_guard(args.output_dir, config)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    split = json.loads((args.output_dir / "splits" / "manifest.json").read_text())
    terminal = Path(split["paths"]["terminal"])
    full_pairs = Path(split["source_pairs"])
    root = json.loads((args.output_dir / "root" / "checkpoint.json").read_text())
    winner = eligible_checkpoints(args.output_dir)[0]
    final_dir = args.output_dir / "terminal"

    def score_checkpoint(label: str, checkpoint: Mapping[str, Any], pairs: Path) -> dict[str, Any]:
        dense_output = final_dir / f"{label}_dense.jsonl"
        masked_output = final_dir / f"{label}_masked.jsonl"
        dense = evaluate(pairs=pairs, output=dense_output, model=Path(checkpoint["model_path"]), config=config, event_dir=final_dir)
        masked = evaluate(pairs=pairs, output=masked_output, model=Path(checkpoint["model_path"]), config=config, event_dir=final_dir, mask=Path(checkpoint["mask_path"]), budget=int(checkpoint["budget"]))
        dense_categories = category_metrics(dense_output, pairs)
        masked_categories = category_metrics(masked_output, pairs)
        return {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "budget": checkpoint["budget"],
            "dense": dense,
            "masked": masked,
            "recovery": recovery(masked, dense),
            "dense_categories": dense_categories,
            "masked_categories": masked_categories,
            "category_recovery": category_recovery(masked_categories, dense_categories),
            "dense_predictions": str(dense_output),
            "masked_predictions": str(masked_output),
        }

    terminal_root = score_checkpoint("terminal_root", root, terminal)
    terminal_winner = score_checkpoint("terminal_winner", winner, terminal)
    full_root = score_checkpoint("full_root", root, full_pairs)
    full_winner = score_checkpoint("full_winner", winner, full_pairs)
    uniform_target = float(config["gates"]["category_uniform_target"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "branches_attempted": len(branch_summaries(args.output_dir)),
        "winner": winner,
        "terminal_root": terminal_root,
        "terminal_winner": terminal_winner,
        "terminal_category_uniform_pass": all(value >= uniform_target for value in terminal_winner["category_recovery"].values()),
        "full_root": full_root,
        "full_winner": full_winner,
        "historical_comparison": {
            "dense_correct": config["contract"]["historical_dense_correct"],
            "root_masked_correct": config["contract"]["historical_root_masked_correct"],
        },
        "tree": branch_summaries(args.output_dir),
    }
    write_json(result_path, result)
    split["terminal_scored"] = True
    split["terminal_scored_after_branches"] = len(branch_summaries(args.output_dir))
    write_json(args.output_dir / "splits" / "manifest.json", split)
    append_event(args.output_dir, {"stage": "terminal_complete", "winner": winner["checkpoint_id"], "budget": winner["budget"], "recovery": terminal_winner["recovery"]})
    return result


def run_tree(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "config.json")
    prepare_root(args, config)
    while len(branch_summaries(args.output_dir)) < int(config["contract"]["trained_branch_attempts"]):
        specs = plan_wave(args.output_dir, config)
        if not specs:
            raise RuntimeError("planner returned no branches before the fixed budget completed")
        wave_index = len(list((args.output_dir / "plans").glob("wave_*.json"))) + 1
        plan_path = args.output_dir / "plans" / f"wave_{wave_index:02d}.json"
        write_json(plan_path, specs)
        processes: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
        for offset, spec in enumerate(specs):
            branch_dir = args.output_dir / "branches" / spec["branch_id"]
            spec_path = branch_dir / "worker_spec.json"
            write_json(spec_path, spec)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(offset % args.device_count)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(args.config),
                "--output-dir",
                str(args.output_dir),
                "--worker-spec",
                str(spec_path),
            ]
            log_path = branch_dir / "worker.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            process._issue24_log_handle = log_handle  # type: ignore[attr-defined]
            processes.append((spec, process))
        failures: list[dict[str, Any]] = []
        for spec, process in processes:
            return_code = process.wait()
            process._issue24_log_handle.close()  # type: ignore[attr-defined]
            if return_code != 0:
                failures.append({"branch_id": spec["branch_id"], "return_code": return_code})
        if failures:
            append_event(args.output_dir, {"stage": "substantive_failure", "failures": failures})
            raise RuntimeError(f"branch wave failed; preserved state: {failures}")
        append_event(args.output_dir, {"stage": "wave_complete", "wave": wave_index, "branches_completed": len(branch_summaries(args.output_dir))})
    return finalize(args, config)


def self_check(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="issue24-self-check-") as temporary:
        root = Path(temporary)
        shape = (4, 16)
        parent = write_mask(root / "parent.npz", list(range(40)), shape=shape)
        attribution_scores = np.arange(64, dtype=np.float32).reshape(shape)
        np.savez_compressed(root / "attribution.npz", mlp_scores=attribution_scores)
        tiny_config = json.loads(json.dumps(config))
        tiny_config["search"]["minimum_budget"] = 4
        tiny_config["search"]["shrink_deltas"] = [1, 2, 4, 8]
        tiny_config["search"]["swap_counts"] = [1, 2]
        tiny_config["search"]["swap_shrink"] = {"add": 1, "remove": 3}
        candidates = generate_candidates(root / "parent.npz", parent["budget"], root / "attribution.npz", root, tiny_config)
        if not candidates or any(candidate["budget"] > parent["budget"] for candidate in candidates):
            raise AssertionError("candidate generator violated parent budget")
        output = root / "run"
        (output / "root").mkdir(parents=True)
        mock_checkpoint = {
            "checkpoint_id": "root",
            "accepted": True,
            "branch_index": 0,
            "depth": 0,
            "budget": 40,
            "development_dense": {"normalized_exact_correct": 10},
            "development_masked": {"normalized_exact_correct": 9},
            "development_recovery": 0.9,
        }
        write_json(output / "root" / "checkpoint.json", mock_checkpoint)
        planned_counts: list[int] = []
        completed = 0
        while completed < 20:
            specs = plan_wave(output, config)
            planned_counts.append(len(specs))
            for spec in specs:
                branch_dir = output / "branches" / spec["branch_id"]
                write_json(branch_dir / "branch_summary.json", {"accepted": False, "branch_id": spec["branch_id"], "branch_index": spec["branch_index"], "checkpoint": None})
            completed += len(specs)
            if completed < 20:
                try:
                    terminal_guard(output, config)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"terminal guard opened after only {completed} branches")
        if completed != 20 or planned_counts != [8, 8, 4]:
            raise AssertionError(f"unexpected fixed-budget plan: completed={completed}, waves={planned_counts}")
        terminal_guard(output, config)
        write_json(args.output_dir / "self_check.json", {"status": "pass", "candidate_count": len(candidates), "wave_sizes": planned_counts, "branches": completed})
        print(json.dumps({"status": "pass", "candidate_count": len(candidates), "wave_sizes": planned_counts, "branches": completed}, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.self_check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        self_check(args, config)
        return
    if args.worker_spec is not None:
        spec = json.loads(args.worker_spec.read_text())
        run_worker(args, config, spec)
        return
    result = run_tree(args, config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
