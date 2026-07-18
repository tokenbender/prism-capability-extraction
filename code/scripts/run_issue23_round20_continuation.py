#!/usr/bin/env python3
"""Continue the Issue #23 arithmetic tree from R8 through R20 in parallel."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

def load_runtime_dependencies() -> None:
    from scripts.run_issue23_best_checkpoint_tree import (
        evaluate_candidates as imported_evaluate_candidates,
        load_model_chain as imported_load_model_chain,
        selected_count as imported_selected_count,
        winner_key as imported_winner_key,
    )
    from scripts.run_issue23_eight_round_repair import (
        evaluate_and_save as imported_evaluate_and_save,
        passes_recovery_gate as imported_passes_recovery_gate,
        recovery as imported_recovery,
        round_pair_roles as imported_round_pair_roles,
        training_args as imported_training_args,
    )
    from scripts.run_issue23_halfhour_pilot import (
        PAIR_SPLIT_SEED as imported_pair_split_seed,
        RunLedger as imported_run_ledger,
        attribute_records as imported_attribute_records,
        build_pair_splits as imported_build_pair_splits,
        build_training_examples as imported_build_training_examples,
        evaluation_records as imported_evaluation_records,
        load_selected_mlp_mask as imported_load_selected_mlp_mask,
        mlp_widths as imported_mlp_widths,
        sha256_file as imported_sha256_file,
        train_repair_adapter as imported_train_repair_adapter,
        write_json as imported_write_json,
        write_jsonl as imported_write_jsonl,
    )
    globals().update({
        "evaluate_candidates": imported_evaluate_candidates,
        "load_model_chain": imported_load_model_chain,
        "selected_count": imported_selected_count,
        "winner_key": imported_winner_key,
        "evaluate_and_save": imported_evaluate_and_save,
        "passes_recovery_gate": imported_passes_recovery_gate,
        "recovery": imported_recovery,
        "round_pair_roles": imported_round_pair_roles,
        "training_args": imported_training_args,
        "PAIR_SPLIT_SEED": imported_pair_split_seed,
        "RunLedger": imported_run_ledger,
        "attribute_records": imported_attribute_records,
        "build_pair_splits": imported_build_pair_splits,
        "build_training_examples": imported_build_training_examples,
        "evaluation_records": imported_evaluation_records,
        "load_selected_mlp_mask": imported_load_selected_mlp_mask,
        "mlp_widths": imported_mlp_widths,
        "sha256_file": imported_sha256_file,
        "train_repair_adapter": imported_train_repair_adapter,
        "write_json": imported_write_json,
        "write_jsonl": imported_write_jsonl,
    })

SCHEMA_VERSION = "prism_arithmetic_issue23_round20_v1"
CONFIG_SCHEMA_VERSION = "prism_arithmetic_issue23_round20_config_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--root-adapter", type=Path, action="append", default=[])
    parser.add_argument("--root-mask", type=Path)
    parser.add_argument("--baseline-mask", type=Path)
    parser.add_argument("--baseline-mask-key", default="mlp_rel_0.001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=float, default=43_200.0)
    parser.add_argument("--device-count", type=int, default=8)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema in {path}")
    contract = config["contract"]
    if int(contract["root_round"]) != 8 or int(contract["target_round"]) != 20:
        raise ValueError("Issue #23 continuation must run literal rounds R9 through R20")
    if config["selection"].get("continue_after_exhausted_depth") is not True:
        raise ValueError("continuation must attempt every depth through R20")
    branches = int(config["round"]["repair_branches"])
    parallel = int(config["execution"]["parallel_branches"])
    if branches != 8 or parallel != 8:
        raise ValueError("arithmetic continuation contract requires eight parallel branches")
    budgets = [int(value) for value in config["selection"]["budgets_ascending"]]
    if budgets != sorted(set(budgets)) or budgets[-1] != int(contract["root_budget"]):
        raise ValueError("selection budgets must be unique, ascending, and end at root budget")
    retentions = [float(value) for value in config["selection"]["retention_fractions"]]
    if not retentions or retentions[0] != 1.0 or any(not 0.0 < value <= 1.0 for value in retentions):
        raise ValueError("retention fractions must start at 1.0 and stay in (0, 1]")
    return config


def require_coordinator_args(args: argparse.Namespace) -> None:
    missing = []
    for name in ("model", "tokenizer", "root_mask", "baseline_mask"):
        if getattr(args, name) in (None, ""):
            missing.append(name)
    if not args.root_adapter:
        missing.append("root_adapter")
    if missing:
        raise ValueError(f"missing coordinator arguments: {', '.join(missing)}")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def worker_args(args: argparse.Namespace, spec_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--model",
        str(args.model),
        "--tokenizer",
        str(args.tokenizer),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--deadline-seconds",
        str(args.deadline_seconds),
        "--device-count",
        str(args.device_count),
        "--worker-spec",
        str(spec_path),
    ]
    return command


def branch_worker(args: argparse.Namespace, config: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    depth = int(spec["depth"])
    branch = int(spec["branch"])
    pseudo_round = depth * 100 + branch
    seed_all(args.seed + pseudo_round)
    branch_dir = Path(spec["branch_dir"])
    result_path = branch_dir / "worker_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    branch_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(branch_dir, args.deadline_seconds)
    splits = build_pair_splits(PAIR_SPLIT_SEED)
    round_config = config["round"]
    selection = config["selection"]
    development_records = evaluation_records(
        splits["screen"],
        int(round_config["development_screen_rows"]),
        seed=args.seed,
        purpose=str(selection["expected_root_control"]["screen_builder_purpose"]),
    )
    roles = round_pair_roles(splits["search"], seed=args.seed, round_index=pseudo_round)
    mining_records = evaluation_records(
        roles["mining"],
        int(round_config["mining_rows"]),
        seed=args.seed + pseudo_round,
        purpose=f"round_{depth}_branch_{branch}_mining",
    )
    attribution_records = evaluation_records(
        roles["attribution"],
        int(round_config["attribution_rows"]),
        seed=args.seed + pseudo_round,
        purpose=f"round_{depth}_branch_{branch}_attribution",
    )
    calibration_records = evaluation_records(
        roles["calibration"],
        int(round_config["calibration_rows"]),
        seed=args.seed + pseudo_round,
        purpose=f"round_{depth}_branch_{branch}_calibration",
    )
    write_jsonl(branch_dir / "mining_records.jsonl", mining_records)
    write_jsonl(branch_dir / "attribution_records.jsonl", attribution_records)
    write_jsonl(branch_dir / "calibration_records.jsonl", calibration_records)

    current_chain = [Path(value) for value in spec["current_chain"]]
    model = load_model_chain(args, current_chain)
    widths = mlp_widths(model)
    current_selected, current_mask_receipt = load_selected_mlp_mask(
        Path(spec["current_mask"]), widths=widths, key="mlp_final"
    )
    if selected_count(current_selected) != int(spec["current_budget"]):
        raise RuntimeError("worker parent mask budget mismatch")
    mining_summary = evaluate_and_save(
        model,
        AutoTokenizerProxy(args.tokenizer),
        mining_records,
        selected=current_selected,
        args=args,
        output=branch_dir / "parent_masked_mining_predictions.jsonl",
    )
    with (branch_dir / "parent_masked_mining_predictions.jsonl").open(encoding="utf-8") as handle:
        mining_predictions = [json.loads(line) for line in handle if line.strip()]
    failures = [row for row in mining_predictions if not bool(row["exact_numeric_correct"])]
    write_jsonl(branch_dir / "mined_failures.jsonl", failures)
    examples, training_receipt = build_training_examples(
        roles["training_pool"],
        failures,
        base_rows=int(round_config["training_base_rows"]),
        repair_rows=int(round_config["failure_repair_rows"]),
        seed=args.seed + pseudo_round * 10_003,
    )
    write_jsonl(
        branch_dir / "training_examples.jsonl",
        (
            {
                "a": example.a,
                "b": example.b,
                "answer": example.answer,
                "train_prompt": example.train_prompt,
                "variant": example.variant,
                "carry_signature": example.carry_signature,
            }
            for example in examples
        ),
    )
    write_json(branch_dir / "training_receipt.json", training_receipt)
    tokenizer = AutoTokenizerProxy(args.tokenizer)
    model, train_history = train_repair_adapter(
        model,
        tokenizer,
        examples,
        training_args(args, config, pseudo_round),
        ledger,
    )
    expected_steps = int(round_config["training"]["steps"])
    if len(train_history) != expected_steps:
        raise RuntimeError(f"round {depth} branch {branch} trained {len(train_history)}/{expected_steps} steps")
    write_json(branch_dir / "train_history.json", train_history)
    adapter_dir = branch_dir / "adapter"
    model.save_pretrained(adapter_dir)
    model = model.merge_and_unload()
    model.enable_input_require_grads()
    model.eval()
    scores = attribute_records(
        model,
        tokenizer,
        attribution_records,
        batch_size=int(round_config["attribution"]["batch_size"]),
        device=args.device,
        ledger=ledger,
    )
    model.disable_input_require_grads()
    np.savez_compressed(branch_dir / "attribution_scores.npz", mlp_scores=scores.numpy())
    calibration_dense = evaluate_and_save(
        model,
        tokenizer,
        calibration_records,
        selected=None,
        args=args,
        output=branch_dir / "calibration_dense_predictions.jsonl",
    )
    development_dense = evaluate_and_save(
        model,
        tokenizer,
        development_records,
        selected=None,
        args=args,
        output=branch_dir / "development_dense_predictions.jsonl",
    )
    dense_floor = int(selection["minimum_dense_correct_vs_round_zero_screen"])
    dense_pass = int(development_dense["correct"]) >= dense_floor
    receipts: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    if dense_pass:
        receipts, raw_passing = evaluate_candidates(
            model=model,
            tokenizer=tokenizer,
            scores=scores,
            parent_selected=current_selected,
            parent_screen_masked_correct=int(spec["current_screen_masked_correct"]),
            widths=widths,
            calibration_records=calibration_records,
            development_records=development_records,
            calibration_dense=calibration_dense,
            development_dense=development_dense,
            branch_dir=branch_dir,
            args=args,
            config=config,
        )
        for candidate in raw_passing:
            passing.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"pairs", "selected", "adapter_dir"}
                }
            )
    write_json(branch_dir / "candidate_receipts.json", receipts)
    summary = {
        "round": depth,
        "branch": branch,
        "parent_budget": int(spec["current_budget"]),
        "parent_mask_sha256": current_mask_receipt.get("mask_sha256"),
        "mining": mining_summary,
        "mined_failure_rows": len(failures),
        "training": training_receipt,
        "final_training_loss": train_history[-1],
        "dense_floor": dense_floor,
        "dense_pass": dense_pass,
        "calibration_dense": calibration_dense,
        "development_dense": development_dense,
        "passing_candidates": len(passing),
        "adapter_path": str(adapter_dir),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
    }
    write_json(branch_dir / "summary.json", summary)
    result = {"summary": summary, "passing": passing}
    write_json(result_path, result)
    return result


class AutoTokenizerProxy:
    """Load one tokenizer lazily per worker and reuse it."""

    _instances: dict[str, Any] = {}

    def __new__(cls, path: str) -> Any:
        from transformers import AutoTokenizer

        if path not in cls._instances:
            tokenizer = AutoTokenizer.from_pretrained(path)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            cls._instances[path] = tokenizer
        return cls._instances[path]


def run_wave(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    depth: int,
    depth_dir: Path,
    current_chain: Sequence[Path],
    current_mask: Path,
    current_budget: int,
    current_screen_masked_correct: int,
) -> list[dict[str, Any]]:
    specs: list[tuple[int, Path, Path]] = []
    for branch in range(int(config["round"]["repair_branches"])):
        branch_dir = depth_dir / f"branch_{branch:02d}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        spec_path = branch_dir / "worker_spec.json"
        write_json(
            spec_path,
            {
                "depth": depth,
                "branch": branch,
                "branch_dir": str(branch_dir),
                "current_chain": [str(path) for path in current_chain],
                "current_mask": str(current_mask),
                "current_budget": current_budget,
                "current_screen_masked_correct": current_screen_masked_correct,
            },
        )
        specs.append((branch, branch_dir, spec_path))

    active: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    for branch, branch_dir, spec_path in specs:
        if (branch_dir / "worker_result.json").exists():
            continue
        log_handle = (branch_dir / "worker.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(branch % args.device_count)
        process = subprocess.Popen(
            worker_args(args, spec_path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        active[branch] = (process, log_handle)

    while active:
        failed: tuple[int, int] | None = None
        for branch, (process, _) in active.items():
            code = process.poll()
            if code not in (None, 0):
                failed = (branch, int(code))
                break
        if failed is not None:
            for process, _ in active.values():
                if process.poll() is None:
                    process.terminate()
            for process, handle in active.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                handle.close()
            raise RuntimeError(f"round {depth} branch {failed[0]} failed with exit {failed[1]}")
        completed = [branch for branch, (process, _) in active.items() if process.poll() == 0]
        for branch in completed:
            process, handle = active.pop(branch)
            process.wait()
            handle.close()
        if active:
            time.sleep(2)

    return [json.loads((branch_dir / "worker_result.json").read_text()) for _, branch_dir, _ in specs]


def terminal_condition(
    args: argparse.Namespace,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    chain: Sequence[Path],
    mask_path: Path,
    widths: Sequence[int],
) -> dict[str, Any]:
    model = load_model_chain(args, chain)
    selected, receipt = load_selected_mlp_mask(mask_path, widths=widths, key="mlp_final")
    dense = evaluate_and_save(
        model,
        tokenizer,
        records,
        selected=None,
        args=args,
        output=args.output_dir / f"terminal_{label}_dense_predictions.jsonl",
    )
    masked = evaluate_and_save(
        model,
        tokenizer,
        records,
        selected=selected,
        args=args,
        output=args.output_dir / f"terminal_{label}_masked_predictions.jsonl",
    )
    overall, per_class = recovery(masked, dense)
    del model
    torch.cuda.empty_cache()
    return {
        "budget": selected_count(selected),
        "dense": dense,
        "masked": masked,
        "recovery": overall,
        "per_carry_class_recovery": per_class,
        "mask_receipt": receipt,
    }


def coordinator(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    require_coordinator_args(args)
    if args.device_count != int(config["execution"]["device_count"]):
        raise ValueError("device count differs from frozen contract")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(args.output_dir, args.deadline_seconds)
    write_json(args.output_dir / "config.json", config)
    seed_all(args.seed)
    splits = build_pair_splits(PAIR_SPLIT_SEED)
    round_config = config["round"]
    selection = config["selection"]
    contract = config["contract"]
    development_records = evaluation_records(
        splits["screen"],
        int(round_config["development_screen_rows"]),
        seed=args.seed,
        purpose=str(selection["expected_root_control"]["screen_builder_purpose"]),
    )
    write_jsonl(args.output_dir / "development_screen_records.jsonl", development_records)
    split_receipt = {
        "pair_split_seed": PAIR_SPLIT_SEED,
        "search_pairs": len(splits["search"]),
        "development_screen_pairs": len(splits["screen"]),
        "fixed_evaluation_pairs": len(splits["holdout"]),
        "pair_disjoint": True,
        "fixed_evaluation_previously_opened_at_r8": True,
        "fixed_evaluation_used_during_r9_r20_search": False,
    }
    write_json(args.output_dir / "split_receipt.json", split_receipt)
    ledger.finish("freeze_contract", **split_receipt)

    root_chain = list(args.root_adapter)
    root_model = load_model_chain(args, root_chain)
    widths = mlp_widths(root_model)
    root_selected, root_mask_receipt = load_selected_mlp_mask(args.root_mask, widths=widths, key="mlp_final")
    root_budget = selected_count(root_selected)
    if root_budget != int(contract["root_budget"]):
        raise RuntimeError(f"root budget mismatch: {root_budget}")
    tokenizer = AutoTokenizerProxy(str(args.tokenizer))
    root_dir = args.output_dir / "round_08_root"
    root_dir.mkdir()
    root_dense = evaluate_and_save(
        root_model,
        tokenizer,
        development_records,
        selected=None,
        args=args,
        output=root_dir / "development_dense_predictions.jsonl",
    )
    root_masked = evaluate_and_save(
        root_model,
        tokenizer,
        development_records,
        selected=root_selected,
        args=args,
        output=root_dir / "development_masked_predictions.jsonl",
    )
    observed_root = {
        "dense_correct": int(root_dense["correct"]),
        "masked_correct": int(root_masked["correct"]),
        "rows": int(root_dense["n"]),
    }
    expected_root = {
        "dense_correct": int(selection["expected_root_control"]["dense_correct"]),
        "masked_correct": int(selection["expected_root_control"]["masked_correct"]),
        "rows": int(selection["expected_root_control"]["rows"]),
    }
    if observed_root != expected_root:
        raise RuntimeError(f"R8 root control mismatch: observed {observed_root}, expected {expected_root}")
    root_recovery, root_per_class = recovery(root_masked, root_dense)
    progress: list[dict[str, Any]] = [{
        "round": 8,
        "status": "accepted_r8_root",
        "budget": root_budget,
        "dense": root_dense,
        "masked": root_masked,
        "recovery": root_recovery,
        "per_carry_class_recovery": root_per_class,
        "adapter_chain": [str(path) for path in root_chain],
        "adapter_chain_sha256": [sha256_file(path / "adapter_model.safetensors") for path in root_chain],
        "root_mask_sha256": sha256_file(args.root_mask),
    }]
    write_json(root_dir / "summary.json", progress[0])
    write_json(args.output_dir / "progress.json", progress)
    del root_model
    torch.cuda.empty_cache()

    current_round = 8
    current_budget = root_budget
    current_chain = root_chain
    current_mask = args.root_mask
    current_screen_masked_correct = int(root_masked["correct"])
    exhausted_depths: list[int] = []

    for depth in range(9, 21):
        ledger.require_time(f"round_{depth}_tree", 900)
        depth_dir = args.output_dir / f"round_{depth:02d}"
        depth_dir.mkdir()
        wave_results = run_wave(
            args,
            config,
            depth=depth,
            depth_dir=depth_dir,
            current_chain=current_chain,
            current_mask=current_mask,
            current_budget=current_budget,
            current_screen_masked_correct=current_screen_masked_correct,
        )
        branch_summaries = [result["summary"] for result in wave_results]
        passing_children: list[dict[str, Any]] = []
        for result in wave_results:
            branch = int(result["summary"]["branch"])
            for candidate in result["passing"]:
                passing_children.append({
                    **candidate,
                    "branch": branch,
                    "adapter_path": result["summary"]["adapter_path"],
                    "development_dense": result["summary"]["development_dense"],
                    "calibration_dense": result["summary"]["calibration_dense"],
                })
        if not passing_children:
            exhausted_depths.append(depth)
            retained = {
                "round": depth,
                "status": "incumbent_retained",
                "incumbent_round": current_round,
                "incumbent_budget": current_budget,
                "incumbent_masked_correct": current_screen_masked_correct,
                "branches": branch_summaries,
            }
            write_json(depth_dir / "summary.json", retained)
            progress.append(retained)
            write_json(args.output_dir / "progress.json", progress)
            ledger.finish(f"round_{depth}_incumbent_retained", incumbent_round=current_round, incumbent_budget=current_budget)
            continue

        winner = min(passing_children, key=winner_key)
        selected_mask_path = depth_dir / "selected_mask.npz"
        shutil.copy2(Path(winner["mask_path"]), selected_mask_path)
        parent_budget = current_budget
        current_round = depth
        current_budget = int(winner["budget"])
        current_mask = selected_mask_path
        current_chain = [*current_chain, Path(winner["adapter_path"])]
        current_screen_masked_correct = int(winner["development_masked"]["correct"])
        accepted = {
            "round": depth,
            "status": "accepted",
            "parent_budget": parent_budget,
            "selected_budget": current_budget,
            "selected_branch": int(winner["branch"]),
            "selected_operator": winner["operator"],
            "selected_retention_of_parent": winner["retention_of_parent"],
            "selected_retention_of_candidate": winner["retention_of_candidate"],
            "selected_calibration_dense": winner["calibration_dense"],
            "selected_calibration_masked": winner["calibration_masked"],
            "selected_development_dense": winner["development_dense"],
            "selected_development_masked": winner["development_masked"],
            "calibration_gate": winner["calibration_gate"],
            "development_gate": winner["development_gate"],
            "masked_correct_gain_vs_parent": winner["masked_correct_gain_vs_parent"],
            "selected_mask": str(selected_mask_path),
            "selected_mask_sha256": sha256_file(selected_mask_path),
            "selected_adapter": winner["adapter_path"],
            "selected_adapter_sha256": sha256_file(Path(winner["adapter_path"]) / "adapter_model.safetensors"),
            "passing_children": len(passing_children),
            "branches": branch_summaries,
        }
        write_json(depth_dir / "summary.json", accepted)
        progress.append(accepted)
        write_json(args.output_dir / "progress.json", progress)
        ledger.finish(
            f"round_{depth}_accepted",
            branch=int(winner["branch"]),
            parent_budget=parent_budget,
            selected_budget=current_budget,
            dense_correct=int(winner["development_dense"]["correct"]),
            masked_correct=current_screen_masked_correct,
        )

    holdout_records = evaluation_records(
        splits["holdout"],
        int(contract["terminal_holdout_rows"]),
        seed=args.seed,
        purpose="terminal_holdout",
    )
    write_jsonl(args.output_dir / "fixed_evaluation_records.jsonl", holdout_records)
    root_terminal = terminal_condition(
        args,
        tokenizer,
        holdout_records,
        label="r8_root",
        chain=root_chain,
        mask_path=args.root_mask,
        widths=widths,
    )
    expected_terminal = contract["expected_r8_terminal_control"]
    observed_terminal = {
        "dense_correct": int(root_terminal["dense"]["correct"]),
        "masked_correct": int(root_terminal["masked"]["correct"]),
        "rows": int(root_terminal["dense"]["n"]),
    }
    if observed_terminal != expected_terminal:
        raise RuntimeError(f"R8 fixed-evaluation control mismatch: observed {observed_terminal}, expected {expected_terminal}")
    winner_terminal = terminal_condition(
        args,
        tokenizer,
        holdout_records,
        label="winner",
        chain=current_chain,
        mask_path=current_mask,
        widths=widths,
    )
    winner_pass, winner_gate = passes_recovery_gate(
        winner_terminal["masked"],
        winner_terminal["dense"],
        minimum_overall=float(selection["minimum_overall_recovery"]),
        minimum_per_class=float(selection["minimum_per_carry_class_recovery"]),
    )
    terminal = {
        "evaluation_set_status": "fixed R8 holdout reused once after R20; not newly sealed",
        "rows": len(holdout_records),
        "pairs": len(splits["holdout"]),
        "r8_root": root_terminal,
        "winner": winner_terminal,
        "winner_gate": winner_gate,
        "winner_pass": winner_pass,
    }
    write_json(args.output_dir / "fixed_evaluation_summary.json", terminal)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "root_round": 8,
        "target_round": 20,
        "winner_round": current_round,
        "winner_budget": current_budget,
        "last_attempted_round": 20,
        "exhausted_depths": exhausted_depths,
        "progress": progress,
        "fixed_evaluation": terminal,
        "elapsed_s": ledger.elapsed,
    }
    write_json(args.output_dir / "result.json", result)
    ledger.finish("complete", winner_round=current_round, winner_budget=current_budget, terminal_pass=winner_pass)
    return result


def self_check(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rounds = list(range(int(config["contract"]["root_round"]) + 1, int(config["contract"]["target_round"]) + 1))
    branches = int(config["round"]["repair_branches"])
    devices = int(config["execution"]["device_count"])
    assignments = [[branch % devices for branch in range(branches)] for _ in rounds]
    if rounds != list(range(9, 21)) or len(rounds) != 12:
        raise AssertionError("continuation does not cover exactly R9-R20")
    if any(sorted(row) != list(range(8)) for row in assignments):
        raise AssertionError("a round does not map one branch to every GPU")
    result = {
        "status": "pass",
        "rounds": rounds,
        "round_count": len(rounds),
        "branches_per_round": branches,
        "trained_branch_attempts": len(rounds) * branches,
        "device_assignments": assignments[0],
        "fixed_evaluation_previously_opened": True,
    }
    (args.output_dir / "self_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.self_check:
        print(json.dumps(self_check(config, args), indent=2, sort_keys=True), flush=True)
        return
    load_runtime_dependencies()
    if args.worker_spec is not None:
        spec = json.loads(args.worker_spec.read_text())
        try:
            result = branch_worker(args, config, spec)
        except BaseException as error:
            write_json(
                Path(spec["branch_dir"]) / "failure.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "fail",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
        return
    try:
        result = coordinator(args, config)
    except BaseException as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "fail",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
