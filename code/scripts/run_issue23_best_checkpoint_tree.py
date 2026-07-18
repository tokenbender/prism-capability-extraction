#!/usr/bin/env python3
"""Run the Issue #23 failure-driven best-checkpoint arithmetic SFT tree."""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.run_issue23_eight_round_repair import (  # noqa: E402
    evaluate_and_save,
    passes_recovery_gate,
    recovery,
    round_pair_roles,
    training_args,
)
from scripts.run_issue23_halfhour_pilot import (  # noqa: E402
    PAIR_SPLIT_SEED,
    RunLedger,
    attribute_records,
    build_pair_splits,
    build_training_examples,
    evaluation_records,
    load_selected_mlp_mask,
    mlp_widths,
    sha256_file,
    train_repair_adapter,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = "prism_arithmetic_issue23_best_checkpoint_tree_v2"
CONFIG_SCHEMA_VERSION = "prism_arithmetic_issue23_best_checkpoint_tree_config_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--root-adapter", type=Path, required=True)
    parser.add_argument("--root-mask", type=Path, required=True)
    parser.add_argument("--baseline-mask", type=Path, required=True)
    parser.add_argument("--baseline-mask-key", default="mlp_rel_0.001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=float, default=2_100.0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unexpected config schema in {path}")
    contract = config["contract"]
    root_round = int(contract["root_round"])
    target_round = int(contract["target_round"])
    if root_round != 1 or target_round != 8:
        raise ValueError("Issue #23 tree contract must continue R1 through R8")
    if config["selection"].get("continue_after_exhausted_depth") is not True:
        raise ValueError("Issue #23 corrected contract must attempt every depth through R8")
    budgets = [int(value) for value in config["selection"]["budgets_ascending"]]
    if budgets != sorted(set(budgets)):
        raise ValueError("selection budgets must be unique and ascending")
    retentions = [float(value) for value in config["selection"]["retention_fractions"]]
    if not retentions or retentions[0] != 1.0 or any(not 0.0 < value <= 1.0 for value in retentions):
        raise ValueError("retention fractions must start at 1.0 and stay in (0, 1]")
    return config


class BranchLedgerView:
    """Prefix helper-stage events with the owning depth and repair branch."""

    def __init__(self, ledger: RunLedger, depth: int, branch: int):
        self.ledger = ledger
        self.prefix = f"round_{depth}_branch_{branch}"

    @property
    def remaining(self) -> float:
        return self.ledger.remaining


    def require_time(self, stage: str, minimum_remaining: float) -> None:
        self.ledger.require_time(f"{self.prefix}_{stage}", minimum_remaining)

    def finish(self, stage: str, **values: Any) -> None:
        self.ledger.finish(f"{self.prefix}_{stage}", **values)

    def event(self, stage: str, **values: Any) -> None:
        self.ledger.event(f"{self.prefix}_{stage}", **values)


def load_model_chain(
    args: argparse.Namespace,
    adapter_chain: Sequence[Path],
) -> torch.nn.Module:
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    for adapter_path in adapter_chain:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def selected_count(selected: Mapping[int, torch.Tensor]) -> int:
    return sum(int(mask.sum().item()) for mask in selected.values())


def flatten_selected(
    selected: Mapping[int, torch.Tensor], widths: Sequence[int]
) -> np.ndarray:
    return np.concatenate(
        [selected[layer].detach().cpu().numpy().astype(np.bool_)[:width] for layer, width in enumerate(widths)]
    )


def selected_from_flat(
    flat: np.ndarray, widths: Sequence[int]
) -> tuple[np.ndarray, dict[int, torch.Tensor]]:
    selected: dict[int, torch.Tensor] = {}
    pairs: list[tuple[int, int]] = []
    offset = 0
    for layer, width in enumerate(widths):
        layer_flat = flat[offset : offset + width]
        selected[layer] = torch.from_numpy(layer_flat.copy()).to(dtype=torch.bool)
        pairs.extend((layer, int(channel)) for channel in np.flatnonzero(layer_flat))
        offset += width
    pair_array = np.asarray(pairs, dtype=np.int64)
    if pair_array.size == 0:
        pair_array = np.zeros((0, 2), dtype=np.int64)
    return pair_array, selected


def ranked(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return indices
    order = np.lexsort((indices, -scores[indices]))
    return indices[order]


def candidate_masks(
    scores: torch.Tensor,
    parent: Mapping[int, torch.Tensor],
    widths: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selection = config["selection"]
    budgets = [int(value) for value in selection["budgets_ascending"]]
    retentions = [float(value) for value in selection["retention_fractions"]]
    parent_flat = flatten_selected(parent, widths)
    parent_budget = int(parent_flat.sum())
    flat_scores = scores.detach().float().cpu().numpy().reshape(-1)
    if flat_scores.size != parent_flat.size:
        raise ValueError(
            f"score size {flat_scores.size} does not match mask size {parent_flat.size}"
        )
    parent_indices = np.flatnonzero(parent_flat)
    external_indices = np.flatnonzero(~parent_flat)
    ranked_parent = ranked(parent_indices, flat_scores)
    ranked_external = ranked(external_indices, flat_scores)
    candidates: list[dict[str, Any]] = []
    seen: set[bytes] = set()

    def add(
        *,
        label: str,
        flat: np.ndarray,
        budget: int,
        requested_retention: float,
        eligible: bool,
        operator: str,
    ) -> None:
        key = np.packbits(flat.astype(np.uint8)).tobytes()
        if key in seen:
            return
        seen.add(key)
        pairs, selected = selected_from_flat(flat, widths)
        overlap = int(np.logical_and(flat, parent_flat).sum())
        candidates.append(
            {
                "label": label,
                "operator": operator,
                "budget": int(budget),
                "requested_retention": float(requested_retention),
                "parent_overlap": overlap,
                "retention_of_candidate": overlap / max(int(budget), 1),
                "retention_of_parent": overlap / max(parent_budget, 1),
                "eligible": bool(eligible),
                "pairs": pairs,
                "selected": selected,
            }
        )

    add(
        label=f"parent_b{parent_budget}",
        flat=parent_flat.copy(),
        budget=parent_budget,
        requested_retention=1.0,
        eligible=True,
        operator="unchanged_parent",
    )
    for budget in [value for value in budgets if value <= parent_budget]:
        for retention in retentions:
            retained = min(parent_budget, budget, int(round(budget * retention)))
            external = budget - retained
            flat = np.zeros_like(parent_flat)
            flat[ranked_parent[:retained]] = True
            flat[ranked_external[:external]] = True
            retention_label = str(retention).replace(".", "p")
            operator = "parent_only_shrink" if external == 0 else "bounded_swap"
            add(
                label=f"b{budget}_retain_{retention_label}",
                flat=flat,
                budget=budget,
                requested_retention=retention,
                eligible=True,
                operator=operator,
            )
        if bool(selection.get("evaluate_global_topk_control", False)):
            flat = np.zeros_like(parent_flat)
            all_indices = np.arange(flat.size, dtype=np.int64)
            flat[ranked(all_indices, flat_scores)[:budget]] = True
            add(
                label=f"b{budget}_global_control",
                flat=flat,
                budget=budget,
                requested_retention=0.0,
                eligible=bool(selection.get("global_topk_control_is_eligible", False)),
                operator="global_topk_control",
            )
    return candidates


def evaluate_candidates(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    scores: torch.Tensor,
    parent_selected: Mapping[int, torch.Tensor],
    parent_screen_masked_correct: int,
    widths: Sequence[int],
    calibration_records: Sequence[Mapping[str, Any]],
    development_records: Sequence[Mapping[str, Any]],
    calibration_dense: Mapping[str, Any],
    development_dense: Mapping[str, Any],
    branch_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = config["selection"]
    minimum_overall = float(selection["minimum_overall_recovery"])
    minimum_per_class = float(selection["minimum_per_carry_class_recovery"])
    same_budget_gain = int(selection["minimum_same_budget_masked_gain"])
    parent_budget = selected_count(parent_selected)
    candidate_dir = branch_dir / "candidates"
    candidate_dir.mkdir()
    receipts: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []

    for candidate in candidate_masks(scores, parent_selected, widths, config):
        label = str(candidate["label"])
        mask_path = candidate_dir / f"{label}.npz"
        np.savez_compressed(
            mask_path,
            mlp_final=candidate["pairs"],
            budget=np.asarray(candidate["budget"], dtype=np.int64),
            parent_overlap=np.asarray(candidate["parent_overlap"], dtype=np.int64),
        )
        calibration_masked = evaluate_and_save(
            model,
            tokenizer,
            calibration_records,
            selected=candidate["selected"],
            args=args,
            output=branch_dir / f"calibration_{label}_predictions.jsonl",
        )
        calibration_pass, calibration_gate = passes_recovery_gate(
            calibration_masked,
            calibration_dense,
            minimum_overall=minimum_overall,
            minimum_per_class=minimum_per_class,
        )
        development_masked = None
        development_gate = None
        development_pass = False
        if calibration_pass:
            development_masked = evaluate_and_save(
                model,
                tokenizer,
                development_records,
                selected=candidate["selected"],
                args=args,
                output=branch_dir / f"development_{label}_predictions.jsonl",
            )
            development_pass, development_gate = passes_recovery_gate(
                development_masked,
                development_dense,
                minimum_overall=minimum_overall,
                minimum_per_class=minimum_per_class,
            )
        masked_gain = (
            int(development_masked["correct"]) - parent_screen_masked_correct
            if development_masked is not None
            else None
        )
        frontier_improvement = bool(
            int(candidate["budget"]) < parent_budget
            or (
                int(candidate["budget"]) == parent_budget
                and masked_gain is not None
                and masked_gain >= same_budget_gain
            )
        )
        passed = bool(
            candidate["eligible"]
            and calibration_pass
            and development_pass
            and frontier_improvement
        )
        receipt = {
            key: value
            for key, value in candidate.items()
            if key not in {"pairs", "selected"}
        }
        receipt.update(
            {
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "calibration_masked": calibration_masked,
                "development_masked": development_masked,
                "calibration_gate": calibration_gate,
                "development_gate": development_gate,
                "masked_correct_gain_vs_parent": masked_gain,
                "frontier_improvement": frontier_improvement,
                "pass": passed,
            }
        )
        receipts.append(receipt)
        if passed:
            passing.append(
                {
                    **receipt,
                    "pairs": candidate["pairs"],
                    "selected": candidate["selected"],
                }
            )
    return receipts, passing


def winner_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    development = candidate["development_masked"]
    calibration = candidate["calibration_masked"]
    return (
        int(candidate["budget"]),
        -int(development["correct"]),
        -int(calibration["correct"]),
        -float(candidate["retention_of_candidate"]),
        int(candidate["branch"]),
        str(candidate["label"]),
    )


def run(args: argparse.Namespace, config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    contract = config["contract"]
    round_config = config["round"]
    selection = config["selection"]
    root_round = int(contract["root_round"])
    target_round = int(contract["target_round"])
    repair_branches = int(round_config["repair_branches"])
    dense_floor = int(selection["minimum_dense_correct_vs_round_zero_screen"])

    splits = build_pair_splits(PAIR_SPLIT_SEED)
    development_records = evaluation_records(
        splits["screen"],
        int(round_config["development_screen_rows"]),
        seed=args.seed,
        purpose=str(selection["expected_root_control"]["screen_builder_purpose"]),
    )
    write_jsonl(ledger.output_dir / "development_screen_records.jsonl", development_records)
    split_receipt = {
        "pair_split_seed": PAIR_SPLIT_SEED,
        "search_pairs": len(splits["search"]),
        "development_screen_pairs": len(splits["screen"]),
        "terminal_holdout_pairs": len(splits["holdout"]),
        "pair_disjoint": True,
        "terminal_holdout_scored_before_tree_frozen": False,
    }
    write_json(ledger.output_dir / "split_receipt.json", split_receipt)
    ledger.finish("freeze_contract", **split_receipt)

    ledger.require_time("load_root_checkpoint", 1_800)
    root_chain = [args.root_adapter]
    root_model = load_model_chain(args, root_chain)
    widths = mlp_widths(root_model)
    root_selected, root_mask_receipt = load_selected_mlp_mask(
        args.root_mask, widths=widths, key="mlp_final"
    )
    baseline_selected, baseline_mask_receipt = load_selected_mlp_mask(
        args.baseline_mask, widths=widths, key=args.baseline_mask_key
    )
    root_budget = int(root_mask_receipt["kept_total"])
    if root_budget != int(contract["root_budget"]):
        raise ValueError(
            f"root mask budget {root_budget} does not match contract {contract['root_budget']}"
        )
    root_dir = ledger.output_dir / "round_01_root"
    root_dir.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        raise RuntimeError(
            f"root checkpoint control mismatch: observed {observed_root}, expected {expected_root}"
        )
    root_recovery, root_per_class = recovery(root_masked, root_dense)
    progress: list[dict[str, Any]] = [
        {
            "round": root_round,
            "checkpoint": "accepted_r1_root",
            "budget": root_budget,
            "dense": root_dense,
            "masked": root_masked,
            "recovery": root_recovery,
            "per_carry_class_recovery": root_per_class,
            "root_adapter_sha256": sha256_file(args.root_adapter / "adapter_model.safetensors"),
            "root_mask_sha256": sha256_file(args.root_mask),
        }
    ]
    write_json(root_dir / "summary.json", progress[0])
    ledger.finish(
        "load_root_checkpoint",
        root_budget=root_budget,
        dense_correct=int(root_dense["correct"]),
        masked_correct=int(root_masked["correct"]),
        recovery=root_recovery,
    )
    del root_model
    torch.cuda.empty_cache()

    current_round = root_round
    current_budget = root_budget
    current_selected = root_selected
    current_chain = root_chain
    current_screen_masked_correct = int(root_masked["correct"])
    termination = "target_round_reached"
    last_attempted_round = root_round
    exhausted_depths: list[int] = []

    for depth in range(root_round + 1, target_round + 1):
        last_attempted_round = depth
        ledger.require_time(f"round_{depth}_tree", 600)
        depth_dir = ledger.output_dir / f"round_{depth:02d}"
        depth_dir.mkdir()
        passing_children: list[dict[str, Any]] = []
        branch_summaries: list[dict[str, Any]] = []

        for branch in range(repair_branches):
            ledger.require_time(f"round_{depth}_branch_{branch}", 420)
            branch_dir = depth_dir / f"branch_{branch:02d}"
            branch_dir.mkdir()
            pseudo_round = depth * 100 + branch
            roles = round_pair_roles(
                splits["search"], seed=args.seed, round_index=pseudo_round
            )
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

            model = load_model_chain(args, current_chain)
            mining_summary = evaluate_and_save(
                model,
                tokenizer,
                mining_records,
                selected=current_selected,
                args=args,
                output=branch_dir / "parent_masked_mining_predictions.jsonl",
            )
            with (branch_dir / "parent_masked_mining_predictions.jsonl").open(
                encoding="utf-8"
            ) as handle:
                mining_predictions = [json.loads(line) for line in handle if line.strip()]
            failures = [
                row for row in mining_predictions if not bool(row["exact_numeric_correct"])
            ]
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

            stage_ledger = BranchLedgerView(ledger, depth, branch)
            model, train_history = train_repair_adapter(
                model,
                tokenizer,
                examples,
                training_args(args, config, pseudo_round),
                stage_ledger,
            )
            expected_steps = int(round_config["training"]["steps"])
            if len(train_history) != expected_steps:
                raise RuntimeError(
                    f"round {depth} branch {branch} trained "
                    f"{len(train_history)}/{expected_steps} steps"
                )
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
                ledger=stage_ledger,
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
            receipts: list[dict[str, Any]] = []
            passing: list[dict[str, Any]] = []
            dense_pass = int(development_dense["correct"]) >= dense_floor
            if dense_pass:
                receipts, passing = evaluate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    scores=scores,
                    parent_selected=current_selected,
                    parent_screen_masked_correct=current_screen_masked_correct,
                    widths=widths,
                    calibration_records=calibration_records,
                    development_records=development_records,
                    calibration_dense=calibration_dense,
                    development_dense=development_dense,
                    branch_dir=branch_dir,
                    args=args,
                    config=config,
                )
            write_json(branch_dir / "candidate_receipts.json", receipts)
            for candidate in passing:
                candidate.update(
                    {
                        "branch": branch,
                        "adapter_path": str(adapter_dir),
                        "adapter_dir": adapter_dir,
                        "development_dense": development_dense,
                        "calibration_dense": calibration_dense,
                    }
                )
                passing_children.append(candidate)
            branch_summary = {
                "round": depth,
                "branch": branch,
                "parent_budget": current_budget,
                "mining": mining_summary,
                "mined_failure_rows": len(failures),
                "training": training_receipt,
                "final_training_loss": train_history[-1],
                "dense_floor": dense_floor,
                "dense_pass": dense_pass,
                "calibration_dense": calibration_dense,
                "development_dense": development_dense,
                "passing_candidates": len(passing),
                "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
            }
            write_json(branch_dir / "summary.json", branch_summary)
            branch_summaries.append(branch_summary)
            ledger.finish(
                f"round_{depth}_branch_{branch}_complete",
                round=depth,
                branch=branch,
                mined_failures=len(failures),
                dense_correct=int(development_dense["correct"]),
                passing_candidates=len(passing),
            )
            del model
            torch.cuda.empty_cache()

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
            write_json(ledger.output_dir / "progress.json", progress)
            ledger.finish(
                f"round_{depth}_incumbent_retained",
                incumbent_round=current_round,
                incumbent_budget=current_budget,
                incumbent_masked_correct=current_screen_masked_correct,
            )
            continue

        winner = min(passing_children, key=winner_key)
        selected_mask_path = depth_dir / "selected_mask.npz"
        np.savez_compressed(
            selected_mask_path,
            mlp_final=winner["pairs"],
            budget=np.asarray(winner["budget"], dtype=np.int64),
            parent_overlap=np.asarray(winner["parent_overlap"], dtype=np.int64),
        )
        parent_budget = current_budget
        current_round = depth
        current_budget = int(winner["budget"])
        current_selected = winner["selected"]
        current_chain = [*current_chain, winner["adapter_dir"]]
        current_screen_masked_correct = int(winner["development_masked"]["correct"])
        accepted_summary = {
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
            "selected_mask_sha256": sha256_file(selected_mask_path),
            "selected_adapter": str(winner["adapter_dir"]),
            "selected_adapter_sha256": sha256_file(
                winner["adapter_dir"] / "adapter_model.safetensors"
            ),
            "passing_children": len(passing_children),
            "branches": branch_summaries,
        }
        write_json(depth_dir / "summary.json", accepted_summary)
        progress.append(accepted_summary)
        write_json(ledger.output_dir / "progress.json", progress)
        ledger.finish(
            f"round_{depth}_accepted",
            round=depth,
            branch=int(winner["branch"]),
            parent_budget=parent_budget,
            selected_budget=current_budget,
            dense_correct=int(winner["development_dense"]["correct"]),
            masked_correct=current_screen_masked_correct,
            recovery=float(winner["development_gate"]["overall"]),
            operator=winner["operator"],
        )

    if exhausted_depths:
        termination = "target_round_reached_after_exhausted_depths"

    ledger.require_time("terminal_holdout", 240)
    holdout_records = evaluation_records(
        splits["holdout"],
        int(contract["terminal_holdout_rows"]),
        seed=args.seed,
        purpose="terminal_holdout",
    )
    write_jsonl(ledger.output_dir / "terminal_holdout_records.jsonl", holdout_records)

    def terminal_condition(
        label: str,
        chain: Sequence[Path],
        selected: Mapping[int, torch.Tensor],
    ) -> dict[str, Any]:
        model = load_model_chain(args, chain)
        dense = evaluate_and_save(
            model,
            tokenizer,
            holdout_records,
            selected=None,
            args=args,
            output=ledger.output_dir / f"terminal_{label}_dense_predictions.jsonl",
        )
        masked = evaluate_and_save(
            model,
            tokenizer,
            holdout_records,
            selected=selected,
            args=args,
            output=ledger.output_dir / f"terminal_{label}_masked_predictions.jsonl",
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
        }

    r0_terminal = terminal_condition("r0", [], baseline_selected)
    r1_terminal = terminal_condition("r1", root_chain, root_selected)
    winner_terminal = terminal_condition("winner", current_chain, current_selected)
    winner_pass, winner_gate = passes_recovery_gate(
        winner_terminal["masked"],
        winner_terminal["dense"],
        minimum_overall=float(selection["minimum_overall_recovery"]),
        minimum_per_class=float(selection["minimum_per_carry_class_recovery"]),
    )
    terminal = {
        "tree_termination": termination,
        "winner_round": current_round,
        "last_attempted_round": last_attempted_round,
        "exhausted_depths": exhausted_depths,
        "rows": len(holdout_records),
        "pairs": len(splits["holdout"]),
        "r0": r0_terminal,
        "r1": r1_terminal,
        "winner": winner_terminal,
        "winner_gate": winner_gate,
        "winner_pass": winner_pass,
        "baseline_mask_receipt": baseline_mask_receipt,
    }
    write_json(ledger.output_dir / "terminal_holdout_summary.json", terminal)
    ledger.finish(
        "terminal_holdout",
        termination=termination,
        winner_round=current_round,
        winner_budget=current_budget,
        winner_dense_correct=int(winner_terminal["dense"]["correct"]),
        winner_masked_correct=int(winner_terminal["masked"]["correct"]),
        winner_recovery=float(winner_terminal["recovery"]),
        winner_pass=winner_pass,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "tree_termination": termination,
        "root_round": root_round,
        "winner_round": current_round,
        "last_attempted_round": last_attempted_round,
        "exhausted_depths": exhausted_depths,
        "target_round": target_round,
        "root_budget": root_budget,
        "winner_budget": current_budget,
        "progress": progress,
        "terminal_holdout": terminal,
        "elapsed_s": ledger.elapsed,
    }
    write_json(ledger.output_dir / "result.json", result)
    ledger.finish(
        "complete",
        termination=termination,
        winner_round=current_round,
        last_attempted_round=last_attempted_round,
        exhausted_depths=exhausted_depths,
        winner_budget=current_budget,
        terminal_pass=winner_pass,
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ledger = RunLedger(args.output_dir, args.deadline_seconds)
    write_json(ledger.output_dir / "config.json", config)
    write_json(
        ledger.output_dir / "args.json",
        {
            **vars(args),
            "config": str(args.config),
            "root_adapter": str(args.root_adapter),
            "root_mask": str(args.root_mask),
            "baseline_mask": str(args.baseline_mask),
            "output_dir": str(args.output_dir),
        },
    )
    try:
        result = run(args, config, ledger)
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_s": ledger.elapsed,
            "traceback": traceback.format_exc(),
        }
        write_json(ledger.output_dir / "failure.json", failure)
        ledger.event("failed", **failure)
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
