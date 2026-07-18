#!/usr/bin/env python3
"""Run eight alternating arithmetic repair-SFT and mask-selection rounds."""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.run_issue23_halfhour_pilot import (  # noqa: E402
    PAIR_SPLIT_SEED,
    RunLedger,
    accuracy,
    attribute_records,
    build_pair_splits,
    build_training_examples,
    evaluate,
    evaluation_records,
    load_selected_mlp_mask,
    mlp_widths,
    sha256_file,
    stable_key,
    topk_mask,
    train_repair_adapter,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = "prism_arithmetic_issue23_eight_round_repair_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seed-mask", type=Path, required=True)
    parser.add_argument("--seed-mask-key", default="mlp_rel_0.001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=float, default=1_700.0)
    return parser.parse_args()


class StageLedgerView:
    """Prefix helper-stage events with their owning round."""

    def __init__(self, ledger: RunLedger, round_index: int):
        self.ledger = ledger
        self.round_index = round_index

    @property
    def remaining(self) -> float:
        return self.ledger.remaining

    def event(self, stage: str, **values: Any) -> None:
        self.ledger.event(
            f"round_{self.round_index}_{stage}",
            round=self.round_index,
            **values,
        )


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "prism_arithmetic_issue23_eight_round_config_v1":
        raise ValueError(f"unexpected config schema in {path}")
    if int(config["contract"]["rounds"]) != 8:
        raise ValueError("Issue #23 iterative contract requires exactly eight rounds")
    budgets = [int(value) for value in config["selection"]["budgets_ascending"]]
    if budgets != sorted(set(budgets)):
        raise ValueError("selection budgets must be unique and ascending")
    return config


def training_args(
    args: argparse.Namespace, config: Mapping[str, Any], round_index: int
) -> SimpleNamespace:
    training = config["round"]["training"]
    return SimpleNamespace(
        lora_r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        train_batch_size=int(training["batch_size"]),
        train_steps=int(training["steps"]),
        warmup_ratio=0.05,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=1.0,
        kl_beta=float(training["kl_beta"]),
        seed=int(args.seed + round_index * 10_003),
        device=args.device,
    )


def recovery(
    masked: Mapping[str, Any], dense: Mapping[str, Any]
) -> tuple[float, dict[str, float]]:
    dense_correct = int(dense["correct"])
    overall = int(masked["correct"]) / max(dense_correct, 1)
    per_class: dict[str, float] = {}
    dense_classes = dense.get("by_carry_class", {})
    masked_classes = masked.get("by_carry_class", {})
    for carry_class in sorted(dense_classes):
        denominator = int(dense_classes[carry_class]["correct"])
        numerator = int(masked_classes[carry_class]["correct"])
        per_class[carry_class] = numerator / max(denominator, 1)
    return overall, per_class


def passes_recovery_gate(
    masked: Mapping[str, Any],
    dense: Mapping[str, Any],
    *,
    minimum_overall: float,
    minimum_per_class: float,
) -> tuple[bool, dict[str, Any]]:
    overall, per_class = recovery(masked, dense)
    receipt = {
        "overall": overall,
        "per_carry_class": per_class,
        "minimum_overall": minimum_overall,
        "minimum_per_carry_class": minimum_per_class,
        "pass": overall >= minimum_overall
        and all(value >= minimum_per_class for value in per_class.values()),
    }
    return bool(receipt["pass"]), receipt


def round_pair_roles(
    search_pairs: Sequence[Mapping[str, Any]], *, seed: int, round_index: int
) -> dict[str, list[dict[str, Any]]]:
    ordered = [dict(row) for row in search_pairs]
    ordered.sort(
        key=lambda row: stable_key(seed, "round_roles", round_index, row["pair_id"])
    )
    roles = {
        "mining": ordered[:256],
        "attribution": ordered[256:512],
        "calibration": ordered[512:768],
        "training_pool": ordered[768:],
    }
    role_sets = {
        name: {str(row["pair_id"]) for row in rows}
        for name, rows in roles.items()
    }
    names = list(role_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if role_sets[left] & role_sets[right]:
                raise AssertionError(f"round role overlap: {left} vs {right}")
    return roles


def evaluate_and_save(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[int, torch.Tensor] | None,
    args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    predictions, summary = evaluate(
        model,
        tokenizer,
        records,
        selected=selected,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(output, predictions)
    return summary


def run(args: argparse.Namespace, config: dict[str, Any], ledger: RunLedger) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    round_config = config["round"]
    selection_config = config["selection"]
    budgets = [int(value) for value in selection_config["budgets_ascending"]]
    rounds = int(config["contract"]["rounds"])
    minimum_overall = float(selection_config["minimum_overall_recovery"])
    minimum_per_class = float(
        selection_config["minimum_per_carry_class_recovery"]
    )
    dense_floor = int(
        selection_config["minimum_dense_correct_vs_round_zero_screen"]
    )

    splits = build_pair_splits(PAIR_SPLIT_SEED)
    development_records = evaluation_records(
        splits["screen"],
        int(round_config["development_screen_rows"]),
        seed=args.seed,
        purpose="development_screen",
    )
    write_jsonl(ledger.output_dir / "development_screen_records.jsonl", development_records)
    split_receipt = {
        "pair_split_seed": PAIR_SPLIT_SEED,
        "search_pairs": len(splits["search"]),
        "development_screen_pairs": len(splits["screen"]),
        "terminal_holdout_pairs": len(splits["holdout"]),
        "pair_disjoint": True,
        "terminal_holdout_scored_before_rounds_complete": False,
    }
    write_json(ledger.output_dir / "split_receipt.json", split_receipt)
    ledger.finish("freeze_contract", **split_receipt)

    ledger.require_time("load_model", 1_400)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    widths = mlp_widths(model)
    initial_selected, initial_mask_receipt = load_selected_mlp_mask(
        args.seed_mask,
        widths=widths,
        key=args.seed_mask_key,
    )
    initial_budget = int(initial_mask_receipt["kept_total"])
    if initial_budget != max(budgets):
        raise ValueError(
            f"initial mask budget {initial_budget} does not match largest ladder "
            f"budget {max(budgets)}"
        )
    current_selected = initial_selected
    current_budget = initial_budget
    ledger.finish(
        "load_model",
        model=args.model,
        tokenizer=args.tokenizer,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        initial_budget=initial_budget,
        initial_mask_sha256=sha256_file(args.seed_mask),
    )

    round_zero_dir = ledger.output_dir / "round_00"
    round_zero_dir.mkdir(parents=True)
    round_zero_dense = evaluate_and_save(
        model,
        tokenizer,
        development_records,
        selected=None,
        args=args,
        output=round_zero_dir / "development_dense_predictions.jsonl",
    )
    round_zero_masked = evaluate_and_save(
        model,
        tokenizer,
        development_records,
        selected=current_selected,
        args=args,
        output=round_zero_dir / "development_masked_predictions.jsonl",
    )
    round_zero_recovery, round_zero_per_class = recovery(
        round_zero_masked, round_zero_dense
    )
    round_zero = {
        "round": 0,
        "budget": current_budget,
        "dense": round_zero_dense,
        "masked": round_zero_masked,
        "recovery": round_zero_recovery,
        "per_carry_class_recovery": round_zero_per_class,
    }
    write_json(round_zero_dir / "summary.json", round_zero)
    ledger.finish(
        "round_0_control",
        budget=current_budget,
        dense_correct=int(round_zero_dense["correct"]),
        masked_correct=int(round_zero_masked["correct"]),
        recovery=round_zero_recovery,
    )

    progress: list[dict[str, Any]] = [round_zero]
    previous_screen_masked_correct = int(round_zero_masked["correct"])

    for round_index in range(1, rounds + 1):
        ledger.require_time(f"round_{round_index}", 600)
        round_dir = ledger.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True)
        roles = round_pair_roles(
            splits["search"], seed=args.seed, round_index=round_index
        )
        mining_records = evaluation_records(
            roles["mining"],
            int(round_config["mining_rows"]),
            seed=args.seed + round_index,
            purpose=f"round_{round_index}_mining",
        )
        attribution_records = evaluation_records(
            roles["attribution"],
            int(round_config["attribution_rows"]),
            seed=args.seed + round_index,
            purpose=f"round_{round_index}_attribution",
        )
        calibration_records = evaluation_records(
            roles["calibration"],
            int(round_config["calibration_rows"]),
            seed=args.seed + round_index,
            purpose=f"round_{round_index}_calibration",
        )
        write_jsonl(round_dir / "mining_records.jsonl", mining_records)
        write_jsonl(round_dir / "attribution_records.jsonl", attribution_records)
        write_jsonl(round_dir / "calibration_records.jsonl", calibration_records)

        mining_summary = evaluate_and_save(
            model,
            tokenizer,
            mining_records,
            selected=current_selected,
            args=args,
            output=round_dir / "parent_masked_mining_predictions.jsonl",
        )
        with (round_dir / "parent_masked_mining_predictions.jsonl").open(
            encoding="utf-8"
        ) as handle:
            mining_predictions = [json.loads(line) for line in handle if line.strip()]
        failures = [
            row for row in mining_predictions if not bool(row["exact_numeric_correct"])
        ]
        write_jsonl(round_dir / "mined_failures.jsonl", failures)

        examples, training_receipt = build_training_examples(
            roles["training_pool"],
            failures,
            base_rows=int(round_config["training_base_rows"]),
            repair_rows=int(round_config["failure_repair_rows"]),
            seed=args.seed + round_index * 10_003,
        )
        write_jsonl(
            round_dir / "training_examples.jsonl",
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
        write_json(round_dir / "training_receipt.json", training_receipt)

        stage_ledger = StageLedgerView(ledger, round_index)
        model, train_history = train_repair_adapter(
            model,
            tokenizer,
            examples,
            training_args(args, config, round_index),
            stage_ledger,
        )
        write_json(round_dir / "train_history.json", train_history)
        expected_steps = int(round_config["training"]["steps"])
        if len(train_history) != expected_steps:
            raise RuntimeError(
                f"round {round_index} trained {len(train_history)}/{expected_steps} steps"
            )
        adapter_dir = round_dir / "adapter"
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
        np.savez_compressed(round_dir / "attribution_scores.npz", mlp_scores=scores.numpy())

        calibration_dense = evaluate_and_save(
            model,
            tokenizer,
            calibration_records,
            selected=None,
            args=args,
            output=round_dir / "calibration_dense_predictions.jsonl",
        )
        development_dense = evaluate_and_save(
            model,
            tokenizer,
            development_records,
            selected=None,
            args=args,
            output=round_dir / "development_dense_predictions.jsonl",
        )
        if int(development_dense["correct"]) < dense_floor:
            raise RuntimeError(
                f"round {round_index} dense screen correct "
                f"{development_dense['correct']} below floor {dense_floor}"
            )

        candidate_receipts: list[dict[str, Any]] = []
        candidate_state: dict[int, tuple[np.ndarray, dict[int, torch.Tensor]]] = {}
        for budget in [value for value in budgets if value <= current_budget]:
            pairs, selected = topk_mask(scores, budget)
            candidate_state[budget] = (pairs, selected)
            calibration_masked = evaluate_and_save(
                model,
                tokenizer,
                calibration_records,
                selected=selected,
                args=args,
                output=round_dir / f"calibration_masked_b{budget}_predictions.jsonl",
            )
            development_masked = evaluate_and_save(
                model,
                tokenizer,
                development_records,
                selected=selected,
                args=args,
                output=round_dir / f"development_masked_b{budget}_predictions.jsonl",
            )
            calibration_pass, calibration_gate = passes_recovery_gate(
                calibration_masked,
                calibration_dense,
                minimum_overall=minimum_overall,
                minimum_per_class=minimum_per_class,
            )
            development_pass, development_gate = passes_recovery_gate(
                development_masked,
                development_dense,
                minimum_overall=minimum_overall,
                minimum_per_class=minimum_per_class,
            )
            candidate_receipts.append(
                {
                    "budget": budget,
                    "calibration_masked": calibration_masked,
                    "development_masked": development_masked,
                    "calibration_gate": calibration_gate,
                    "development_gate": development_gate,
                    "pass": calibration_pass and development_pass,
                }
            )

        passing = [row for row in candidate_receipts if bool(row["pass"])]
        if not passing:
            write_json(round_dir / "candidate_receipts.json", candidate_receipts)
            raise RuntimeError(
                f"round {round_index} has no passing mask at or below parent "
                f"budget {current_budget}"
            )
        selected_receipt = min(passing, key=lambda row: int(row["budget"]))
        selected_budget = int(selected_receipt["budget"])
        selected_pairs, selected_mask = candidate_state[selected_budget]
        selected_mask_path = round_dir / "selected_mask.npz"
        np.savez_compressed(
            selected_mask_path,
            mlp_final=selected_pairs,
            mlp_scores=scores.numpy(),
            budget=np.asarray(selected_budget, dtype=np.int64),
        )
        write_json(round_dir / "candidate_receipts.json", candidate_receipts)
        current_selected = selected_mask
        parent_budget = current_budget
        current_budget = selected_budget
        selected_screen_masked = selected_receipt["development_masked"]
        selected_screen_correct = int(selected_screen_masked["correct"])
        screen_gain = selected_screen_correct - previous_screen_masked_correct
        previous_screen_masked_correct = selected_screen_correct

        round_summary = {
            "round": round_index,
            "parent_budget": parent_budget,
            "selected_budget": selected_budget,
            "budget_fraction": selected_budget / sum(widths),
            "mining": mining_summary,
            "mined_failure_rows": len(failures),
            "training": training_receipt,
            "final_training_loss": train_history[-1],
            "calibration_dense": calibration_dense,
            "development_dense": development_dense,
            "selected_calibration_masked": selected_receipt["calibration_masked"],
            "selected_development_masked": selected_screen_masked,
            "calibration_gate": selected_receipt["calibration_gate"],
            "development_gate": selected_receipt["development_gate"],
            "screen_masked_correct_gain_vs_previous_round": screen_gain,
            "selected_mask_sha256": sha256_file(selected_mask_path),
        }
        write_json(round_dir / "summary.json", round_summary)
        progress.append(round_summary)
        write_json(ledger.output_dir / "progress.json", progress)
        ledger.finish(
            f"round_{round_index}_complete",
            round=round_index,
            parent_budget=parent_budget,
            selected_budget=selected_budget,
            dense_correct=int(development_dense["correct"]),
            masked_correct=selected_screen_correct,
            recovery=float(selected_receipt["development_gate"]["overall"]),
            screen_gain=screen_gain,
        )

    ledger.require_time("terminal_holdout", 240)
    holdout_records = evaluation_records(
        splits["holdout"],
        len(splits["holdout"]) * 2,
        seed=args.seed,
        purpose="terminal_holdout",
    )
    write_jsonl(ledger.output_dir / "terminal_holdout_records.jsonl", holdout_records)
    final_holdout_dense = evaluate_and_save(
        model,
        tokenizer,
        holdout_records,
        selected=None,
        args=args,
        output=ledger.output_dir / "terminal_r8_dense_predictions.jsonl",
    )
    final_holdout_masked = evaluate_and_save(
        model,
        tokenizer,
        holdout_records,
        selected=current_selected,
        args=args,
        output=ledger.output_dir / "terminal_r8_masked_predictions.jsonl",
    )
    final_holdout_pass, final_holdout_gate = passes_recovery_gate(
        final_holdout_masked,
        final_holdout_dense,
        minimum_overall=minimum_overall,
        minimum_per_class=minimum_per_class,
    )

    r0_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    r0_model.eval()
    r0_holdout_dense = evaluate_and_save(
        r0_model,
        tokenizer,
        holdout_records,
        selected=None,
        args=args,
        output=ledger.output_dir / "terminal_r0_dense_predictions.jsonl",
    )
    r0_holdout_masked = evaluate_and_save(
        r0_model,
        tokenizer,
        holdout_records,
        selected=initial_selected,
        args=args,
        output=ledger.output_dir / "terminal_r0_masked_predictions.jsonl",
    )
    r0_holdout_recovery, r0_holdout_per_class = recovery(
        r0_holdout_masked, r0_holdout_dense
    )
    del r0_model
    torch.cuda.empty_cache()

    terminal = {
        "rows": len(holdout_records),
        "pairs": len(splits["holdout"]),
        "r0_budget": initial_budget,
        "r0_dense": r0_holdout_dense,
        "r0_masked": r0_holdout_masked,
        "r0_recovery": r0_holdout_recovery,
        "r0_per_carry_class_recovery": r0_holdout_per_class,
        "r8_budget": current_budget,
        "r8_dense": final_holdout_dense,
        "r8_masked": final_holdout_masked,
        "r8_gate": final_holdout_gate,
        "pass": final_holdout_pass,
    }
    write_json(ledger.output_dir / "terminal_holdout_summary.json", terminal)
    ledger.finish(
        "terminal_holdout",
        rows=len(holdout_records),
        r0_budget=initial_budget,
        r0_dense_correct=int(r0_holdout_dense["correct"]),
        r0_masked_correct=int(r0_holdout_masked["correct"]),
        r0_recovery=r0_holdout_recovery,
        r8_budget=current_budget,
        r8_dense_correct=int(final_holdout_dense["correct"]),
        r8_masked_correct=int(final_holdout_masked["correct"]),
        r8_recovery=float(final_holdout_gate["overall"]),
        pass_gate=final_holdout_pass,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "rounds_completed": rounds,
        "initial_budget": initial_budget,
        "final_budget": current_budget,
        "progress": progress,
        "terminal_holdout": terminal,
        "elapsed_s": ledger.elapsed,
    }
    write_json(ledger.output_dir / "result.json", result)
    ledger.finish(
        "complete",
        rounds_completed=rounds,
        final_budget=current_budget,
        terminal_pass=final_holdout_pass,
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
            "seed_mask": str(args.seed_mask),
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
