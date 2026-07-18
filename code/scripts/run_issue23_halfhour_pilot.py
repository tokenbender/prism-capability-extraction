#!/usr/bin/env python3
"""Run the Issue #23 fixed-budget arithmetic repair-SFT pilot.

The pilot keeps the experiment contract explicit: pair-disjoint search/screen/
holdout splits, unmasked repair training, fresh post-training ReLP attribution,
and zero-isolated evaluation at the same MLP-channel budget as the R0 control.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from train_lora_2digit_kl import (  # noqa: E402
    AdditionDataset,
    AdditionExample,
    answer_ce_loss,
    carry_signature_label,
    collate_rows,
    kl_loss,
    move_batch,
)
from scripts.evaluate_arithmetic_standalone import (  # noqa: E402
    carry_fields,
    generate_predictions,
    load_selected_mlp_mask,
    logical_zero_isolation,
    mlp_widths,
    render_prompt,
    sha256_file,
    summarize_predictions,
)
from src.circuit_tracing.relp import ReLPAttributor  # noqa: E402

SCHEMA_VERSION = "prism_arithmetic_issue23_halfhour_pilot_v1"
PAIR_SPLIT_SEED = 23_071_826
SEARCH_PAIRS = 4_100
SCREEN_PAIRS = 2_000
HOLDOUT_PAIRS = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--seed-mask", type=Path, required=True)
    parser.add_argument("--seed-mask-key", default="mlp_rel_0.001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--fixed-budget", type=int, default=202_923)
    parser.add_argument("--train-base-rows", type=int, default=2_048)
    parser.add_argument("--repair-rows", type=int, default=512)
    parser.add_argument("--mining-rows", type=int, default=512)
    parser.add_argument("--screen-rows", type=int, default=500)
    parser.add_argument("--attribution-rows", type=int, default=512)
    parser.add_argument("--attribution-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline-seconds", type=float, default=1_500.0)
    return parser.parse_args()


class RunLedger:
    def __init__(self, output_dir: Path, deadline_seconds: float):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline_seconds = deadline_seconds
        self.completed: list[str] = []
        self.events_path = output_dir / "events.jsonl"

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return self.deadline_seconds - self.elapsed

    def event(self, stage: str, **values: Any) -> None:
        payload = {
            "stage": stage,
            "elapsed_s": self.elapsed,
            "remaining_s": self.remaining,
            **values,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        (self.output_dir / "run_state.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": stage,
                    "elapsed_s": self.elapsed,
                    "remaining_s": self.remaining,
                    "completed_stages": self.completed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, sort_keys=True), flush=True)

    def finish(self, stage: str, **values: Any) -> None:
        self.completed.append(stage)
        self.event(stage, status="complete", **values)

    def require_time(self, stage: str, reserve_seconds: float) -> None:
        if self.remaining < reserve_seconds:
            self.event(
                stage,
                status="stopped_before_stage",
                required_remaining_s=reserve_seconds,
            )
            raise TimeoutError(
                f"refusing to start {stage}: {self.remaining:.1f}s remain, "
                f"need {reserve_seconds:.1f}s"
            )


def stable_key(seed: int, *parts: object) -> str:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def pair_metadata(a: int, b: int) -> dict[str, Any]:
    answer = a + b
    return {
        "pair_id": f"{a:02d}-{b:02d}",
        "a": a,
        "b": b,
        "answer": answer,
        "answer_text": str(answer),
        "result_length": len(str(answer)),
        "carry_signature": carry_signature_label(a, b),
        **carry_fields(a, b),
    }


def build_pair_splits(seed: int) -> dict[str, list[dict[str, Any]]]:
    rows = [pair_metadata(a, b) for a in range(10, 100) for b in range(10, 100)]
    rows.sort(key=lambda row: stable_key(seed, row["pair_id"]))
    splits = {
        "search": rows[:SEARCH_PAIRS],
        "screen": rows[SEARCH_PAIRS : SEARCH_PAIRS + SCREEN_PAIRS],
        "holdout": rows[SEARCH_PAIRS + SCREEN_PAIRS :],
    }
    assert len(splits["holdout"]) == HOLDOUT_PAIRS
    pair_sets = {name: {row["pair_id"] for row in split} for name, split in splits.items()}
    assert not (pair_sets["search"] & pair_sets["screen"])
    assert not (pair_sets["search"] & pair_sets["holdout"])
    assert not (pair_sets["screen"] & pair_sets["holdout"])
    return splits


def balanced_pairs(
    pairs: Sequence[Mapping[str, Any]], count: int, *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(pairs):
        raise ValueError(f"invalid balanced pair count {count} for {len(pairs)} rows")
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in pairs:
        row = dict(raw)
        strata[(str(row["carry_signature"]), int(row["result_length"]))].append(row)
    for key, rows in strata.items():
        rows.sort(key=lambda row: stable_key(seed, purpose, key, row["pair_id"]))
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in strata}
    keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                offsets[key] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("balanced selection exhausted before requested count")
    return selected


def evaluation_records(
    pairs: Sequence[Mapping[str, Any]], row_count: int, *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    if row_count % 2:
        raise ValueError("evaluation row count must be even for compact/spaced balance")
    selected_pairs = balanced_pairs(pairs, row_count // 2, seed=seed, purpose=purpose)
    records: list[dict[str, Any]] = []
    for pair in selected_pairs:
        for prompt_format in ("compact", "spaced"):
            records.append(
                {
                    **dict(pair),
                    "id": f"{purpose}-{pair['pair_id']}-{prompt_format}",
                    "prompt": render_prompt(int(pair["a"]), int(pair["b"]), prompt_format),
                    "prompt_format": prompt_format,
                }
            )
    records.sort(key=lambda row: stable_key(seed, purpose, row["id"]))
    return records


def training_example(pair: Mapping[str, Any], variant: str) -> AdditionExample:
    a = int(pair["a"])
    b = int(pair["b"])
    canonical = f"{a} + {b} ="
    templates = {
        "canonical": canonical,
        "canonical_spaced": canonical + " ",
        "compact": f"{a}+{b}=",
        "compact_spaced": f"{a}+{b}= ",
        "words": f"{a} plus {b} equals ",
        "question": f"What is {a} + {b}? ",
        "sum": f"Sum: {a} + {b} = ",
    }
    return AdditionExample(
        a=a,
        b=b,
        answer=str(a + b),
        prompt=canonical,
        train_prompt=templates[variant],
        variant=variant,
        objective="addition",
        carry_signature=str(pair["carry_signature"]),
    )


def build_training_examples(
    search_pairs: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    base_rows: int,
    repair_rows: int,
    seed: int,
) -> tuple[list[AdditionExample], dict[str, Any]]:
    if base_rows % 2:
        raise ValueError("train-base-rows must be even")
    seed_pairs = balanced_pairs(
        search_pairs,
        base_rows // 2,
        seed=seed,
        purpose="train_seed",
    )
    examples = [
        training_example(pair, variant)
        for pair in seed_pairs
        for variant in ("canonical", "question")
    ]
    unique_failures: dict[str, dict[str, Any]] = {}
    for failure in failures:
        unique_failures[str(failure["pair_id"])] = dict(failure)
    failure_pairs = sorted(
        unique_failures.values(),
        key=lambda row: stable_key(seed, "repair", row["pair_id"]),
    )
    repair_examples: list[AdditionExample] = []
    repair_variants = ("canonical_spaced", "compact", "words", "sum")
    if failure_pairs:
        index = 0
        while len(repair_examples) < repair_rows:
            pair = failure_pairs[index % len(failure_pairs)]
            variant = repair_variants[(index // len(failure_pairs)) % len(repair_variants)]
            repair_examples.append(training_example(pair, variant))
            index += 1
    examples.extend(repair_examples)
    random.Random(seed).shuffle(examples)
    receipt = {
        "base_rows": len(examples) - len(repair_examples),
        "base_unique_pairs": len(seed_pairs),
        "repair_rows": len(repair_examples),
        "repair_unique_pairs": len(failure_pairs),
        "total_rows": len(examples),
        "search_only": True,
        "variants": sorted({example.variant for example in examples}),
    }
    return examples, receipt


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[int, torch.Tensor] | None,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    isolation = (
        logical_zero_isolation(model, selected)
        if selected is not None
        else contextlib.nullcontext()
    )
    with isolation:
        predictions = generate_predictions(
            model,
            tokenizer,
            records,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    return predictions, summarize_predictions(predictions)


def train_repair_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[AdditionExample],
    args: argparse.Namespace,
    ledger: RunLedger,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, lora)
    dataset = AdditionDataset(examples, tokenizer, kl_on="prompt", append_eos=False)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda rows: collate_rows(rows, int(tokenizer.pad_token_id)),
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = max(1, int(args.train_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.train_steps,
    )
    history: list[dict[str, float]] = []
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.train_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, args.device)
        with torch.no_grad(), model.disable_adapter():
            base_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        adapted_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        ce = answer_ce_loss(adapted_logits, batch["labels"])
        kl = kl_loss(
            adapted_logits,
            base_logits,
            batch["kl_logit_mask"],
            temperature=1.0,
        )
        loss = ce + args.kl_beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": float(step),
            "loss": float(loss.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        if step == 1 or step % 10 == 0 or step == args.train_steps:
            ledger.event("repair_sft", status="running", **row)
        if ledger.remaining < 420:
            ledger.event("repair_sft", status="stopped_at_step", **row)
            break
    return model, history


def encode_attribution_batch(
    tokenizer: Any, records: Sequence[Mapping[str, Any]], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for record in records:
        prompt = str(record.get("generation_prompt", record["prompt"]))
        answer = str(record["answer_text"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"answer added no tokens for {record['id']}")
        token_rows.append([int(value) for value in full_ids])
        label_rows.append([-100] * len(prompt_ids) + [int(value) for value in full_ids[len(prompt_ids) :]])
    width = max(len(row) for row in token_rows)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(token_rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(token_rows), width), -100, dtype=torch.long)
    for index, (tokens, gold) in enumerate(zip(token_rows, label_rows)):
        input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        labels[index, : len(gold)] = torch.tensor(gold, dtype=torch.long)
    return input_ids.to(device), labels.to(device)


def attribute_records(
    model: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: str,
    ledger: RunLedger,
) -> torch.Tensor:
    model.eval()
    tokenizer.padding_side = "right"
    attributor = ReLPAttributor(model, tokenizer, device=device)
    scores = torch.zeros(
        (int(model.config.num_hidden_layers), int(model.config.intermediate_size)),
        dtype=torch.float32,
    )
    completed = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        input_ids, labels = encode_attribution_batch(tokenizer, batch_records, device)

        def metric_fn(logits: torch.Tensor, gold: torch.Tensor = labels) -> torch.Tensor:
            shifted_gold = gold[:, 1:]
            selected_positions = shifted_gold.ne(-100)
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            safe_gold = shifted_gold.masked_fill(~selected_positions, 0)
            gathered = logp.gather(2, safe_gold.unsqueeze(-1)).squeeze(-1)
            return gathered[selected_positions].sum()

        attributes = attributor.attribute(input_ids, metric_fn)
        for layer_index, tensor in attributes.items():
            scores[layer_index] += tensor.abs().sum(dim=(0, 1)).detach().cpu()
        completed += len(batch_records)
        ledger.event(
            "fresh_relp",
            status="running",
            examples_completed=completed,
            examples_total=len(records),
        )
        if ledger.remaining < 240:
            raise TimeoutError("deadline reserve reached during fresh ReLP attribution")
    scores /= max(completed, 1)
    return scores


def topk_mask(scores: torch.Tensor, budget: int) -> tuple[np.ndarray, dict[int, torch.Tensor]]:
    if scores.ndim != 2:
        raise ValueError(f"scores must be rank two, got {tuple(scores.shape)}")
    if budget <= 0 or budget > scores.numel():
        raise ValueError(f"fixed budget {budget} outside [1, {scores.numel()}]")
    top = torch.topk(scores.flatten(), k=budget, sorted=True)
    width = int(scores.shape[1])
    layers = torch.div(top.indices, width, rounding_mode="floor")
    channels = top.indices.remainder(width)
    pairs = torch.stack((layers, channels), dim=1).cpu().numpy().astype(np.int64)
    selected = {
        layer: torch.zeros(width, dtype=torch.bool)
        for layer in range(int(scores.shape[0]))
    }
    for layer, channel in pairs.tolist():
        selected[int(layer)][int(channel)] = True
    return pairs, selected


def accuracy(summary: Mapping[str, Any]) -> float:
    return float(summary["accuracy"])


def run(args: argparse.Namespace, ledger: RunLedger) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    splits = build_pair_splits(PAIR_SPLIT_SEED)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": PAIR_SPLIT_SEED,
        "unit": "ordered_operand_pair",
        "search": splits["search"],
        "screen": splits["screen"],
        "holdout": splits["holdout"],
        "pair_disjoint": True,
    }
    write_json(ledger.output_dir / "split_manifest.json", split_manifest)
    screen_records = evaluation_records(
        splits["screen"], args.screen_rows, seed=args.seed, purpose="sealed_screen"
    )
    mining_records = evaluation_records(
        splits["search"], args.mining_rows, seed=args.seed, purpose="failure_mining"
    )
    attribution_records = evaluation_records(
        splits["search"], args.attribution_rows, seed=args.seed + 1, purpose="fresh_relp"
    )
    write_jsonl(ledger.output_dir / "screen_records.jsonl", screen_records)
    write_jsonl(ledger.output_dir / "mining_records.jsonl", mining_records)
    write_jsonl(ledger.output_dir / "attribution_records.jsonl", attribution_records)
    ledger.finish(
        "freeze_data",
        search_pairs=len(splits["search"]),
        screen_pairs=len(splits["screen"]),
        holdout_pairs=len(splits["holdout"]),
    )

    ledger.require_time("load_model", 1_200)
    dtype = getattr(torch, args.dtype)
    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    widths = mlp_widths(model)
    seed_selected, seed_mask_receipt = load_selected_mlp_mask(
        args.seed_mask,
        widths=widths,
        key=args.seed_mask_key,
    )
    if int(seed_mask_receipt["kept_total"]) != args.fixed_budget:
        raise ValueError(
            f"seed mask keeps {seed_mask_receipt['kept_total']} channels; "
            f"fixed budget is {args.fixed_budget}"
        )
    ledger.finish(
        "load_model",
        model=args.model,
        tokenizer=tokenizer_source,
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        seed_mask_sha256=sha256_file(args.seed_mask),
        fixed_budget=args.fixed_budget,
    )

    ledger.require_time("r0_controls", 900)
    r0_dense_predictions, r0_dense_summary = evaluate(
        model,
        tokenizer,
        screen_records,
        selected=None,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    r0_masked_predictions, r0_masked_summary = evaluate(
        model,
        tokenizer,
        screen_records,
        selected=seed_selected,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    mining_predictions, mining_summary = evaluate(
        model,
        tokenizer,
        mining_records,
        selected=seed_selected,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(ledger.output_dir / "r0_dense_screen_predictions.jsonl", r0_dense_predictions)
    write_jsonl(ledger.output_dir / "r0_masked_screen_predictions.jsonl", r0_masked_predictions)
    write_jsonl(ledger.output_dir / "r0_masked_mining_predictions.jsonl", mining_predictions)
    failures = [row for row in mining_predictions if not bool(row["exact_numeric_correct"])]
    write_jsonl(ledger.output_dir / "mined_failures.jsonl", failures)
    ledger.finish(
        "r0_controls",
        dense_screen_accuracy=accuracy(r0_dense_summary),
        masked_screen_accuracy=accuracy(r0_masked_summary),
        mining_accuracy=accuracy(mining_summary),
        mined_failures=len(failures),
    )

    examples, training_receipt = build_training_examples(
        splits["search"],
        failures,
        base_rows=args.train_base_rows,
        repair_rows=args.repair_rows,
        seed=args.seed,
    )
    write_jsonl(
        ledger.output_dir / "training_examples.jsonl",
        ({
            "a": example.a,
            "b": example.b,
            "answer": example.answer,
            "train_prompt": example.train_prompt,
            "variant": example.variant,
            "carry_signature": example.carry_signature,
        } for example in examples),
    )
    write_json(ledger.output_dir / "training_receipt.json", training_receipt)

    ledger.require_time("repair_sft", 720)
    model, train_history = train_repair_adapter(model, tokenizer, examples, args, ledger)
    adapter_dir = ledger.output_dir / "repair_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    write_json(ledger.output_dir / "train_history.json", train_history)
    if len(train_history) < args.train_steps:
        raise TimeoutError(
            f"repair SFT stopped at {len(train_history)}/{args.train_steps} steps"
        )
    ledger.finish(
        "repair_sft",
        steps=len(train_history),
        final_loss=train_history[-1]["loss"],
        final_ce=train_history[-1]["ce"],
        final_kl=train_history[-1]["kl"],
    )

    ledger.require_time("fresh_relp", 420)
    model = model.merge_and_unload()
    model.enable_input_require_grads()
    model.eval()
    scores = attribute_records(
        model,
        tokenizer,
        attribution_records,
        batch_size=args.attribution_batch_size,
        device=args.device,
        ledger=ledger,
    )
    pairs, repaired_selected = topk_mask(scores, args.fixed_budget)
    repaired_mask_path = ledger.output_dir / "repaired_fixed_budget_mask.npz"
    np.savez_compressed(
        repaired_mask_path,
        mlp_final=pairs,
        mlp_scores=scores.numpy(),
        fixed_budget=np.asarray(args.fixed_budget, dtype=np.int64),
    )
    ledger.finish(
        "fresh_relp",
        attributed_examples=len(attribution_records),
        fixed_budget=args.fixed_budget,
        mask_sha256=sha256_file(repaired_mask_path),
    )

    ledger.require_time("r1_gate", 150)
    r1_dense_predictions, r1_dense_summary = evaluate(
        model,
        tokenizer,
        screen_records,
        selected=None,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    r1_masked_predictions, r1_masked_summary = evaluate(
        model,
        tokenizer,
        screen_records,
        selected=repaired_selected,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(ledger.output_dir / "r1_dense_screen_predictions.jsonl", r1_dense_predictions)
    write_jsonl(ledger.output_dir / "r1_masked_screen_predictions.jsonl", r1_masked_predictions)

    masked_gain = int(r1_masked_summary["correct"]) - int(r0_masked_summary["correct"])
    dense_loss = int(r0_dense_summary["correct"]) - int(r1_dense_summary["correct"])
    gate = {
        "fixed_budget": args.fixed_budget,
        "screen_rows": len(screen_records),
        "r0_dense": r0_dense_summary,
        "r0_masked": r0_masked_summary,
        "r1_dense": r1_dense_summary,
        "r1_masked": r1_masked_summary,
        "masked_correct_gain": masked_gain,
        "dense_correct_loss": dense_loss,
        "thresholds": {
            "minimum_masked_correct_gain": 10,
            "maximum_dense_correct_loss": 3,
        },
        "promote": masked_gain >= 10 and dense_loss <= 3,
        "holdout_touched": False,
    }
    write_json(ledger.output_dir / "gate.json", gate)
    ledger.finish(
        "r1_gate",
        r0_dense_accuracy=accuracy(r0_dense_summary),
        r0_masked_accuracy=accuracy(r0_masked_summary),
        r1_dense_accuracy=accuracy(r1_dense_summary),
        r1_masked_accuracy=accuracy(r1_masked_summary),
        masked_correct_gain=masked_gain,
        dense_correct_loss=dense_loss,
        promote=gate["promote"],
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "experiment_decision": "promote" if gate["promote"] else "stop",
        "gate": gate,
        "training": training_receipt,
        "seed_mask": seed_mask_receipt,
        "elapsed_s": ledger.elapsed,
    }
    write_json(ledger.output_dir / "result.json", result)
    ledger.finish("complete", decision=result["experiment_decision"])
    return result


def main() -> None:
    args = parse_args()
    ledger = RunLedger(args.output_dir, args.deadline_seconds)
    write_json(ledger.output_dir / "args.json", vars(args) | {
        "seed_mask": str(args.seed_mask),
        "output_dir": str(args.output_dir),
    })
    try:
        result = run(args, ledger)
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
