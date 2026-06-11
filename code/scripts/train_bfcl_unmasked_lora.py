#!/usr/bin/env python3
"""Train an unmasked BFCL LoRA collimation adapter before attribution.

This runner is for the BFCL pre-attribution-collimation experiment. It adapts
`Qwen/Qwen3-8B` on strict BFCL-style tool-call rows without installing an MLP
keep mask. The merged local model can then be used as the target of a fresh
ReLP attribution run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bfcl_masked_lora import (  # noqa: E402
    ToolMindDataset,
    answer_ce_loss,
    collate,
    kl_loss,
    model_input_device,
    move_batch,
    read_jsonl,
    save_adapter_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train-jsonl", type=Path, default=Path("data/bfcl_strict_10k_mix_len1024/train.jsonl"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--experiment-id", default="bfcl_pre_attr_collimation_v0")
    p.add_argument("--github-issue", default="2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--max-memory", default=None, help="Optional comma list, e.g. 0:78GiB,1:78GiB")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--policy-kl-beta", type=float, default=1.0)
    p.add_argument("--ce-beta", type=float, default=0.2)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-merged", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-project", default="prism-bfcl")
    p.add_argument("--wandb-group", default="issue-2")
    p.add_argument("--wandb-job-type", default="train")
    p.add_argument("--wandb-name")
    p.add_argument("--wandb-tags", default="bfcl,pre-attribution,collimation,qwen3-8b")
    return p.parse_args()


def parse_max_memory(spec: str | None) -> dict[int, str] | None:
    if not spec:
        return None
    out: dict[int, str] = {}
    for item in spec.split(","):
        key, value = item.split(":", 1)
        out[int(key.strip())] = value.strip()
    return out


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on run environment
        raise RuntimeError("W&B logging requested but wandb is not installed") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        name=args.wandb_name or args.experiment_id,
        mode=args.wandb_mode,
        tags=tags,
        config=config,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    config = {
        "experiment_id": args.experiment_id,
        "github_issue": args.github_issue,
        "method": "unmasked_bfcl_lora_collimation_before_attribution",
        "args": jsonable_args(args),
        "public_upload_policy": "adapter_and_receipts_only_no_base_or_merged_full_weights",
    }
    config_path = args.out_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    wandb_run = maybe_init_wandb(args, config)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(args.train_jsonl)
    random.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] raw_rows={len(rows)} train={args.train_jsonl}", flush=True)
    dataset = ToolMindDataset(rows, tokenizer, args.max_seq_length)
    if len(dataset) == 0:
        raise RuntimeError("no train rows survived tokenization/length filtering")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda xs: collate(xs, tokenizer.pad_token_id),
    )

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "attn_implementation": "eager"}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        max_memory = parse_max_memory(args.max_memory)
        if max_memory:
            load_kwargs["max_memory"] = max_memory

    print(
        f"[model] {args.model} device={args.device} device_map={args.device_map} dtype={args.dtype}",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not args.device_map:
        base = base.to(args.device)
    base.config.use_cache = False
    input_device = model_input_device(base)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_rslora=args.use_rslora,
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    model.train()

    total_batches = math.ceil(len(loader) * args.epochs)
    if args.max_steps is not None:
        total_batches = min(total_batches, args.max_steps * args.grad_accum)
    total_steps = math.ceil(total_batches / args.grad_accum)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    summary = {
        "config": config,
        "n_rows": len(dataset),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "logs": [],
        "checkpoints": [],
    }
    summary_path = args.out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    start = time.time()
    global_step = 0
    seen_batches = 0
    running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)

    try:
        while global_step < total_steps:
            for raw_batch in loader:
                if global_step >= total_steps:
                    break
                batch = move_batch(raw_batch, str(input_device))
                with torch.no_grad(), model.disable_adapter():
                    teacher_logits = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    ).logits

                student_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                policy_kl = kl_loss(
                    student_logits,
                    teacher_logits,
                    batch["kl_logit_mask"],
                    temperature=args.kl_temperature,
                )
                ce = answer_ce_loss(student_logits, batch["labels"])
                loss = args.policy_kl_beta * policy_kl + args.ce_beta * ce
                (loss / args.grad_accum).backward()

                seen_batches += 1
                running["loss"] += float(loss.detach().cpu())
                running["policy_kl"] += float(policy_kl.detach().cpu())
                running["ce"] += float(ce.detach().cpu())
                running["n"] += 1

                if seen_batches % args.grad_accum != 0:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == 1 or global_step % args.eval_every == 0 or global_step == total_steps:
                    denom = max(running["n"], 1)
                    row = {
                        "step": global_step,
                        "loss": running["loss"] / denom,
                        "policy_kl": running["policy_kl"] / denom,
                        "ce": running["ce"] / denom,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": time.time() - start,
                    }
                    summary["logs"].append(row)
                    summary_path.write_text(json.dumps(summary, indent=2))
                    print(json.dumps(row), flush=True)
                    if wandb_run is not None:
                        wandb_run.log({f"train/{k}": v for k, v in row.items()}, step=global_step)
                    running = {"loss": 0.0, "policy_kl": 0.0, "ce": 0.0, "n": 0}

                if args.save_every and global_step % args.save_every == 0:
                    checkpoint_dir = save_adapter_checkpoint(model, tokenizer, args.out_dir, global_step)
                    summary["checkpoints"].append({"step": global_step, "adapter_dir": str(checkpoint_dir)})
                    summary_path.write_text(json.dumps(summary, indent=2))
                    print(f"[checkpoint] step={global_step} adapter={checkpoint_dir}", flush=True)

        model.eval()
        summary["elapsed_s"] = time.time() - start
        adapter_dir = args.out_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        summary["adapter_dir"] = str(adapter_dir)
        print(f"[done] adapter={adapter_dir}", flush=True)

        if args.save_merged:
            merged_dir = args.out_dir / "merged"
            print(f"[merge] saving local attribution target {merged_dir}", flush=True)
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            summary["merged_dir"] = str(merged_dir)
            summary["merged_upload_policy"] = "local_attribution_target_only_do_not_upload_base_weights"

        summary_path.write_text(json.dumps(summary, indent=2))
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "n_rows": len(dataset),
                    "total_steps": total_steps,
                    "adapter_dir": summary.get("adapter_dir"),
                    "merged_dir": summary.get("merged_dir"),
                }
            )
            try:
                import wandb

                artifact = wandb.Artifact(f"{args.experiment_id}-train-receipts", type="train_receipts")
                artifact.add_file(str(config_path))
                artifact.add_file(str(summary_path))
                wandb_run.log_artifact(artifact)
            except Exception as exc:  # pragma: no cover - W&B runtime surface
                print(f"[wandb] artifact logging skipped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
