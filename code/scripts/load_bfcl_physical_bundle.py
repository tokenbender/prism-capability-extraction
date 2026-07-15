#!/usr/bin/env python3
"""Cold-load and evaluate a jagged Qwen BFCL physical-substrate bundle.

The bundle keeps the complete attention/residual/norm scaffold while replacing
each Qwen MLP with the physically retained intermediate channels recorded in
``substrate_metadata.json``.  Stock ``AutoModelForCausalLM.from_pretrained``
cannot reconstruct these non-uniform layer widths, so this loader builds the
correct module shapes before assigning the serialized tensors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bfcl_direct_qwen3 import (
    messages_for_generation,
    normalized_prediction_ok,
    parse_tool_calls,
    prediction_ok,
    read_records,
    write_jsonl,
)


SUPPORTED_FORMAT = "qwen_physical_mlp_substrate_v1"


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder-layer list without depending on one wrapper layout."""

    current = model
    for _ in range(8):
        layers = getattr(current, "layers", None)
        if layers is not None:
            return layers
        for name in ("model", "base_model"):
            child = getattr(current, name, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            break
    raise AttributeError("could not locate the decoder layers")


class PhysicalQwenMLP(nn.Module):
    """Qwen gated MLP whose intermediate width may differ in every layer."""

    def __init__(self, original: nn.Module, intermediate_size: int):
        super().__init__()
        if intermediate_size <= 0:
            raise ValueError("physical bundle widths must be positive")

        self.hidden_size = int(original.down_proj.out_features)
        self.intermediate_size = int(intermediate_size)
        self.act_fn = original.act_fn
        device = original.gate_proj.weight.device
        dtype = original.gate_proj.weight.dtype

        self.gate_proj = nn.Linear(
            original.gate_proj.in_features,
            self.intermediate_size,
            bias=original.gate_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            original.up_proj.in_features,
            self.intermediate_size,
            bias=original.up_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            original.down_proj.out_features,
            bias=original.down_proj.bias is not None,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


def read_bundle_metadata(bundle: Path) -> dict[str, Any]:
    metadata_path = bundle / "substrate_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing bundle metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != SUPPORTED_FORMAT:
        raise ValueError(
            f"unsupported bundle format {metadata.get('format')!r}; "
            f"expected {SUPPORTED_FORMAT!r}"
        )
    return metadata


def install_physical_mlps(model: nn.Module, metadata: dict[str, Any]) -> list[int]:
    layers = decoder_layers(model)
    kept = metadata.get("isolation", {}).get("kept_per_layer", {})
    expected_keys = {str(index) for index in range(len(layers))}
    if set(kept) != expected_keys:
        raise ValueError("metadata kept_per_layer keys do not cover the decoder exactly")

    widths = [int(kept[str(index)]) for index in range(len(layers))]
    dense_width = int(model.config.intermediate_size)
    if any(width <= 0 or width > dense_width for width in widths):
        raise ValueError(f"invalid physical widths for dense width {dense_width}: {widths}")
    recorded_total = int(metadata.get("isolation", {}).get("kept_total", -1))
    if recorded_total != sum(widths):
        raise ValueError(f"kept_total mismatch: {recorded_total} != {sum(widths)}")

    for layer, width in zip(layers, widths):
        layer.mlp = PhysicalQwenMLP(layer.mlp, width)
    return widths


def load_physical_bundle(
    bundle: Path,
    *,
    device: str = "cuda:0",
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Reconstruct a bundle without ever allocating the original dense MLPs."""

    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    bundle = bundle.resolve()
    metadata = read_bundle_metadata(bundle)
    generation_config_path = bundle / "generation_config.json"
    if not generation_config_path.is_file():
        raise FileNotFoundError(
            "physical bundle is missing generation_config.json; the frozen Qwen3 "
            "contract requires its multi-EOS and padding settings"
        )
    checkpoint_files = sorted(bundle.glob("*.safetensors"))
    if len(checkpoint_files) != 1:
        raise ValueError(
            f"expected exactly one safetensors checkpoint, found {len(checkpoint_files)}"
        )

    config = AutoConfig.from_pretrained(bundle, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError(f"unsupported model_type for physical Qwen loader: {config.model_type}")
    dtype_name = str(metadata.get("dtype"))
    if dtype_name != "bfloat16":
        raise ValueError(f"unsupported bundle dtype: {dtype_name!r}")
    expected_dtype = torch.bfloat16
    if getattr(config, "dtype", None) != expected_dtype:
        raise ValueError(
            f"config dtype {getattr(config, 'dtype', None)} does not match {expected_dtype}"
        )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
        )
        widths = install_physical_mlps(model, metadata)

    state = load_file(checkpoint_files[0], device="cpu")
    dtype_mismatches = {
        name: str(tensor.dtype)
        for name, tensor in state.items()
        if torch.is_floating_point(tensor) and tensor.dtype != expected_dtype
    }
    if dtype_mismatches:
        raise ValueError(f"checkpoint contains non-BF16 floating tensors: {dtype_mismatches}")
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint state mismatch: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("one or more parameters remained on the meta device")
    loaded_dtypes = {parameter.dtype for parameter in model.parameters()}
    if loaded_dtypes != {expected_dtype}:
        raise RuntimeError(f"loaded parameter dtype mismatch: {loaded_dtypes}")

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    model = model.to(requested)
    model.generation_config = GenerationConfig.from_pretrained(
        bundle,
        local_files_only=True,
    )
    model.eval()

    # Qwen3-8B's frozen Issue #12 tokenizer used the legacy pre-tokenizer.  An
    # explicit False preserves that byte-level contract and suppresses the
    # Transformers migration warning; True would change experiment inputs.
    tokenizer = AutoTokenizer.from_pretrained(
        bundle,
        local_files_only=True,
        fix_mistral_regex=False,
    )
    if getattr(tokenizer, "fix_mistral_regex", None) is not False:
        raise RuntimeError("tokenizer did not preserve fix_mistral_regex=False")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    receipt = {
        "format": metadata["format"],
        "bundle": str(bundle),
        "device": str(requested),
        "attention_implementation": "eager",
        "layers": len(widths),
        "kept_total": sum(widths),
        "kept_per_layer": widths,
        "checkpoint": checkpoint_files[0].name,
        "parameter_dtype": str(expected_dtype),
        "generation_config_restored": True,
        "tokenizer_fix_mistral_regex": False,
    }
    return model, tokenizer, receipt


def reference_parity(
    generated: list[dict[str, Any]],
    reference_path: Path,
) -> dict[str, Any]:
    """Compare the complete generation/scoring boundary, not only its total."""

    reference = read_records(reference_path)
    generated_ids = [str(row["id"]) for row in generated]
    reference_ids = [str(row["id"]) for row in reference]
    if generated_ids != reference_ids:
        raise ValueError("generated and reference prediction IDs/order do not match")

    fields = (
        "prediction_text",
        "prediction_calls",
        "raw_correct",
        "normalized_correct",
    )
    field_diffs = {field: 0 for field in fields}
    differing_ids: list[str] = []
    for actual, expected in zip(generated, reference):
        differs = False
        for field in fields:
            if actual.get(field) != expected.get(field):
                field_diffs[field] += 1
                differs = True
        if differs:
            differing_ids.append(str(actual["id"]))
    return {
        "status": "pass" if not differing_ids else "fail",
        "reference_predictions": str(reference_path),
        "examples": len(generated),
        "field_diff_counts": field_diffs,
        "differing_ids": differing_ids,
    }


def evaluate(args: argparse.Namespace) -> None:
    rows = read_records(args.pairs)
    end = len(rows) if args.end is None else args.end
    rows = rows[args.start : end]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("evaluation selection is empty")

    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer, load_receipt = load_physical_bundle(
        args.bundle,
        device=args.device,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    input_device = model.get_input_embeddings().weight.device

    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        encoded_items = [
            tokenizer.apply_chat_template(
                messages_for_generation(
                    row,
                    bfcl_canonicalization_prompt=args.bfcl_canonicalization_prompt,
                ),
                tools=row.get("tools") or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=args.enable_thinking,
            )
            for row in batch_rows
        ]
        encoded = tokenizer.pad(
            encoded_items,
            padding=True,
            return_tensors="pt",
        ).to(input_device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        for row, sequence in zip(batch_rows, generated):
            text = tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
            calls = parse_tool_calls(text)
            raw_correct = prediction_ok(calls, row)
            normalized_correct = normalized_prediction_ok(calls, row)
            output_rows.append(
                {
                    "id": row["id"],
                    "category": row.get("category"),
                    "prediction_text": text,
                    "prediction_calls": calls,
                    "target": row.get("target"),
                    "reference_calls": row.get("reference_calls"),
                    "correct": normalized_correct if args.normalized else raw_correct,
                    "raw_correct": raw_correct,
                    "normalized_correct": normalized_correct,
                }
            )
        print(f"evaluated {len(output_rows)}/{len(rows)}", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - started
    write_jsonl(args.output, output_rows)
    total = len(output_rows)
    raw_correct = sum(bool(row["raw_correct"]) for row in output_rows)
    normalized_correct = sum(bool(row["normalized_correct"]) for row in output_rows)
    parity = (
        reference_parity(output_rows, args.reference_predictions)
        if args.reference_predictions is not None
        else None
    )
    summary = {
        "status": "complete" if parity is None or parity["status"] == "pass" else "fail",
        "examples": total,
        "raw_exact_correct": raw_correct,
        "raw_exact_accuracy": raw_correct / total,
        "normalized_exact_correct": normalized_correct,
        "normalized_exact_accuracy": normalized_correct / total,
        "reported_metric": "normalized_exact" if args.normalized else "raw_exact",
        "load_seconds": load_seconds,
        "evaluation_seconds": eval_seconds,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "bfcl_canonicalization_prompt": args.bfcl_canonicalization_prompt,
        "pairs": str(args.pairs),
        "output": str(args.output),
        "load_receipt": load_receipt,
        "reference_parity": parity,
    }
    if args.device.startswith("cuda"):
        summary["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        summary["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if parity is not None and parity["status"] != "pass":
        raise RuntimeError(f"reference prediction parity failed: {parity}")


def load_check(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model, _, receipt = load_physical_bundle(args.bundle, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.synchronize()
    receipt.update(
        {
            "status": "pass",
            "load_seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    print(json.dumps(receipt, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("load-check", help="strictly reconstruct the bundle")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--device", default="cuda:0")
    check.set_defaults(func=load_check)

    evaluate_parser = subparsers.add_parser("eval", help="run the frozen direct BFCL scorer")
    evaluate_parser.add_argument("--bundle", type=Path, required=True)
    evaluate_parser.add_argument("--pairs", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--reference-predictions", type=Path)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=512)
    evaluate_parser.add_argument("--start", type=int, default=0)
    evaluate_parser.add_argument("--end", type=int)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--enable-thinking", action="store_true")
    evaluate_parser.add_argument("--bfcl-canonicalization-prompt", action="store_true")
    evaluate_parser.add_argument("--normalized", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
