#!/usr/bin/env python3
"""Run a bounded zero-isolated arithmetic MACE search in one loaded model.

The search deliberately separates historical donor-patch rankings from the
new acceptance test. Historical masks and layer rankings only propose
candidates. Every branch decision is made by unguided standalone generation
with the complement set to zero and no donor activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from evaluate_arithmetic_standalone import (
    generate_predictions,
    logical_zero_isolation,
    mlp_widths,
    model_receipt,
    sha256_file,
    summarize_predictions,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "prism_arithmetic_mace_search_v1"
Pair = tuple[int, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no records in {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"record IDs are not unique in {path}")
    return rows


def load_mask_pairs(
    path: Path,
    *,
    key: str,
    widths: Sequence[int],
) -> set[Pair]:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"{key!r} missing from {path}")
        raw = np.asarray(payload[key])
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{key!r} must have shape [N, 2]")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{key!r} must contain integer pairs")
    pairs = {(int(layer), int(channel)) for layer, channel in raw.tolist()}
    if len(pairs) != len(raw):
        raise ValueError(f"{path}:{key} contains duplicate pairs")
    validate_pairs(pairs, widths=widths)
    return pairs


def validate_pairs(pairs: set[Pair], *, widths: Sequence[int]) -> None:
    if not pairs:
        raise ValueError("candidate selects zero channels")
    per_layer = [0] * len(widths)
    for layer, channel in pairs:
        if not 0 <= layer < len(widths):
            raise ValueError(f"layer {layer} outside candidate model")
        if not 0 <= channel < int(widths[layer]):
            raise ValueError(
                f"channel {channel} outside layer {layer} width {widths[layer]}"
            )
        per_layer[layer] += 1
    empty = [index for index, count in enumerate(per_layer) if count == 0]
    if empty:
        raise ValueError(
            "physical candidates must retain at least one channel per layer; "
            f"empty layers={empty}"
        )


def full_pairs(widths: Sequence[int]) -> set[Pair]:
    return {
        (layer, channel)
        for layer, width in enumerate(widths)
        for channel in range(int(width))
    }


def selected_masks(
    pairs: set[Pair],
    *,
    widths: Sequence[int],
) -> dict[int, torch.Tensor]:
    validate_pairs(pairs, widths=widths)
    selected = {
        layer: torch.zeros(int(width), dtype=torch.bool)
        for layer, width in enumerate(widths)
    }
    for layer, channel in pairs:
        selected[layer][channel] = True
    return selected


def canonical_pairs_sha256(pairs: set[Pair]) -> str:
    encoded = json.dumps(
        sorted([list(pair) for pair in pairs]),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_mask(path: Path, pairs: set[Pair]) -> None:
    ordered = np.asarray(sorted(pairs), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mlp_final=ordered)


def parse_mask_spec(spec: str) -> tuple[str, Path, str]:
    """Parse NAME=PATH[:KEY], using mlp_final when KEY is omitted."""

    if "=" not in spec:
        raise ValueError("mask spec must be NAME=PATH[:KEY]")
    name, raw = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("mask spec name is empty")
    if ":" in raw:
        raw_path, key = raw.rsplit(":", 1)
    else:
        raw_path, key = raw, "mlp_final"
    if not raw_path or not key:
        raise ValueError("mask spec needs a path and key")
    return name, Path(raw_path), key


def ranked_layers(path: Path | None, *, num_layers: int) -> list[int]:
    """Read historical layer proposals, then append every missing layer."""

    ordered: list[int] = []
    if path is not None:
        payload = json.loads(path.read_text())
        ranking = payload.get("rankings", {}).get("top_by_delta_per_1k", [])
        for row in ranking:
            if not str(row.get("kind", "")).startswith("add_"):
                continue
            layer = int(row["layer"])
            if 0 <= layer < num_layers and layer not in ordered:
                ordered.append(layer)
    ordered.extend(layer for layer in range(num_layers) if layer not in ordered)
    return ordered


def compare_with_dense(
    dense: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dense_by_id = {str(row["id"]): row for row in dense}
    candidate_by_id = {str(row["id"]): row for row in candidate}
    if dense_by_id.keys() != candidate_by_id.keys():
        raise ValueError("dense and candidate prediction IDs differ")
    dense_correct = {
        record_id
        for record_id, row in dense_by_id.items()
        if bool(row["exact_numeric_correct"])
    }
    if not dense_correct:
        raise ValueError("dense parent has zero correct records")
    matched_correct = {
        record_id
        for record_id in dense_correct
        if bool(candidate_by_id[record_id]["exact_numeric_correct"])
    }
    candidate_correct = {
        record_id
        for record_id, row in candidate_by_id.items()
        if bool(row["exact_numeric_correct"])
    }
    disagreements = [
        {
            "id": record_id,
            "dense_correct": bool(dense_by_id[record_id]["exact_numeric_correct"]),
            "candidate_correct": bool(
                candidate_by_id[record_id]["exact_numeric_correct"]
            ),
            "dense_prediction_text": dense_by_id[record_id]["prediction_text"],
            "candidate_prediction_text": candidate_by_id[record_id][
                "prediction_text"
            ],
        }
        for record_id in sorted(dense_by_id)
        if (
            dense_by_id[record_id]["prediction_text"]
            != candidate_by_id[record_id]["prediction_text"]
            or bool(dense_by_id[record_id]["exact_numeric_correct"])
            != bool(candidate_by_id[record_id]["exact_numeric_correct"])
        )
    ]
    return {
        "dense_correct": len(dense_correct),
        "candidate_correct": len(candidate_correct),
        "matched_dense_correct": len(matched_correct),
        "n": len(dense_by_id),
        "dense_accuracy": len(dense_correct) / len(dense_by_id),
        "candidate_accuracy": len(candidate_correct) / len(candidate_by_id),
        "matched_dense_recovery": len(matched_correct) / len(dense_correct),
        "candidate_only_correct": len(candidate_correct - dense_correct),
        "dense_only_correct": len(dense_correct - candidate_correct),
        "prediction_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def candidate_for_prefix(
    lower: set[Pair],
    upper: set[Pair],
    order: Sequence[int],
    prefix: int,
) -> set[Pair]:
    if not lower <= upper:
        raise ValueError("ranked-prefix search requires lower to be a subset")
    chosen_layers = set(order[:prefix])
    return lower | {
        pair for pair in upper - lower if pair[0] in chosen_layers
    }


def software_receipt() -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception as error:  # pragma: no cover
        versions["transformers"] = f"unavailable: {error}"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument(
        "--seed-mask",
        action="append",
        required=True,
        help="repeat NAME=PATH[:KEY]",
    )
    parser.add_argument(
        "--ceiling-mask",
        required=True,
        help="NAME=PATH[:KEY], normally the rel_0.001 mask",
    )
    parser.add_argument(
        "--safety-mask",
        help="NAME=PATH[:KEY], normally the positive mask",
    )
    parser.add_argument("--ranking-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--recovery-floor", type=float, default=0.90)
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args()

    if not 0.0 < args.recovery_floor <= 1.0:
        parser.error("--recovery-floor must be in (0, 1]")
    if args.max_rounds < 3:
        parser.error("--max-rounds must be at least 3")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch size and max new tokens must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=args.tokenizer_revision or args.model_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation=args.attention_implementation,
    ).to(args.device)
    model.eval()
    widths = mlp_widths(model)
    records = read_jsonl(args.records_jsonl)

    seed_specs = [parse_mask_spec(spec) for spec in args.seed_mask]
    ceiling_spec = parse_mask_spec(args.ceiling_mask)
    safety_spec = (
        parse_mask_spec(args.safety_mask) if args.safety_mask else None
    )
    source_specs = [*seed_specs, ceiling_spec]
    if safety_spec is not None:
        source_specs.append(safety_spec)
    source_receipts = {}
    masks = {}
    for name, path, key in source_specs:
        if name in masks:
            raise ValueError(f"duplicate mask name {name!r}")
        pairs = load_mask_pairs(path, key=key, widths=widths)
        masks[name] = pairs
        source_receipts[name] = {
            "path": str(path),
            "key": key,
            "file_sha256": sha256_file(path),
            "selection_sha256": canonical_pairs_sha256(pairs),
            "kept": len(pairs),
        }

    write_jsonl(args.output_dir / "records.jsonl", records)
    dense_started = time.perf_counter()
    dense_predictions = generate_predictions(
        model,
        tokenizer,
        records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    dense_seconds = time.perf_counter() - dense_started
    write_jsonl(args.output_dir / "dense_predictions.jsonl", dense_predictions)
    dense_summary = summarize_predictions(dense_predictions)

    rounds: list[dict[str, Any]] = []
    result_by_hash: dict[str, dict[str, Any]] = {}

    def evaluate(
        *,
        label: str,
        pairs: set[Pair],
        parent: str | None,
        operation: str,
    ) -> dict[str, Any] | None:
        selection_hash = canonical_pairs_sha256(pairs)
        if selection_hash in result_by_hash:
            return result_by_hash[selection_hash]
        if len(rounds) >= args.max_rounds:
            return None
        validate_pairs(pairs, widths=widths)
        round_number = len(rounds) + 1
        round_dir = args.output_dir / "rounds" / f"{round_number:02d}_{label}"
        round_dir.mkdir(parents=True)
        mask_path = round_dir / "mask.npz"
        write_mask(mask_path, pairs)
        started = time.perf_counter()
        with logical_zero_isolation(
            model,
            selected_masks(pairs, widths=widths),
        ):
            predictions = generate_predictions(
                model,
                tokenizer,
                records,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        seconds = time.perf_counter() - started
        comparison = compare_with_dense(dense_predictions, predictions)
        disagreements = comparison.pop("disagreements")
        passing = (
            comparison["matched_dense_recovery"] >= args.recovery_floor
        )
        receipt = {
            "round": round_number,
            "label": label,
            "parent": parent,
            "operation": operation,
            "selection_sha256": selection_hash,
            "mask_sha256": sha256_file(mask_path),
            "kept": len(pairs),
            "available": sum(widths),
            "kept_fraction": len(pairs) / sum(widths),
            "seconds": seconds,
            "summary": summarize_predictions(predictions),
            "comparison": comparison,
            "recovery_floor": args.recovery_floor,
            "passes_mace": passing,
            "mask_path": str(mask_path),
            "predictions_path": str(round_dir / "predictions.jsonl"),
        }
        write_jsonl(round_dir / "predictions.jsonl", predictions)
        write_jsonl(round_dir / "disagreements.jsonl", disagreements)
        write_json(round_dir / "receipt.json", receipt)
        rounds.append(receipt)
        result_by_hash[selection_hash] = receipt
        write_json(args.output_dir / "search_ledger.json", rounds)
        return receipt

    for name, _, _ in seed_specs:
        evaluate(
            label=name,
            pairs=masks[name],
            parent=None,
            operation="audited_seed",
        )

    passing = [row for row in rounds if row["passes_mace"]]
    upper_name = ceiling_spec[0]
    if passing:
        best_seed = min(passing, key=lambda row: row["kept"])
        upper_name = best_seed["label"]
    if not passing:
        evaluate(
            label=upper_name,
            pairs=masks[upper_name],
            parent=None,
            operation="audited_ceiling",
        )
        passing = [row for row in rounds if row["passes_mace"]]

    if not passing and safety_spec is not None:
        upper_name = safety_spec[0]
        evaluate(
            label=upper_name,
            pairs=masks[upper_name],
            parent=None,
            operation="audited_safety_ceiling",
        )
        passing = [row for row in rounds if row["passes_mace"]]

    if not passing:
        upper_name = "full_dense_mlp"
        masks[upper_name] = full_pairs(widths)
        evaluate(
            label=upper_name,
            pairs=masks[upper_name],
            parent=None,
            operation="full_mask_safety_control",
        )
        passing = [row for row in rounds if row["passes_mace"]]

    if not passing:
        raise RuntimeError(
            "even the full MLP mask failed matched-dense recovery; "
            "the evaluator is internally inconsistent"
        )

    upper_pairs = masks[upper_name]

    failing_subsets = []
    for row in rounds:
        if row["passes_mace"]:
            continue
        pairs = next(
            (
                value
                for value in masks.values()
                if canonical_pairs_sha256(value) == row["selection_sha256"]
            ),
            None,
        )
        if pairs is not None and pairs < upper_pairs:
            failing_subsets.append((row, pairs))

    if failing_subsets and len(rounds) < args.max_rounds:
        lower_row, lower_pairs = max(
            failing_subsets,
            key=lambda item: (
                item[0]["comparison"]["matched_dense_recovery"],
                item[0]["kept"],
            ),
        )
        order = ranked_layers(args.ranking_json, num_layers=len(widths))
        active_layers = [
            layer
            for layer in order
            if any(pair[0] == layer for pair in upper_pairs - lower_pairs)
        ]
        low = 0
        high = len(active_layers)
        passing_pairs = upper_pairs
        passing_label = upper_name
        while high - low > 1 and len(rounds) < args.max_rounds:
            middle = (low + high) // 2
            candidate = candidate_for_prefix(
                lower_pairs,
                upper_pairs,
                active_layers,
                middle,
            )
            row = evaluate(
                label=f"ranked_prefix_{middle:02d}",
                pairs=candidate,
                parent=lower_row["label"],
                operation=(
                    f"add upper-minus-lower channels in first {middle} "
                    "historically ranked layers"
                ),
            )
            if row is not None and row["passes_mace"]:
                high = middle
                passing_pairs = candidate
                passing_label = row["label"]
            else:
                low = middle

        # Greedily remove whole added layer shells from the smallest passing
        # prefix. Each removal is accepted only by an actual standalone score.
        added_by_layer = {
            layer: {
                pair
                for pair in passing_pairs - lower_pairs
                if pair[0] == layer
            }
            for layer in active_layers[:high]
        }
        for layer, shell in sorted(
            added_by_layer.items(),
            key=lambda item: (len(item[1]), item[0]),
        ):
            if not shell or len(rounds) >= args.max_rounds:
                continue
            trial = passing_pairs - shell
            if any(
                not any(pair[0] == layer_index for pair in trial)
                for layer_index in range(len(widths))
            ):
                continue
            row = evaluate(
                label=f"drop_shell_layer_{layer:02d}",
                pairs=trial,
                parent=passing_label,
                operation=f"drop accepted upper shell at layer {layer}",
            )
            if row is not None and row["passes_mace"]:
                passing_pairs = trial
                passing_label = row["label"]

    passing = [row for row in rounds if row["passes_mace"]]
    winner = min(
        passing,
        key=lambda row: (
            row["kept"],
            -row["comparison"]["matched_dense_recovery"],
            row["round"],
        ),
    )
    winner_source = Path(winner["mask_path"])
    winner_dir = args.output_dir / "winner"
    winner_dir.mkdir()
    shutil.copy2(winner_source, winner_dir / "mask.npz")
    shutil.copy2(
        Path(winner["predictions_path"]),
        winner_dir / "predictions.jsonl",
    )
    winner_receipt = {
        **winner,
        "selection_rule": (
            "fewest channels among actually evaluated candidates meeting "
            "the frozen matched-dense recovery floor"
        ),
        "bounded_minimality": True,
        "global_minimality_claimed": False,
        "tested_candidates": len(rounds),
    }
    write_json(winner_dir / "receipt.json", winner_receipt)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "isolation": "zero_unselected_down_proj_inputs",
            "donor_activations": False,
            "generation": "unguided_greedy_fixed_global_cap",
            "max_new_tokens": args.max_new_tokens,
            "recovery_floor": args.recovery_floor,
            "max_rounds": args.max_rounds,
            "branching_note": (
                "historical donor-patch rankings propose layer order only; "
                "all acceptance decisions use standalone generation"
            ),
        },
        "inputs": {
            "records": {
                "path": str(args.records_jsonl),
                "sha256": sha256_file(args.records_jsonl),
                "rows": len(records),
            },
            "masks": source_receipts,
            "ranking_json": (
                {
                    "path": str(args.ranking_json),
                    "sha256": sha256_file(args.ranking_json),
                }
                if args.ranking_json is not None
                else None
            ),
        },
        "model": model_receipt(model, tokenizer, args.model),
        "runtime": {
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "batch_size": args.batch_size,
            "dense_seconds": dense_seconds,
        },
        "dense_summary": dense_summary,
        "rounds": rounds,
        "winner": winner_receipt,
        "software": software_receipt(),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    checksum_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(args.output_dir)}\n"
            for path in checksum_paths
        )
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
