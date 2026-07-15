#!/usr/bin/env python3
"""Assemble BFCL physical shards and audit them against a frozen prediction file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_rows(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    total = len(rows)
    normalized = sum(bool(row.get("normalized_correct")) for row in rows)
    raw = sum(bool(row.get("raw_correct")) for row in rows)
    return {
        "examples": total,
        "raw_correct": raw,
        "raw_accuracy": raw / total if total else 0.0,
        "normalized_correct": normalized,
        "normalized_accuracy": normalized / total if total else 0.0,
    }


def grouped_scores(
    ordered_ids: list[str],
    physical: dict[str, dict[str, Any]],
    frozen: dict[str, dict[str, Any]],
    group_for_id: dict[str, str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for eval_id in ordered_ids:
        groups[group_for_id[eval_id]].append(eval_id)
    out = []
    for group in sorted(groups):
        ids = groups[group]
        physical_score = score_rows([physical[eval_id] for eval_id in ids])
        frozen_score = score_rows([frozen[eval_id] for eval_id in ids])
        out.append(
            {
                "group": group,
                "examples": len(ids),
                "physical_normalized_correct": physical_score["normalized_correct"],
                "physical_normalized_accuracy": physical_score["normalized_accuracy"],
                "frozen_normalized_correct": frozen_score["normalized_correct"],
                "frozen_normalized_accuracy": frozen_score["normalized_accuracy"],
                "normalized_correct_delta": (
                    physical_score["normalized_correct"] - frozen_score["normalized_correct"]
                ),
                "physical_raw_correct": physical_score["raw_correct"],
                "frozen_raw_correct": frozen_score["raw_correct"],
                "raw_correct_delta": physical_score["raw_correct"] - frozen_score["raw_correct"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--physical", type=Path, action="append", required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    ordered_ids = [str(row["id"]) for row in pairs]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("duplicate IDs in pairs file")

    physical_rows: list[dict[str, Any]] = []
    for path in args.physical:
        physical_rows.extend(read_jsonl(path))
    frozen_rows = read_jsonl(args.frozen)
    physical = {str(row["id"]): row for row in physical_rows}
    frozen = {str(row["id"]): row for row in frozen_rows}
    expected = set(ordered_ids)
    if set(physical) != expected:
        raise ValueError(
            f"physical ID mismatch: missing={sorted(expected - set(physical))[:5]} "
            f"extra={sorted(set(physical) - expected)[:5]}"
        )
    if set(frozen) != expected:
        raise ValueError("frozen prediction IDs do not match pairs IDs")

    ordered_physical = [physical[eval_id] for eval_id in ordered_ids]
    ordered_frozen = [frozen[eval_id] for eval_id in ordered_ids]
    diffs = []
    for eval_id in ordered_ids:
        before = frozen[eval_id]
        after = physical[eval_id]
        text_diff = before.get("prediction_text") != after.get("prediction_text")
        call_diff = before.get("prediction_calls") != after.get("prediction_calls")
        correctness_diff = before.get("normalized_correct") != after.get("normalized_correct")
        raw_correctness_diff = before.get("raw_correct") != after.get("raw_correct")
        if text_diff or call_diff or correctness_diff or raw_correctness_diff:
            diffs.append(
                {
                    "id": eval_id,
                    "category": next(row["category"] for row in pairs if str(row["id"]) == eval_id),
                    "text_diff": text_diff,
                    "call_diff": call_diff,
                    "normalized_correctness_diff": correctness_diff,
                    "raw_correctness_diff": raw_correctness_diff,
                    "frozen_prediction_text": before.get("prediction_text"),
                    "physical_prediction_text": after.get("prediction_text"),
                    "frozen_prediction_calls": before.get("prediction_calls"),
                    "physical_prediction_calls": after.get("prediction_calls"),
                    "frozen_normalized_correct": before.get("normalized_correct"),
                    "physical_normalized_correct": after.get("normalized_correct"),
                    "frozen_raw_correct": before.get("raw_correct"),
                    "physical_raw_correct": after.get("raw_correct"),
                }
            )

    category_for_id = {str(row["id"]): str(row["category"]) for row in pairs}
    split_rows = read_jsonl(args.split_manifest)
    split_for_id = {str(row["eval_id"]): str(row["split_role"]) for row in split_rows}
    if not expected.issubset(split_for_id):
        raise ValueError("split manifest does not cover every BFCL ID")

    category_scores = grouped_scores(ordered_ids, physical, frozen, category_for_id)
    split_scores = grouped_scores(ordered_ids, physical, frozen, split_for_id)
    physical_score = score_rows(ordered_physical)
    frozen_score = score_rows(ordered_frozen)
    losses = sum(
        bool(frozen[eval_id].get("normalized_correct"))
        and not bool(physical[eval_id].get("normalized_correct"))
        for eval_id in ordered_ids
    )
    gains = sum(
        not bool(frozen[eval_id].get("normalized_correct"))
        and bool(physical[eval_id].get("normalized_correct"))
        for eval_id in ordered_ids
    )
    summary = {
        "status": "complete",
        "pairs": str(args.pairs),
        "pairs_sha256": sha256(args.pairs),
        "frozen_predictions": str(args.frozen),
        "frozen_predictions_sha256": sha256(args.frozen),
        "physical_shards": [str(path) for path in args.physical],
        "physical": physical_score,
        "frozen": frozen_score,
        "normalized_correct_delta": (
            physical_score["normalized_correct"] - frozen_score["normalized_correct"]
        ),
        "raw_correct_delta": physical_score["raw_correct"] - frozen_score["raw_correct"],
        "normalized_correctness_losses": losses,
        "normalized_correctness_gains": gains,
        "text_diff_count": sum(row["text_diff"] for row in diffs),
        "call_diff_count": sum(row["call_diff"] for row in diffs),
        "normalized_correctness_diff_count": sum(
            row["normalized_correctness_diff"] for row in diffs
        ),
        "historical_cross_state_ratio_vs_base_664": (
            physical_score["normalized_correct"] / 664
        ),
        "cross_state_ratio_vs_preserved_live_dense_b007_672": (
            physical_score["normalized_correct"] / 672
        ),
        "physical_score_ratio_to_comparison": (
            physical_score["normalized_correct"] / frozen_score["normalized_correct"]
        ),
        "category_scores": category_scores,
        "split_scores": split_scores,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "physical_predictions.jsonl", ordered_physical)
    write_jsonl(args.out_dir / "prediction_diff.jsonl", diffs)
    write_csv(args.out_dir / "category_scores.csv", category_scores)
    write_csv(args.out_dir / "split_scores.csv", split_scores)
    (args.out_dir / "physical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
