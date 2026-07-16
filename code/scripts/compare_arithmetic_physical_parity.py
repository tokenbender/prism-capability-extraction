#!/usr/bin/env python3
"""Compare logical-zero and physical arithmetic predictions by frozen row ID."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "prism_arithmetic_physical_parity_v1"
# A gate rejection is expected search control flow. Keep it distinct from
# Python/runtime failures so callers never continue after a broken comparison.
GATE_FAILURE_EXIT_CODE = 3
ACCEPTANCE_SCHEMA_VERSION = "prism_arithmetic_physical_acceptance_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no predictions in {path}")
    return rows


def index_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = str(row["id"])
        if record_id in indexed:
            raise ValueError(f"duplicate prediction ID {record_id!r}")
        indexed[record_id] = row
    return indexed


def compare_predictions(
    logical_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    *,
    retention_floor: float = 0.99,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 < retention_floor <= 1.0:
        raise ValueError("retention floor must be in (0, 1]")
    logical = index_predictions(logical_rows)
    physical = index_predictions(physical_rows)
    if logical.keys() != physical.keys():
        raise ValueError("logical and physical prediction ID sets differ")

    logical_correct_ids = {
        record_id
        for record_id, row in logical.items()
        if bool(row["exact_numeric_correct"])
    }
    if not logical_correct_ids:
        raise ValueError("logical substrate has zero correct predictions")
    physical_correct_ids = {
        record_id
        for record_id, row in physical.items()
        if bool(row["exact_numeric_correct"])
    }
    retained_ids = logical_correct_ids & physical_correct_ids
    identical_ids = {
        record_id
        for record_id in logical
        if (
            logical[record_id].get("prediction_text")
            == physical[record_id].get("prediction_text")
            and logical[record_id].get("prediction_token_ids")
            == physical[record_id].get("prediction_token_ids")
        )
    }
    differences = []
    for record_id in sorted(logical):
        logical_row = logical[record_id]
        physical_row = physical[record_id]
        same_prediction = record_id in identical_ids
        same_correctness = bool(
            logical_row["exact_numeric_correct"]
        ) == bool(physical_row["exact_numeric_correct"])
        if same_prediction and same_correctness:
            continue
        differences.append(
            {
                "id": record_id,
                "logical_prediction_text": logical_row.get("prediction_text"),
                "physical_prediction_text": physical_row.get("prediction_text"),
                "logical_prediction_token_ids": logical_row.get(
                    "prediction_token_ids"
                ),
                "physical_prediction_token_ids": physical_row.get(
                    "prediction_token_ids"
                ),
                "logical_correct": bool(
                    logical_row["exact_numeric_correct"]
                ),
                "physical_correct": bool(
                    physical_row["exact_numeric_correct"]
                ),
            }
        )
    retention = len(retained_ids) / len(logical_correct_ids)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if retention >= retention_floor else "fail",
        "rows": len(logical),
        "logical_correct": len(logical_correct_ids),
        "physical_correct": len(physical_correct_ids),
        "retained_logical_correct": len(retained_ids),
        "logical_correctness_retention": retention,
        "retention_floor": retention_floor,
        "identical_prediction_rows": len(identical_ids),
        "identical_prediction_fraction": len(identical_ids) / len(logical),
        "logical_only_correct": len(logical_correct_ids - physical_correct_ids),
        "physical_only_correct": len(physical_correct_ids - logical_correct_ids),
        "difference_rows": len(differences),
    }
    return summary, differences


def compare_dense_recovery(
    dense_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    *,
    recovery_floor: float = 0.90,
) -> dict[str, Any]:
    if not 0.0 < recovery_floor <= 1.0:
        raise ValueError("recovery floor must be in (0, 1]")
    dense = index_predictions(dense_rows)
    physical = index_predictions(physical_rows)
    if dense.keys() != physical.keys():
        raise ValueError("dense and physical prediction ID sets differ")

    dense_correct_ids = {
        record_id
        for record_id, row in dense.items()
        if bool(row["exact_numeric_correct"])
    }
    if not dense_correct_ids:
        raise ValueError("dense parent has zero correct predictions")
    physical_correct_ids = {
        record_id
        for record_id, row in physical.items()
        if bool(row["exact_numeric_correct"])
    }
    matched_ids = dense_correct_ids & physical_correct_ids
    recovery = len(matched_ids) / len(dense_correct_ids)
    return {
        "status": "pass" if recovery >= recovery_floor else "fail",
        "dense_correct": len(dense_correct_ids),
        "physical_correct": len(physical_correct_ids),
        "matched_dense_correct": len(matched_ids),
        "matched_dense_recovery": recovery,
        "recovery_floor": recovery_floor,
        "dense_only_correct": len(dense_correct_ids - physical_correct_ids),
        "physical_only_correct": len(physical_correct_ids - dense_correct_ids),
    }


def compare_physical_acceptance(
    logical_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    *,
    retention_floor: float = 0.99,
    recovery_floor: float = 0.90,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, differences = compare_predictions(
        logical_rows,
        physical_rows,
        retention_floor=retention_floor,
    )
    parity_status = summary["status"]
    dense_recovery = compare_dense_recovery(
        dense_rows,
        physical_rows,
        recovery_floor=recovery_floor,
    )
    summary["schema_version"] = ACCEPTANCE_SCHEMA_VERSION
    summary["parity_status"] = parity_status
    summary["dense_recovery"] = dense_recovery
    summary["gates"] = {
        "logical_correctness_retention": {
            "status": parity_status,
            "value": summary["logical_correctness_retention"],
            "floor": retention_floor,
        },
        "matched_dense_recovery": {
            "status": dense_recovery["status"],
            "value": dense_recovery["matched_dense_recovery"],
            "floor": recovery_floor,
        },
    }
    summary["status"] = (
        "pass"
        if parity_status == "pass" and dense_recovery["status"] == "pass"
        else "fail"
    )
    return summary, differences


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument(
        "--dense",
        type=Path,
        help=(
            "frozen dense predictions; when supplied, status also requires "
            "matched-dense recovery to pass"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--differences", type=Path, required=True)
    parser.add_argument("--retention-floor", type=float, default=0.99)
    parser.add_argument("--recovery-floor", type=float, default=0.90)
    args = parser.parse_args()

    logical_rows = read_jsonl(args.logical)
    physical_rows = read_jsonl(args.physical)
    if args.dense is None:
        summary, differences = compare_predictions(
            logical_rows,
            physical_rows,
            retention_floor=args.retention_floor,
        )
    else:
        summary, differences = compare_physical_acceptance(
            logical_rows,
            physical_rows,
            read_jsonl(args.dense),
            retention_floor=args.retention_floor,
            recovery_floor=args.recovery_floor,
        )
    summary["inputs"] = {
        "logical": {
            "path": str(args.logical),
            "sha256": sha256_file(args.logical),
        },
        "physical": {
            "path": str(args.physical),
            "sha256": sha256_file(args.physical),
        },
    }
    if args.dense is not None:
        summary["inputs"]["dense"] = {
            "path": str(args.dense),
            "sha256": sha256_file(args.dense),
        }
    summary["differences_path"] = str(args.differences)
    write_json(args.output, summary)
    write_jsonl(args.differences, differences)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(GATE_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
