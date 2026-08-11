from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

INCIDENT_TYPES = {
    "controlled_study",
    "maintainer_assessment",
    "maintainer_incident",
    "official_warning",
    "user_complaint",
}
RESOLUTION_STATUSES = {"mitigated", "not_applicable", "resolved", "unresolved"}
SIGNAL_RECEIPTS = {
    "vllm-5569-bnb-gibberish": "strong evidence; GitHub: 32 comments, 0 reactions; multiple users reproduced",
    "vllm-10819-marlin24-abnormal-output": "strong evidence; GitHub: 3 comments, 0 reactions; failing kernel isolated and fix linked",
    "vllm-11756-cutlass24-quant-crash": "strong evidence; GitHub: 2 comments, 0 reactions; second user reproduced and later master fixed it",
    "pytorch-153825-h100-sparse-slower": "strong evidence; GitHub: 8 comments, 0 reactions; maintainer reproduced",
    "vllm-21236-w4a16-small-speedup": "limited evidence; GitHub: 4 comments, 0 reactions; one benchmark report, closed not planned",
    "vllm-5793-gpu-dependent-marlin-quality": "strong evidence; GitHub: 8 comments, 0 reactions; fix plus independent 2xRTX-3090 confirmation",
    "llamacpp-21915-kv-cache-corruption": "strong evidence; GitHub: 10 comments, 0 reactions; official-Docker repro plus a second-model witness",
    "vllm-39583-quant-backend-fragmentation": "moderate evidence, polarized attention; GitHub: 16 comments, 15 reactions (6 +1, 9 -1); maintainer telemetry",
    "reddit-17xetyp-gptq-quality-speed-trade": "weak evidence; Reddit: score 8, 16 comments, 100% upvoted; one subjective report",
    "reddit-1us7a22-capability-specific-q4-loss": "weak evidence; Reddit: score 33, 50 comments, 86% upvoted; one unverified benchmark claim",
    "reddit-1sm2us4-q4-long-output-cliff": "weak evidence; Reddit: score 16, 41 comments, 72% upvoted; one model-dependent claim",
    "reddit-1c9u2jd-layer-pruning-hidden-loss": "moderate evidence, high attention; Reddit: score 241, 75 comments, 97% upvoted; benchmark plus author caveat",
    "arxiv-2505.20276-long-context-loss": "strong controlled evidence; 9.7K examples, 5 models, 5 quantizers; OpenAlex: 1 citation",
    "arxiv-2605.02404-task-vs-distribution-lossless": "strong controlled evidence; task- and distribution-level evaluation; OpenAlex: 0 citations (new 2026 preprint)",
    "llamacpp-quantize-requant-warning": "official implementation warning; no item-level engagement metric",
    "llamacpp-2094-low-bit-perplexity": "strong measurement; GitHub Discussion: 14 comments, 0 reactions, accepted answer; multi-bit perplexity table",
    "arxiv-2607.08786-moderate-sparsity-kernel-gap": "strong controlled evidence; specialized kernel versus dense and prior sparse baselines; OpenAlex: 0 citations (new 2026 preprint)",
    "x-2085977125845352679-q4-mtp-reply": "weak evidence; parent X post: 9 likes, 2 replies, 665 views; reply-level metrics unavailable",
    "llamacpp-25202-dsv4-quantized-kv-gibberish": "strong evidence; GitHub PR: 5 comments, 2 rocket reactions; root cause and reviewer-confirmed fix",
    "reddit-1rgl42y-qwen-local-runtime-gibberish": "moderate evidence, low agreement; Reddit: score 0, 34 comments, 37% upvoted; several quant/provider controls, one reporter",
    "x-2085707841357140439-prefix-cache-conditioned-corruption": "moderate within-run evidence; X: 3 likes, 3 replies, 199 views; 18/18 cache-busting rescues, no independent reproduction",
    "x-2085067196703600740-reap-pruning-language-loss": "moderate controlled practitioner evidence; X: 21 likes, 4 replies, 1 repost, 3,325 views; 150 matched scenarios",
    "arxiv-2208.07339-emergent-int8-outliers": "strong established evidence; Semantic Scholar: 1,164 citations; models from 125M to 176B",
    "arxiv-2306.00978-awq-salient-channel-loss": "strong established evidence; OpenAlex: 74 arXiv-record and 186 final-title-record citations; multi-model evaluation",
    "arxiv-2411.07191-super-weight-ablation": "strong controlled evidence; broad multi-model ablations; OpenAlex: 0 citations on the exact arXiv record",
}
COVERAGE_FIELDS = (
    "model",
    "optimization",
    "complaint_verbatim",
    "requirement",
    "domains",
    "runtime",
    "hardware",
    "baseline",
    "resolution",
)
ANNOTATION_FIELDS = {
    "report_id",
    "incident_type",
    "model",
    "optimization",
    "complaint_verbatim",
    "complaint_normalized",
    "requirement",
    "domains",
    "runtime",
    "hardware",
    "baseline",
    "resolution_status",
    "resolution",
    "not_applicable",
}
CSV_FIELDS = (
    "published_at",
    "report_id",
    "source",
    "signal_strength",
    "model",
    "optimization",
    "requirement_and_domains",
    "complaint_or_finding",
    "runtime_and_hardware",
    "baseline",
    "resolution_and_status",
    "missing_information",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def validate_annotations(
    reports_doc: dict[str, Any], annotations_doc: dict[str, Any]
) -> None:
    reports = reports_doc.get("reports")
    annotations = annotations_doc.get("incidents")
    if not isinstance(reports, list):
        raise ValueError("reports document must contain a reports list")
    if not isinstance(annotations, list):
        raise ValueError("annotations document must contain an incidents list")

    report_ids = [report.get("id") for report in reports]
    annotation_ids = [annotation.get("report_id") for annotation in annotations]
    if len(report_ids) != len(set(report_ids)):
        raise ValueError("duplicate report id")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate annotation report_id")

    missing = sorted(set(report_ids) - set(annotation_ids))
    unknown = sorted(set(annotation_ids) - set(report_ids))
    if missing or unknown:
        raise ValueError(
            f"annotation coverage mismatch: missing={missing}, unknown={unknown}"
        )

    for annotation in annotations:
        report_id = annotation.get("report_id", "<unknown>")
        absent_fields = sorted(ANNOTATION_FIELDS - set(annotation))
        extra_fields = sorted(set(annotation) - ANNOTATION_FIELDS)
        if absent_fields or extra_fields:
            raise ValueError(
                f"{report_id}: annotation fields mismatch: "
                f"missing={absent_fields}, extra={extra_fields}"
            )
        if annotation["incident_type"] not in INCIDENT_TYPES:
            raise ValueError(f"{report_id}: unknown incident_type")
        if annotation["resolution_status"] not in RESOLUTION_STATUSES:
            raise ValueError(f"{report_id}: unknown resolution_status")
        for field in ANNOTATION_FIELDS - {"domains", "not_applicable"}:
            if not isinstance(annotation[field], str):
                raise ValueError(f"{report_id}: {field} must be a string")
        domains = annotation["domains"]
        if (
            not isinstance(domains, list)
            or not domains
            or any(not isinstance(domain, str) or not domain.strip() for domain in domains)
        ):
            raise ValueError(f"{report_id}: domains must be a non-empty string list")
        not_applicable = annotation["not_applicable"]
        if not isinstance(not_applicable, list) or any(
            field not in COVERAGE_FIELDS for field in not_applicable
        ):
            raise ValueError(f"{report_id}: invalid not_applicable fields")
        if len(not_applicable) != len(set(not_applicable)):
            raise ValueError(f"{report_id}: duplicate not_applicable field")
        for field in not_applicable:
            value = annotation[field]
            if value:
                raise ValueError(
                    f"{report_id}: {field} is populated and marked not_applicable"
                )
        for required in ("optimization", "complaint_normalized", "requirement"):
            if not annotation[required].strip():
                raise ValueError(f"{report_id}: {required} must not be empty")


def _is_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return value is not None


def build_atlas(
    reports_doc: dict[str, Any], annotations_doc: dict[str, Any]
) -> list[dict[str, Any]]:
    validate_annotations(reports_doc, annotations_doc)
    annotations = {
        annotation["report_id"]: annotation
        for annotation in annotations_doc["incidents"]
    }
    rows: list[dict[str, Any]] = []
    for report in reports_doc["reports"]:
        annotation = annotations[report["id"]]
        missing_information = [
            field
            for field in COVERAGE_FIELDS
            if field not in annotation["not_applicable"]
            and not _is_present(annotation[field])
        ]
        complaint_or_finding = annotation["complaint_normalized"]
        if annotation["complaint_verbatim"]:
            complaint_or_finding = (
                f'"{annotation["complaint_verbatim"]}" — {complaint_or_finding}'
            )
        runtime_and_hardware = " | ".join(
            value
            for value in (
                f'runtime: {annotation["runtime"]}' if annotation["runtime"] else "",
                f'hardware: {annotation["hardware"]}' if annotation["hardware"] else "",
            )
            if value
        )
        resolution_and_status = annotation["resolution_status"]
        if annotation["resolution"]:
            resolution_and_status += f': {annotation["resolution"]}'
        rows.append(
            {
                "published_at": report["published_at"],
                "report_id": report["id"],
                "source": report["source_url"],
                "signal_strength": SIGNAL_RECEIPTS[report["id"]],
                "model": annotation["model"],
                "optimization": annotation["optimization"],
                "requirement_and_domains": " | ".join(
                    [annotation["requirement"], *annotation["domains"]]
                ),
                "complaint_or_finding": complaint_or_finding,
                "runtime_and_hardware": runtime_and_hardware,
                "baseline": annotation["baseline"],
                "resolution_and_status": resolution_and_status,
                "missing_information": missing_information,
                "complaint_verbatim": annotation["complaint_verbatim"],
                "requirement": annotation["requirement"],
                "domains": annotation["domains"],
                "runtime": annotation["runtime"],
                "hardware": annotation["hardware"],
                "resolution": annotation["resolution"],
                "resolution_status": annotation["resolution_status"],
                "not_applicable": annotation["not_applicable"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["published_at"], row["report_id"]),
        reverse=True,
    )


def build_coverage_summary(rows: list[dict[str, Any]], as_of: str) -> dict[str, Any]:
    coverage: dict[str, dict[str, int]] = {}
    for field in COVERAGE_FIELDS:
        counts = Counter()
        for row in rows:
            if field in row["not_applicable"]:
                counts["not_applicable"] += 1
            elif _is_present(row[field]):
                counts["present"] += 1
            else:
                counts["missing"] += 1
        coverage[field] = {
            "present": counts["present"],
            "missing": counts["missing"],
            "not_applicable": counts["not_applicable"],
        }

    return {
        "status": "complete",
        "as_of": as_of,
        "row_count": len(rows),
        "field_coverage": coverage,
        "resolution_status_counts": dict(
            sorted(Counter(row["resolution_status"] for row in rows).items())
        ),
        "unresolved_report_ids": [
            row["report_id"]
            for row in rows
            if row["resolution_status"] == "unresolved"
        ],
        "missingness_by_report": [
            {
                "report_id": row["report_id"],
                "missing_information": row["missing_information"],
            }
            for row in rows
            if row["missing_information"]
        ],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: " | ".join(row[field])
                    if isinstance(row[field], list)
                    else row[field]
                    for field in CSV_FIELDS
                }
            )


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized sparse-inference incident atlas."
    )
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_doc = load_json(args.reports)
    annotations_doc = load_json(args.annotations)
    rows = build_atlas(reports_doc, annotations_doc)
    summary = build_coverage_summary(rows, annotations_doc["as_of"])
    write_csv(rows, args.csv_output)
    write_summary(summary, args.summary_output)
    print(
        f"validated {len(rows)} incidents; wrote {args.csv_output} and "
        f"{args.summary_output}"
    )


if __name__ == "__main__":
    main()
