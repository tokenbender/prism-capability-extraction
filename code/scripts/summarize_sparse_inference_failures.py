from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


EVIDENCE_CLASSES = {
    "demonstrated",
    "verified_witness",
    "suggestive",
    "claim_only",
}
SOURCE_KINDS = {
    "github_issue",
    "github_pull_request",
    "github_discussion",
    "official_documentation",
    "reddit_post",
    "reddit_comment",
    "x_thread_reply",
    "x_post",
    "paper",
}
MECHANISM_STATUSES = {"confirmed", "source_hypothesis", "inferred", "unknown"}
SCORE_FIELDS = {
    "surprise",
    "compression",
    "mechanistic_contact",
    "generativity",
    "adjacency",
    "leverage",
    "transfer",
    "underpricing",
    "actionability",
}
REQUIRED_FIELDS = {
    "id",
    "title",
    "source_kind",
    "source_url",
    "published_at",
    "method_family",
    "primary_pattern",
    "secondary_patterns",
    "failure_surfaces",
    "claim",
    "observation",
    "mechanism",
    "mechanism_status",
    "evidence_class",
    "evidence_limit",
    "discriminating_test",
    "score",
}
STRONG_EVIDENCE = {"demonstrated", "verified_witness"}


def _count(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _require_text(report: dict[str, Any], field: str) -> None:
    value = report[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{report.get('id', '<unknown>')}: {field} must be non-empty text")


def validate_reports(reports_doc: dict[str, Any], scope: dict[str, Any]) -> list[dict[str, Any]]:
    reports = reports_doc.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("reports must be a non-empty list")

    pattern_ids = {pattern["id"] for pattern in scope["failure_patterns"]}
    window_start = date.fromisoformat(scope["corpus_contract"]["window_start"])
    window_end = date.fromisoformat(scope["corpus_contract"]["window_end"])
    seen_ids: set[str] = set()
    seen_observations: set[tuple[str, str]] = set()

    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("each report must be an object")
        missing = REQUIRED_FIELDS - report.keys()
        if missing:
            raise ValueError(f"{report.get('id', '<unknown>')}: missing {sorted(missing)}")

        for field in REQUIRED_FIELDS - {"secondary_patterns", "failure_surfaces", "score"}:
            _require_text(report, field)

        report_id = report["id"]
        if report_id in seen_ids:
            raise ValueError(f"duplicate report id: {report_id}")
        seen_ids.add(report_id)

        observation_key = (report["source_url"], report["observation"])
        if observation_key in seen_observations:
            raise ValueError(f"duplicate source observation: {report_id}")
        seen_observations.add(observation_key)

        if report["source_kind"] not in SOURCE_KINDS:
            raise ValueError(f"{report_id}: unknown source_kind {report['source_kind']}")
        if report["evidence_class"] not in EVIDENCE_CLASSES:
            raise ValueError(f"{report_id}: unknown evidence_class {report['evidence_class']}")
        if report["mechanism_status"] not in MECHANISM_STATUSES:
            raise ValueError(
                f"{report_id}: unknown mechanism_status {report['mechanism_status']}"
            )
        if not report["source_url"].startswith(("https://", "http://")):
            raise ValueError(f"{report_id}: source_url must be HTTP(S)")

        published_at = date.fromisoformat(report["published_at"])
        if not window_start <= published_at <= window_end:
            raise ValueError(f"{report_id}: published_at outside corpus window")

        secondary_patterns = report["secondary_patterns"]
        if not isinstance(secondary_patterns, list):
            raise ValueError(f"{report_id}: secondary_patterns must be a list")
        report_patterns = [report["primary_pattern"], *secondary_patterns]
        unknown_patterns = set(report_patterns) - pattern_ids
        if unknown_patterns:
            raise ValueError(f"{report_id}: unknown patterns {sorted(unknown_patterns)}")
        if len(report_patterns) != len(set(report_patterns)):
            raise ValueError(f"{report_id}: duplicate failure pattern")

        failure_surfaces = report["failure_surfaces"]
        if (
            not isinstance(failure_surfaces, list)
            or not failure_surfaces
            or not all(isinstance(value, str) and value for value in failure_surfaces)
        ):
            raise ValueError(f"{report_id}: failure_surfaces must be non-empty text")

        score = report["score"]
        if not isinstance(score, dict) or set(score) != SCORE_FIELDS:
            raise ValueError(f"{report_id}: score fields must be exactly {sorted(SCORE_FIELDS)}")
        for field, value in score.items():
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
                raise ValueError(f"{report_id}: score.{field} must be an integer from 0 to 3")

    return reports


def score_report(report: dict[str, Any]) -> dict[str, Any]:
    score = report["score"]
    adjusted_heat = (
        score["surprise"]
        + score["compression"]
        + score["mechanistic_contact"]
        + score["generativity"]
        - score["adjacency"]
    )
    alpha = (
        score["leverage"]
        + score["transfer"]
        + score["underpricing"]
        + score["actionability"]
    )
    return {
        "id": report["id"],
        "title": report["title"],
        "evidence_class": report["evidence_class"],
        "primary_pattern": report["primary_pattern"],
        "adjusted_heat": adjusted_heat,
        "alpha": alpha,
        "desire": adjusted_heat * alpha,
        "source_url": report["source_url"],
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-row["desire"], row["id"]))
    return [{"rank": index, **row} for index, row in enumerate(ranked, start=1)]


def build_summary(reports_doc: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    reports = validate_reports(reports_doc, scope)
    scored = [score_report(report) for report in reports]
    scored_by_id = {row["id"]: row for row in scored}

    pattern_reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for pattern in [report["primary_pattern"], *report["secondary_patterns"]]:
            pattern_reports[pattern].append(report)

    pattern_synthesis = []
    for pattern in sorted(pattern_reports):
        rows = pattern_reports[pattern]
        pattern_synthesis.append(
            {
                "pattern": pattern,
                "report_count": len(rows),
                "strong_receipt_count": sum(
                    row["evidence_class"] in STRONG_EVIDENCE for row in rows
                ),
                "report_ids": sorted(row["id"] for row in rows),
                "failure_surfaces": sorted(
                    {surface for row in rows for surface in row["failure_surfaces"]}
                ),
            }
        )

    def scored_class(evidence_classes: set[str]) -> list[dict[str, Any]]:
        return [
            scored_by_id[report["id"]]
            for report in reports
            if report["evidence_class"] in evidence_classes
        ]

    return {
        "status": "complete",
        "as_of": scope["as_of"],
        "research_question": scope["research_question"],
        "report_count": len(reports),
        "counts": {
            "source_kind": _count([report["source_kind"] for report in reports]),
            "evidence_class": _count([report["evidence_class"] for report in reports]),
            "method_family": _count([report["method_family"] for report in reports]),
            "primary_pattern": _count([report["primary_pattern"] for report in reports]),
        },
        "ranked_evidence": _rank(scored_class(STRONG_EVIDENCE)),
        "suggestive_observations": _rank(scored_class({"suggestive"})),
        "high_voltage_sleepers": _rank(scored_class({"claim_only"})),
        "pattern_synthesis": pattern_synthesis,
        "quality_gates": {
            "unique_report_ids": True,
            "unique_source_observations": True,
            "all_sources_linked": True,
            "all_dates_inside_corpus_window": True,
            "all_claims_evidence_bounded": True,
            "claim_only_excluded_from_ranked_evidence": True,
            "ranking_orthogonal_to_evidence_class": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and rank sparse-inference failure reports."
    )
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_doc = json.loads(args.reports.read_text())
    scope = json.loads(args.scope.read_text())
    summary = build_summary(reports_doc, scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"validated {summary['report_count']} reports; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
