from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_sparse_inference_failures import build_summary  # noqa: E402


def scope() -> dict[str, object]:
    return {
        "as_of": "2026-08-08",
        "research_question": "When does sparse inference fail?",
        "corpus_contract": {
            "window_start": "2023-01-01",
            "window_end": "2026-08-08",
        },
        "failure_patterns": [
            {"id": "kernel_correctness"},
            {"id": "decode_economics"},
        ],
    }


def report(
    report_id: str,
    *,
    evidence_class: str,
    primary_pattern: str = "kernel_correctness",
    source_url: str | None = None,
    source_kind: str = "github_issue",
) -> dict[str, object]:
    return {
        "id": report_id,
        "title": f"Report {report_id}",
        "source_kind": source_kind,
        "source_url": source_url or f"https://example.com/{report_id}",
        "published_at": "2025-01-01",
        "method_family": "weight_quantization",
        "primary_pattern": primary_pattern,
        "secondary_patterns": ["decode_economics"]
        if primary_pattern == "kernel_correctness"
        else [],
        "failure_surfaces": ["correctness"],
        "claim": "The sparse path fails.",
        "observation": f"Observed failure {report_id}.",
        "mechanism": "A backend kernel violates the dense-path contract.",
        "mechanism_status": "confirmed",
        "evidence_class": evidence_class,
        "evidence_limit": "One backend version.",
        "discriminating_test": "Run dense and sparse kernels on identical inputs.",
        "score": {
            "surprise": 3,
            "compression": 3,
            "mechanistic_contact": 3,
            "generativity": 2,
            "adjacency": 1,
            "leverage": 3,
            "transfer": 2,
            "underpricing": 2,
            "actionability": 3,
        },
    }


def test_summary_keeps_evidence_strength_orthogonal_to_ranking() -> None:
    reports_doc = {
        "reports": [
            report("strong", evidence_class="demonstrated"),
            report(
                "sleeper",
                evidence_class="claim_only",
                primary_pattern="decode_economics",
            ),
        ]
    }

    summary = build_summary(reports_doc, scope())

    assert summary["report_count"] == 2
    assert [row["id"] for row in summary["ranked_evidence"]] == ["strong"]
    assert [row["id"] for row in summary["high_voltage_sleepers"]] == ["sleeper"]
    assert summary["ranked_evidence"][0]["desire"] == 100
    assert summary["counts"]["evidence_class"] == {
        "claim_only": 1,
        "demonstrated": 1,
    }
    patterns = {row["pattern"]: row for row in summary["pattern_synthesis"]}
    assert patterns["decode_economics"]["report_count"] == 2
    assert patterns["decode_economics"]["strong_receipt_count"] == 1


@pytest.mark.parametrize("source_kind", ["github_pull_request", "x_post"])
def test_summary_accepts_primary_source_kinds(source_kind: str) -> None:
    summary = build_summary(
        {
            "reports": [
                report(
                    source_kind,
                    evidence_class="verified_witness",
                    source_kind=source_kind,
                )
            ]
        },
        scope(),
    )

    assert summary["counts"]["source_kind"] == {source_kind: 1}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate report id"),
        (
            lambda rows: rows[0]["score"].update({"surprise": 4}),
            "score.surprise",
        ),
        (
            lambda rows: rows[0].update({"primary_pattern": "unknown"}),
            "unknown patterns",
        ),
        (
            lambda rows: rows[0].update({"published_at": "2027-01-01"}),
            "outside corpus window",
        ),
    ],
)
def test_summary_rejects_invalid_corpus(mutation, message: str) -> None:
    rows = [report("one", evidence_class="verified_witness")]
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        build_summary({"reports": rows}, scope())
