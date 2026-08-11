from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_sparse_inference_complaint_census import (  # noqa: E402
    build_coverage,
    join_lineages,
    normalized_incident_to_row,
    unresolved_rows,
    validate_rows,
)


def normalized_incident(source: dict, **overrides: object) -> dict:
    incident = {
        "model": "Example-Model-7B",
        "exact_file": "example-model-q4_k_m.gguf",
        "optimization": "Q4_K_M GGUF",
        "runtime_hardware": "llama.cpp on RTX 4090",
        "requirement_domains": ["serving_correctness"],
        "complaint_or_finding": "The quantized model emitted gibberish.",
        "observed_failure": "Deterministic prompts produced gibberish output.",
        "baseline_or_control": "The full-precision model answered normally.",
        "resolution_status": "unresolved",
        "resolution": "",
        "missing_information": ["runtime commit"],
        "incident_key": "example-model-q4-gibberish",
        "method_family": "weight_quantization",
        "primary_pattern": "kernel_correctness",
        "secondary_patterns": ["composition_incompatibility"],
        "failure_surfaces": ["serving_correctness"],
        "evidence_class": "suggestive",
        "mechanism_status": "unknown",
        "mechanism": "",
        "discriminating_test": "Compare the same file in two runtimes.",
        "source": source,
    }
    incident.update(overrides)
    return incident


def test_normalizes_huggingface_receipts_without_conflating_evidence() -> None:
    row = normalized_incident_to_row(
        normalized_incident(
            {
                "provider": "uploader",
                "repo": "uploader/example-model-GGUF",
                "num": 3,
                "url": "https://huggingface.co/uploader/example-model-GGUF/discussions/3",
                "title": "Gibberish output",
                "created_at": "2025-01-02T03:04:05Z",
                "author": "reporter",
                "comments_count": 7,
                "reaction_users": 2,
                "owner_participating": True,
                "comments": [{"text": "reproduced"}, {"text": "fixed"}],
            }
        )
    )

    assert row["published_at"] == "2025-01-02"
    assert row["source_kind"] == "huggingface_discussion"
    assert row["attention_receipt"]["comments"] == 7
    assert row["evidence_receipt"]["conversation_comments"] == 2
    assert row["evidence_receipt"]["owner_participating"] is True
    assert "comments" not in row["evidence_class"]


def test_preserves_claim_when_source_does_not_distinguish_observation() -> None:
    row = normalized_incident_to_row(
        normalized_incident(
            {
                "permalink": "/r/LocalLLaMA/comments/example/failure/",
                "subreddit": "LocalLLaMA",
                "title": "Failure",
                "created_utc": 1_735_689_600,
            },
            observed_failure="",
            requirement_domains=[],
        )
    )

    assert row["observed_failure"] == row["complaint_or_finding"]
    assert "distinct observed failure" in row["missing_information"]
    assert row["requirement_domains"] == ["serving_correctness"]
    assert row["source_url"].startswith("https://www.reddit.com/r/LocalLLaMA/")
    assert "explicit requirement domains" in row["missing_information"]


def test_lineage_join_is_conservative_and_marks_cross_source() -> None:
    hf_row = normalized_incident_to_row(
        normalized_incident(
            {
                "provider": "uploader",
                "repo": "uploader/example-model-GGUF",
                "num": 1,
                "url": "https://huggingface.co/uploader/example-model-GGUF/discussions/1",
                "title": "Gibberish",
                "created_at": "2025-01-01",
            }
        )
    )
    github_row = normalized_incident_to_row(
        normalized_incident(
            {
                "html_url": "https://github.com/runtime/runtime/issues/1",
                "repository_url": "https://api.github.com/repos/runtime/runtime",
                "title": "Same file emits gibberish",
                "created_at": "2025-01-02",
                "body": "Reproduction",
            },
            incident_key="runtime-example-gibberish",
        )
    )
    assert github_row["source_url"] == "https://github.com/runtime/runtime/issues/1"
    different_failure = copy.deepcopy(github_row)
    different_failure["incident_id"] = "different"
    different_failure["source_url"] = "https://github.com/runtime/runtime/issues/2"
    different_failure["primary_pattern"] = "decode_economics"
    different_failure["complaint_or_finding"] = "The model was slow."
    different_failure["observed_failure"] = "Decode throughput was 2 tokens per second."

    rows = join_lineages([hf_row, github_row, different_failure])

    assert rows[0]["lineage_id"] == rows[1]["lineage_id"]
    assert rows[0]["cross_source_lineage"] is True
    assert rows[0]["lineage_source_count"] == 2
    assert rows[2]["lineage_id"] != rows[0]["lineage_id"]


def test_validate_rows_rejects_out_of_window_dates() -> None:
    row = normalized_incident_to_row(
        normalized_incident(
            {
                "permalink": "/r/LocalLLaMA/comments/example/failure/",
                "subreddit": "LocalLLaMA",
                "title": "Failure",
                "created_utc": 1_600_000_000,
            }
        )
    )
    with pytest.raises(ValueError, match="outside corpus window"):
        validate_rows([row], "2022-08-15", "2026-08-08")


def test_quality_gates_measure_lineages_and_source_minima() -> None:
    rows = []
    for index in range(255):
        if index < 70:
            kind_source = {
                "provider": "provider",
                "repo": f"provider/model-{index}-GGUF",
                "num": index,
                "url": f"https://huggingface.co/provider/model-{index}-GGUF/discussions/{index}",
                "title": "Failure",
                "created_at": "2025-01-01",
            }
        elif index < 85:
            kind_source = {
                "title": f"Controlled study {index}",
                "published_at": "2025-01-01",
                "url": f"https://arxiv.org/abs/{index}",
                "abstract": "Controlled failure result",
                "cited_by_count": 0,
            }
        else:
            kind_source = {
                "permalink": f"/r/LocalLLaMA/comments/{index}/failure/",
                "subreddit": "LocalLLaMA",
                "title": "Failure",
                "created_utc": 1_735_689_600,
            }
        row = normalized_incident_to_row(
            normalized_incident(
                kind_source,
                model=f"Unique-Model-{index:04d}",
                exact_file="",
                incident_key=f"unique-failure-{index}",
            )
        )
        rows.append(row)

    rows = join_lineages(rows)
    coverage = build_coverage(rows, "2026-08-08", 250)

    assert coverage["status"] == "complete"
    assert coverage["deduplicated_lineage_count"] == 255
    assert coverage["source_counts"]["huggingface_discussion"] == 70
    assert coverage["source_counts"]["paper"] == 15


def test_unresolved_output_includes_unknown_and_unresolved_only() -> None:
    rows = [
        {"incident_id": "a", "resolution_status": "unknown"},
        {"incident_id": "b", "resolution_status": "unresolved"},
        {"incident_id": "c", "resolution_status": "resolved"},
    ]
    assert [row["incident_id"] for row in unresolved_rows(rows)] == ["a", "b"]
