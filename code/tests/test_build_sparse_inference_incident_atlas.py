from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_sparse_inference_incident_atlas import (  # noqa: E402
    build_atlas,
    build_coverage_summary,
    validate_annotations,
)


def reports() -> dict:
    return {
        "reports": [
            {
                "id": "vllm-5569-bnb-gibberish",
                "published_at": "2024-06-15",
                "source_url": "https://github.com/vllm-project/vllm/issues/5569",
            }
        ]
    }


def annotations() -> dict:
    return {
        "as_of": "2026-08-08",
        "incidents": [
            {
                "report_id": "vllm-5569-bnb-gibberish",
                "incident_type": "maintainer_incident",
                "model": "Llama-3-8B",
                "optimization": "BitsAndBytes 4-bit serving",
                "complaint_verbatim": "output is gibberish",
                "complaint_normalized": "The quantized serving path corrupted output.",
                "requirement": "Preserve generated text under serving.",
                "domains": ["serving_correctness"],
                "runtime": "vLLM",
                "hardware": "",
                "baseline": "Transformers generated normal output.",
                "resolution_status": "resolved",
                "resolution": "The integration was corrected.",
                "not_applicable": ["hardware"],
            }
        ],
    }


def test_builds_source_bound_rows_and_coverage() -> None:
    rows = build_atlas(reports(), annotations())

    assert len(rows) == 1
    assert rows[0]["source"] == "https://github.com/vllm-project/vllm/issues/5569"
    assert rows[0]["complaint_or_finding"] == (
        '"output is gibberish" — The quantized serving path corrupted output.'
    )
    assert rows[0]["runtime_and_hardware"] == "runtime: vLLM"
    assert rows[0]["resolution_and_status"] == (
        "resolved: The integration was corrected."
    )
    assert rows[0]["missing_information"] == []

    summary = build_coverage_summary(rows, "2026-08-08")
    assert summary["status"] == "complete"
    assert summary["row_count"] == 1
    assert summary["field_coverage"]["hardware"] == {
        "present": 0,
        "missing": 0,
        "not_applicable": 1,
    }
    assert summary["resolution_status_counts"] == {"resolved": 1}
    assert summary["unresolved_report_ids"] == []


def test_rejects_populated_not_applicable_fields() -> None:
    invalid = copy.deepcopy(annotations())
    invalid["incidents"][0]["hardware"] = "NVIDIA A100"

    with pytest.raises(ValueError, match="populated and marked not_applicable"):
        validate_annotations(reports(), invalid)


def test_rejects_annotation_coverage_mismatch() -> None:
    invalid = annotations()
    invalid["incidents"][0]["report_id"] = "unknown-report"

    with pytest.raises(ValueError, match="annotation coverage mismatch"):
        validate_annotations(reports(), invalid)
