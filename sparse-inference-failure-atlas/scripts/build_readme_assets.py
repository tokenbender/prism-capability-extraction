#!/usr/bin/env python3
"""Build aggregate data and SVG figures for the public write-up."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

SOURCE_GROUPS = (
    ("GitHub", {"github_issue", "github_pull_request", "github_discussion"}),
    ("Reddit", {"reddit_post"}),
    ("Hugging Face", {"huggingface_discussion"}),
    ("Hacker News", {"hackernews_comment", "hackernews_story"}),
    ("Papers and official docs", {"paper", "official_documentation"}),
    ("X", {"x_post", "x_thread_reply"}),
)

OPERATIONAL_GROUPS = (
    (
        "Protocol, runtime, composition, or loading",
        {
            "template_tokenizer_mismatch",
            "composition_incompatibility",
            "model_loading_failure",
            "support_fragmentation",
            "unknown",
        },
    ),
    (
        "Model or task-selective behavior",
        {
            "capability_specific_quality",
            "task_selective_quality",
            "benchmark_blindness",
            "pruning_recovery_gap",
        },
    ),
    (
        "Representation, state, conversion, or kernels",
        {
            "stateful_precision_failure",
            "kernel_correctness",
            "conversion_pipeline_damage",
        },
    ),
    (
        "Economics, capacity, or serving overhead",
        {"decode_economics", "representation_overhead", "memory_representation"},
    ),
)

RUNTIME_SYMPTOM_PATTERNS = {
    "kernel_correctness",
    "composition_incompatibility",
    "stateful_precision_failure",
    "conversion_pipeline_damage",
    "template_tokenizer_mismatch",
    "support_fragmentation",
    "model_loading_failure",
}

SYMPTOM_RE = re.compile(
    r"\b(?:gibberish|nonsense|garbage|repetitive|repetition|looping|loops)\b"
    r"|(?:!\s*){4,}|exclamation marks",
    re.IGNORECASE,
)

COLORS = ("#7C3AED", "#2563EB", "#0891B2", "#0F766E", "#CA8A04", "#C2410C")
INK = "#17202A"
MUTED = "#5D6873"
LINE = "#D8D2C5"
PAPER = "#FFFDF8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="complaint_census.json")
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser.parse_args()


def lineage_id(row: dict[str, Any]) -> str:
    return str(row.get("lineage_id") or row.get("incident_key") or row["incident_id"])


def grouped_count(rows: list[dict[str, Any]], groups: tuple[tuple[str, set[str]], ...], field: str) -> list[dict[str, Any]]:
    known = set().union(*(members for _, members in groups))
    observed = {str(row.get(field, "")) for row in rows}
    unknown = observed - known
    if unknown:
        raise ValueError(f"unmapped {field} values: {sorted(unknown)}")
    return [
        {
            "label": label,
            "count": sum(str(row.get(field, "")) in members for row in rows),
        }
        for label, members in groups
    ]


def summarize(document: dict[str, Any]) -> dict[str, Any]:
    rows = document["incidents"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("incidents must be a non-empty list")

    published = sorted(str(row["published_at"]) for row in rows if row.get("published_at"))
    sources = grouped_count(rows, SOURCE_GROUPS, "source_kind")
    if sum(item["count"] for item in sources) != len(rows):
        raise ValueError("source groups do not cover the corpus")

    tool_rows = [row for row in rows if "tool_calling" in (row.get("requirement_domains") or [])]
    evidence = Counter(str(row.get("evidence_class")) for row in tool_rows)
    resolution = Counter(str(row.get("resolution_status")) for row in tool_rows)
    operational = grouped_count(tool_rows, OPERATIONAL_GROUPS, "primary_pattern")
    if sum(item["count"] for item in operational) != len(tool_rows):
        raise ValueError("operational groups do not cover the tool-calling cohort")

    symptom_rows = [
        row
        for row in rows
        if SYMPTOM_RE.search(
            " ".join(
                str(row.get(field, ""))
                for field in ("source_title", "observed_failure", "complaint_or_finding")
            )
        )
    ]
    symptom_lineages = {lineage_id(row) for row in symptom_rows}
    confirmed_rows = [row for row in symptom_rows if row.get("mechanism_status") == "confirmed"]
    confirmed_patterns: dict[str, set[str]] = defaultdict(set)
    for row in confirmed_rows:
        confirmed_patterns[lineage_id(row)].add(str(row.get("primary_pattern")))
    ambiguous = {key: value for key, value in confirmed_patterns.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"confirmed lineages have inconsistent primary labels: {ambiguous}")
    runtime_lineages = {
        key
        for key, patterns in confirmed_patterns.items()
        if next(iter(patterns)) in RUNTIME_SYMPTOM_PATTERNS
    }

    return {
        "as_of": document.get("as_of"),
        "window": {"start": published[0], "end": published[-1]},
        "corpus": {
            "records": len(rows),
            "source_groups": sources,
        },
        "tool_calling_cohort": {
            "records": len(tool_rows),
            "evidence": {
                "demonstrated": evidence["demonstrated"],
                "verified_witness": evidence["verified_witness"],
                "suggestive": evidence["suggestive"],
                "claim_only": evidence["claim_only"],
                "strong_total": evidence["demonstrated"] + evidence["verified_witness"],
            },
            "mechanism_confirmed": sum(
                row.get("mechanism_status") == "confirmed" for row in tool_rows
            ),
            "resolution": {
                "unresolved": resolution["unresolved"],
                "unknown": resolution["unknown"],
                "resolved": resolution["resolved"],
                "mitigated": resolution["mitigated"],
                "not_applicable": resolution["not_applicable"],
            },
            "single_source_lineage": sum(
                int(row.get("lineage_source_count", 0)) == 1 for row in tool_rows
            ),
            "missing_exact_model_file": sum(not row.get("exact_file") for row in tool_rows),
            "operational_layers": operational,
        },
        "visible_nonsense_slice": {
            "records": len(symptom_rows),
            "lineages": len(symptom_lineages),
            "primary_labels": len({row.get("primary_pattern") for row in symptom_rows}),
            "confirmed_lineages": len(confirmed_patterns),
            "runtime_pipeline_configuration_lineages": len(runtime_lineages),
            "other_lineages": len(confirmed_patterns) - len(runtime_lineages),
        },
    }


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title>",
        f"<desc id=\"desc\">{escape(description)}</desc>",
        f'<rect width="{width}" height="{height}" rx="18" fill="{PAPER}"/>',
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="17.5" fill="none" stroke="{LINE}"/>',
        f'<text x="48" y="52" fill="{INK}" font-family="system-ui, sans-serif" font-size="25" font-weight="750">{escape(title)}</text>',
        f'<text x="48" y="79" fill="{MUTED}" font-family="system-ui, sans-serif" font-size="14">{escape(description)}</text>',
    ]


def horizontal_bars(
    title: str,
    description: str,
    rows: list[dict[str, Any]],
    note: str,
    *,
    height: int,
    scale_max: int | None = None,
) -> str:
    width = 960
    left = 390
    right = 84
    bar_width = width - left - right
    max_value = scale_max or max(int(row["count"]) for row in rows)
    lines = svg_start(width, height, title, description)
    y = 118
    for index, row in enumerate(rows):
        count = int(row["count"])
        fill_width = round(bar_width * count / max_value)
        color = COLORS[index % len(COLORS)]
        label = str(row["label"])
        lines.extend(
            [
                f'<text x="48" y="{y + 20}" fill="{INK}" font-family="system-ui, sans-serif" font-size="15">{escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="28" rx="6" fill="#EEEAE1"/>',
                f'<rect x="{left}" y="{y}" width="{fill_width}" height="28" rx="6" fill="{color}"/>',
                f'<text x="{left + bar_width + 16}" y="{y + 20}" fill="{INK}" font-family="system-ui, sans-serif" font-size="15" font-weight="700">{count}</text>',
            ]
        )
        y += 48
    lines.append(
        f'<text x="48" y="{height - 28}" fill="{MUTED}" font-family="system-ui, sans-serif" font-size="13">{escape(note)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def evidence_bottleneck_svg(summary: dict[str, Any]) -> str:
    tool = summary["tool_calling_cohort"]
    rows = [
        {"label": "Demonstrated or verified", "count": tool["evidence"]["strong_total"]},
        {"label": "Mechanism confirmed", "count": tool["mechanism_confirmed"]},
        {"label": "Unresolved", "count": tool["resolution"]["unresolved"]},
        {"label": "Exact model file missing", "count": tool["missing_exact_model_file"]},
        {"label": "Single-source lineage", "count": tool["single_source_lineage"]},
    ]
    return horizontal_bars(
        "Most reports prove breakage, not cause",
        "Independent counts within the 101-record tool-calling cohort",
        rows,
        "Counts overlap. Strong evidence of failure does not imply a confirmed mechanism.",
        height=410,
        scale_max=tool["records"],
    )


def symptom_svg(summary: dict[str, Any]) -> str:
    data = summary["visible_nonsense_slice"]
    runtime = int(data["runtime_pipeline_configuration_lineages"])
    other = int(data["other_lineages"])
    total = runtime + other
    full = 824
    runtime_width = round(full * runtime / total)
    lines = svg_start(
        960,
        330,
        "Visible nonsense does not identify the damaged layer",
        "A symptom search crossed kernels, state, conversion, templates, loading, and model behavior",
    )
    stats = (
        (str(data["records"]), "matching records"),
        (str(data["lineages"]), "source lineages"),
        (str(data["primary_labels"]), "primary labels"),
        (str(data["confirmed_lineages"]), "confirmed lineages"),
    )
    for index, (number, label) in enumerate(stats):
        x = 48 + index * 210
        lines.extend(
            [
                f'<text x="{x}" y="132" fill="{COLORS[index]}" font-family="system-ui, sans-serif" font-size="27" font-weight="800">{escape(number)}</text>',
                f'<text x="{x}" y="155" fill="{MUTED}" font-family="system-ui, sans-serif" font-size="13">{escape(label)}</text>',
            ]
        )
    lines.extend(
        [
            f'<rect x="48" y="190" width="{full}" height="42" rx="8" fill="#EEEAE1"/>',
            f'<rect x="48" y="190" width="{runtime_width}" height="42" rx="8" fill="{COLORS[0]}"/>',
            f'<text x="64" y="217" fill="#FFFFFF" font-family="system-ui, sans-serif" font-size="15" font-weight="750">{runtime} runtime, pipeline, or configuration lineages</text>',
            f'<text x="{48 + runtime_width + 10}" y="217" fill="{INK}" font-family="system-ui, sans-serif" font-size="14" font-weight="750">{other} other</text>',
            f'<text x="48" y="274" fill="{MUTED}" font-family="system-ui, sans-serif" font-size="13">Failure-mined corpus; this is a diagnostic warning, not a population failure rate.</text>',
        ]
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(document)

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    (args.assets_dir / "evidence-bottleneck.svg").write_text(
        evidence_bottleneck_svg(summary), encoding="utf-8"
    )
    (args.assets_dir / "nonsense-is-not-a-diagnosis.svg").write_text(
        symptom_svg(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
