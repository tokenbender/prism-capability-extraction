from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

CSV_FIELDS = (
    "published_at",
    "incident_id",
    "lineage_id",
    "source_kind",
    "source_url",
    "source_title",
    "provider_or_repository",
    "model",
    "exact_file",
    "optimization",
    "requirement_domains",
    "complaint_or_finding",
    "observed_failure",
    "baseline_or_control",
    "runtime_hardware",
    "primary_pattern",
    "secondary_patterns",
    "failure_surfaces",
    "resolution_status",
    "resolution",
    "evidence_class",
    "mechanism_status",
    "mechanism",
    "discriminating_test",
    "missing_information",
    "attention_receipt",
    "evidence_receipt",
    "lineage_source_count",
    "cross_source_lineage",
)

REQUIRED_ROW_FIELDS = {
    "incident_id",
    "published_at",
    "source_kind",
    "source_url",
    "complaint_or_finding",
    "observed_failure",
    "optimization",
    "requirement_domains",
    "resolution_status",
    "evidence_class",
    "mechanism_status",
    "missing_information",
}

RESOLUTION_STATUSES = {"mitigated", "not_applicable", "resolved", "unknown", "unresolved"}
EVIDENCE_CLASSES = {"claim_only", "suggestive", "verified_witness", "demonstrated"}
MECHANISM_STATUSES = {"confirmed", "inferred", "source_hypothesis", "unknown"}
SCHEMA_FACTS = {
    "model": "model",
    "exact_file": "exact model file",
    "optimization": "optimization recipe",
    "runtime_hardware": "runtime and hardware",
    "baseline_or_control": "baseline or control",
    "resolution": "resolution",
    "mechanism": "mechanism",
    "discriminating_test": "discriminating test",
}
FAILURE_TAG_PATTERNS = {
    "gibberish": r"gibber|garbage|nonsense|corrupt(?:ed|ion)? output",
    "crash": r"crash|segfault|illegal memory|device lost|runtimeerror|assert",
    "load": r"fail(?:ed|s)? to load|load(?:ing)? failure|unsupported architecture|missing tensor",
    "slow": r"slow|throughput|tok(?:en)?s?/s|latency|no speedup|regression",
    "oom": r"out of memory|\boom\b|memory fit|vram",
    "tool": r"tool.?call|structured output|invalid json|grammar",
    "template": r"chat template|jinja|tokenizer|eos|bos",
    "cache": r"kv.?cache|prefix.?cache|state reuse|cache corruption",
    "repeat": r"repeat|loop|degenerate",
    "quality": r"quality|accuracy|perplexity|benchmark score|capability",
    "multilingual": r"multilingual|language loss|non.?english|wrong language",
    "multimodal": r"multimodal|vision|image|mmproj",
}


def load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def load_legacy_signal_receipts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["report_id"]: row.get("signal_strength", "")
            for row in csv.DictReader(handle)
        }


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip()))


def _date_string(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        try:
            return date.fromisoformat(text[:10]).isoformat()
        except ValueError:
            return ""


def _stable_id(prefix: str, source_url: str, incident_key: str) -> str:
    digest = hashlib.sha1(f"{source_url}\0{incident_key}".encode()).hexdigest()[:14]
    return f"{prefix}-{digest}"


def _source_kind(source: dict[str, Any]) -> str:
    explicit = _clean_string(source.get("source_kind"))
    if explicit:
        return explicit
    if "provider" in source and "repo" in source and "num" in source:
        return "huggingface_discussion"
    if "html_url" in source and "repository_url" in source:
        return "github_issue"
    if "permalink" in source and "subreddit" in source:
        return "reddit_post"
    if "objectID" in source:
        return f"hackernews_{source.get('tag', 'record')}"
    if "abstract" in source or "publication_year" in source or "cited_by_count" in source:
        return "paper"
    return "unknown"


def _source_url(source: dict[str, Any], kind: str) -> str:
    if kind == "github_issue":
        return _clean_string(source.get("html_url"))
    if kind == "reddit_post" and source.get("permalink"):
        return f"https://www.reddit.com{source['permalink']}"
    if kind.startswith("hackernews_") and source.get("objectID"):
        return f"https://news.ycombinator.com/item?id={source['objectID']}"
    for field in ("url", "html_url", "doi"):
        value = _clean_string(source.get(field))
        if value:
            return value
    location = source.get("primary_location")
    if isinstance(location, dict):
        return _clean_string(location.get("landing_page_url"))
    return ""


def _published_at(source: dict[str, Any]) -> str:
    for field in ("created_at", "createdAt", "published_at", "publication_date", "created_utc"):
        value = source.get(field)
        result = _date_string(value)
        if result:
            return result
    return ""


def _source_title(source: dict[str, Any]) -> str:
    return _clean_string(source.get("title") or source.get("story_title") or source.get("query_title"))


def _source_author(source: dict[str, Any]) -> str:
    author = source.get("author") or source.get("user")
    if isinstance(author, dict):
        return _clean_string(author.get("login") or author.get("name"))
    return _clean_string(author)


def _provider_or_repository(source: dict[str, Any], kind: str) -> str:
    if kind == "huggingface_discussion":
        return _clean_string(source.get("repo") or source.get("provider"))
    if kind == "github_issue":
        repository_url = _clean_string(source.get("repository_url"))
        if "/repos/" in repository_url:
            return repository_url.split("/repos/", 1)[1]
        return _clean_string(source.get("repo"))
    if kind == "reddit_post":
        return f"r/{source.get('subreddit', '')}".rstrip("/")
    if kind.startswith("hackernews_"):
        return "Hacker News"
    if kind == "paper":
        return "controlled study"
    if kind == "official_documentation":
        return _source_author(source)
    return ""


def _attention_receipt(source: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "huggingface_discussion":
        return {
            "comments": source.get("comments_count"),
            "reaction_users": source.get("reaction_users"),
            "reactions": source.get("reactions"),
            "model_downloads": source.get("model_downloads"),
            "model_likes": source.get("model_likes"),
        }
    if kind == "github_issue":
        return {
            "comments": source.get("comments"),
            "reactions": source.get("reactions"),
            "state": source.get("state"),
            "state_reason": source.get("state_reason"),
        }
    if kind == "reddit_post":
        return {
            "score": source.get("score"),
            "comments": source.get("num_comments"),
            "upvote_ratio": source.get("upvote_ratio"),
        }
    if kind.startswith("hackernews_"):
        return {"points": source.get("points"), "comments": source.get("num_comments")}
    if kind == "paper":
        return {"citations": source.get("cited_by_count")}
    if kind == "x_post":
        return {
            field: source.get(field)
            for field in ("likes", "replies", "reposts", "bookmarks", "views")
        }
    return {"official_source": bool(source.get("official"))} if source.get("official") else {}


def _evidence_receipt(
    incident: dict[str, Any], source: dict[str, Any], kind: str
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "evidence_class": incident["evidence_class"],
        "mechanism_status": incident["mechanism_status"],
        "explicit_baseline": bool(incident["baseline_or_control"]),
        "resolution_status": incident["resolution_status"],
    }
    if kind == "huggingface_discussion":
        receipt.update(
            {
                "conversation_comments": len(source.get("comments") or []),
                "owner_participating": bool(source.get("owner_participating")),
            }
        )
    elif kind == "github_issue":
        receipt["issue_body_captured"] = bool(source.get("body"))
        receipt["reply_text_captured"] = False
    elif kind == "reddit_post":
        conversation = incident.get("conversation_receipt")
        if isinstance(conversation, dict):
            receipt["replies_captured"] = conversation.get("comments_retrieved", 0)
        else:
            receipt["replies_captured"] = 0
    elif kind.startswith("hackernews_"):
        receipt["record_type"] = source.get("tag")
    elif kind == "paper":
        receipt["controlled_study"] = True
    elif kind == "official_documentation":
        receipt["official_source"] = True
    return receipt


def normalized_incident_to_row(incident: dict[str, Any]) -> dict[str, Any]:
    source = incident.get("source")
    if not isinstance(source, dict):
        raise ValueError("normalized incident missing source object")
    kind = _source_kind(source)
    source_url = _source_url(source, kind)
    incident_key = _clean_string(incident.get("incident_key")) or "unkeyed"
    prefix = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-") or "incident"
    row = {
        "incident_id": _stable_id(prefix, source_url, incident_key),
        "incident_key": incident_key,
        "published_at": _published_at(source),
        "source_kind": kind,
        "source_url": source_url,
        "source_title": _source_title(source),
        "source_author": _source_author(source),
        "provider_or_repository": _provider_or_repository(source, kind),
        "model": _clean_string(incident.get("model")),
        "exact_file": _clean_string(incident.get("exact_file")),
        "optimization": _clean_string(incident.get("optimization")),
        "method_family": _clean_string(incident.get("method_family")),
        "requirement_domains": _string_list(incident.get("requirement_domains")),
        "complaint_or_finding": _clean_string(incident.get("complaint_or_finding")),
        "observed_failure": _clean_string(incident.get("observed_failure")),
        "baseline_or_control": _clean_string(incident.get("baseline_or_control")),
        "runtime_hardware": _clean_string(incident.get("runtime_hardware")),
        "primary_pattern": _clean_string(incident.get("primary_pattern")) or "unknown",
        "secondary_patterns": _string_list(incident.get("secondary_patterns")),
        "failure_surfaces": _string_list(incident.get("failure_surfaces")),
        "resolution_status": _clean_string(incident.get("resolution_status")) or "unknown",
        "resolution": _clean_string(incident.get("resolution")),
        "evidence_class": _clean_string(incident.get("evidence_class")) or "claim_only",
        "mechanism_status": _clean_string(incident.get("mechanism_status")) or "unknown",
        "mechanism": _clean_string(incident.get("mechanism")),
        "discriminating_test": _clean_string(incident.get("discriminating_test")),
        "missing_information": _string_list(incident.get("missing_information")),
    }
    if not row["observed_failure"] and row["complaint_or_finding"]:
        row["observed_failure"] = row["complaint_or_finding"]
        row["missing_information"].append("distinct observed failure")
    if not row["complaint_or_finding"] and row["observed_failure"]:
        row["complaint_or_finding"] = row["observed_failure"]
        row["missing_information"].append("normalized complaint")
    if not row["requirement_domains"]:
        row["requirement_domains"] = row["failure_surfaces"] or [row["primary_pattern"]]
        row["missing_information"].append("explicit requirement domains")
    row["missing_information"] = list(dict.fromkeys(row["missing_information"]))
    row["attention_receipt"] = _attention_receipt(source, kind)
    row["evidence_receipt"] = _evidence_receipt(incident, source, kind)
    return row


def legacy_reports_to_rows(
    reports_doc: dict[str, Any],
    annotations_doc: dict[str, Any],
    signal_receipts: dict[str, str],
) -> list[dict[str, Any]]:
    annotations = {
        item["report_id"]: item
        for item in annotations_doc.get("incidents", [])
        if isinstance(item, dict) and item.get("report_id")
    }
    rows: list[dict[str, Any]] = []
    for report in reports_doc.get("reports", []):
        annotation = annotations.get(report.get("id"))
        if not isinstance(annotation, dict):
            raise ValueError(f"legacy report {report.get('id')}: missing annotation")
        configuration = report.get("configuration") or {}
        runtime_hardware = " | ".join(
            value
            for value in (
                f"runtime: {annotation.get('runtime', '')}" if annotation.get("runtime") else "",
                f"hardware: {annotation.get('hardware', '')}" if annotation.get("hardware") else "",
            )
            if value
        )
        missing = [
            field
            for field in ("model", "runtime", "hardware", "baseline", "resolution")
            if not annotation.get(field) and field not in annotation.get("not_applicable", [])
        ]
        row = {
            "incident_id": report["id"],
            "incident_key": report["id"],
            "published_at": _date_string(report.get("published_at")),
            "source_kind": _clean_string(report.get("source_kind")) or "unknown",
            "source_url": _clean_string(report.get("source_url")),
            "source_title": _clean_string(report.get("title")),
            "source_author": "",
            "provider_or_repository": urlparse(report.get("source_url", "")).netloc,
            "model": _clean_string(annotation.get("model")),
            "exact_file": _clean_string(configuration.get("file") or configuration.get("checkpoint")),
            "optimization": _clean_string(annotation.get("optimization")),
            "method_family": _clean_string(report.get("method_family")),
            "requirement_domains": _string_list(annotation.get("domains")),
            "complaint_or_finding": _clean_string(annotation.get("complaint_normalized")),
            "observed_failure": _clean_string(report.get("observation")),
            "baseline_or_control": _clean_string(annotation.get("baseline")),
            "runtime_hardware": runtime_hardware,
            "primary_pattern": _clean_string(report.get("primary_pattern")) or "unknown",
            "secondary_patterns": _string_list(report.get("secondary_patterns")),
            "failure_surfaces": _string_list(report.get("failure_surfaces")),
            "resolution_status": _clean_string(annotation.get("resolution_status")) or "unknown",
            "resolution": _clean_string(annotation.get("resolution")),
            "evidence_class": _clean_string(report.get("evidence_class")) or "claim_only",
            "mechanism_status": _clean_string(report.get("mechanism_status")) or "unknown",
            "mechanism": _clean_string(report.get("mechanism")),
            "discriminating_test": _clean_string(report.get("discriminating_test")),
            "missing_information": missing,
            "attention_receipt": {"legacy_receipt": signal_receipts.get(report["id"], "")},
            "evidence_receipt": {
                "evidence_class": _clean_string(report.get("evidence_class")) or "claim_only",
                "mechanism_status": _clean_string(report.get("mechanism_status")) or "unknown",
                "explicit_baseline": bool(annotation.get("baseline")),
                "evidence_limit": _clean_string(report.get("evidence_limit")),
            },
        }
        rows.append(row)
    return rows


def annotate_schema_missingness(row: dict[str, Any]) -> dict[str, Any]:
    missing = _string_list(row.get("missing_information"))
    for field, label in SCHEMA_FACTS.items():
        if not row.get(field) and label not in missing:
            missing.append(label)
    row["missing_information"] = missing
    return row


def validate_rows(rows: list[dict[str, Any]], window_start: str, as_of: str) -> None:
    incident_ids = [row.get("incident_id") for row in rows]
    if len(incident_ids) != len(set(incident_ids)):
        duplicate_ids = [item for item, count in Counter(incident_ids).items() if count > 1]
        raise ValueError(f"duplicate incident ids: {duplicate_ids[:10]}")
    for row in rows:
        missing_fields = REQUIRED_ROW_FIELDS - set(row)
        if missing_fields:
            raise ValueError(f"{row.get('incident_id')}: missing fields {sorted(missing_fields)}")
        if not row["source_url"]:
            raise ValueError(f"{row['incident_id']}: missing source URL")
        if not row["published_at"]:
            raise ValueError(f"{row['incident_id']}: missing published date")
        if not window_start <= row["published_at"] <= as_of:
            raise ValueError(f"{row['incident_id']}: date outside corpus window")
        if row["resolution_status"] not in RESOLUTION_STATUSES:
            raise ValueError(f"{row['incident_id']}: invalid resolution status")
        if row["evidence_class"] not in EVIDENCE_CLASSES:
            raise ValueError(f"{row['incident_id']}: invalid evidence class")
        if row["mechanism_status"] not in MECHANISM_STATUSES:
            raise ValueError(f"{row['incident_id']}: invalid mechanism status")
        if not row["requirement_domains"]:
            raise ValueError(f"{row['incident_id']}: empty requirement domains")
        if not row["complaint_or_finding"] or not row["observed_failure"]:
            raise ValueError(f"{row['incident_id']}: missing claim or observation")


def _canonical_text(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    drop = {"model", "quantized", "quantization", "gguf", "checkpoint", "instruct", "official"}
    return "-".join(token for token in tokens if token not in drop)


def _failure_tags(row: dict[str, Any]) -> set[str]:
    text = " ".join(
        [row["source_title"], row["complaint_or_finding"], row["observed_failure"]]
    ).lower()
    return {
        name for name, pattern in FAILURE_TAG_PATTERNS.items() if re.search(pattern, text)
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def join_lineages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    union_find = UnionFind(len(rows))
    exact_url_groups: dict[str, list[int]] = defaultdict(list)
    exact_file_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    model_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    tags = [_failure_tags(row) for row in rows]

    for index, row in enumerate(rows):
        exact_url_groups[row["source_url"]].append(index)
        file_key = _canonical_text(row["exact_file"])
        if len(file_key) >= 12:
            exact_file_groups[(file_key, row["primary_pattern"])].append(index)
        model_key = _canonical_text(row["model"])
        if len(model_key) >= 12:
            model_groups[(model_key, row["primary_pattern"])].append(index)

    for indices in exact_url_groups.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    for indices in exact_file_groups.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    for indices in model_groups.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if tags[left] and tags[right] and tags[left] & tags[right]:
                    union_find.union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[union_find.find(index)].append(index)

    for indices in groups.values():
        incident_ids = sorted(rows[index]["incident_id"] for index in indices)
        lineage_id = "lineage-" + hashlib.sha1("\0".join(incident_ids).encode()).hexdigest()[:14]
        source_urls = sorted({rows[index]["source_url"] for index in indices})
        source_kinds = sorted({rows[index]["source_kind"] for index in indices})
        for index in indices:
            rows[index]["lineage_id"] = lineage_id
            rows[index]["lineage_sources"] = source_urls
            rows[index]["lineage_source_count"] = len(source_urls)
            rows[index]["cross_source_lineage"] = len(source_kinds) > 1
    return rows


def build_coverage(rows: list[dict[str, Any]], as_of: str, minimum: int) -> dict[str, Any]:
    source_counts = Counter(row["source_kind"] for row in rows)
    pattern_counts = Counter(row["primary_pattern"] for row in rows)
    resolution_counts = Counter(row["resolution_status"] for row in rows)
    evidence_counts = Counter(row["evidence_class"] for row in rows)
    lineage_counts = Counter(row["lineage_id"] for row in rows)
    missing_counts = Counter(
        field for row in rows for field in row.get("missing_information", [])
    )
    unresolved = [
        row for row in rows if row["resolution_status"] in {"unknown", "unresolved"}
    ]
    schema_missing_counts = {
        field: sum(not bool(row.get(field)) for row in rows)
        for field in SCHEMA_FACTS
    }
    paper_count = sum(row["source_kind"] == "paper" for row in rows)
    hf_count = sum(row["source_kind"] == "huggingface_discussion" for row in rows)
    gates = {
        "minimum_deduplicated_lineages": len(lineage_counts) >= minimum,
        "minimum_huggingface_rows": hf_count >= 70,
        "minimum_controlled_studies": paper_count >= 15,
        "all_rows_source_bound": all(row["source_url"] for row in rows),
        "all_rows_observation_bound": all(row["observed_failure"] for row in rows),
        "all_rows_missingness_explicit": all(
            isinstance(row["missing_information"], list) for row in rows
        ),
        "attention_separated_from_evidence": all(
            isinstance(row["attention_receipt"], dict)
            and isinstance(row["evidence_receipt"], dict)
            for row in rows
        ),
    }
    return {
        "status": "complete" if all(gates.values()) else "failed_quality_gate",
        "as_of": as_of,
        "row_count": len(rows),
        "deduplicated_lineage_count": len(lineage_counts),
        "cross_source_lineage_count": sum(
            1
            for lineage_id in lineage_counts
            if any(
                row["lineage_id"] == lineage_id and row["cross_source_lineage"]
                for row in rows
            )
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "primary_pattern_counts": dict(sorted(pattern_counts.items())),
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "evidence_class_counts": dict(sorted(evidence_counts.items())),
        "missing_information_counts": dict(sorted(missing_counts.items())),
        "schema_missing_counts": schema_missing_counts,
        "unresolved_by_pattern": dict(
            sorted(Counter(row["primary_pattern"] for row in unresolved).items())
        ),
        "unresolved_by_source": dict(
            sorted(Counter(row["source_kind"] for row in unresolved).items())
        ),
        "quality_gates": gates,
    }


def unresolved_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["resolution_status"] in {"unknown", "unresolved"}
    ]


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate the sparse-inference complaint census.")
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--legacy-atlas", type=Path)
    parser.add_argument("--normalized", type=Path, action="append", default=[], required=True)
    parser.add_argument("--window-start", default="2022-08-15")
    parser.add_argument("--as-of", default="2026-08-08")
    parser.add_argument("--minimum-lineages", type=int, default=250)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--unresolved-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_doc = load_object(args.reports)
    annotations_doc = load_object(args.annotations)
    rows = legacy_reports_to_rows(
        reports_doc,
        annotations_doc,
        load_legacy_signal_receipts(args.legacy_atlas),
    )
    for path in args.normalized:
        document = load_object(path)
        incidents = document.get("incidents")
        if not isinstance(incidents, list):
            raise ValueError(f"{path}: incidents must be a list")
        rows.extend(normalized_incident_to_row(item) for item in incidents if item.get("include", True))

    rows = [
        annotate_schema_missingness(row)
        for row in {row["incident_id"]: row for row in rows}.values()
    ]
    validate_rows(rows, args.window_start, args.as_of)
    rows = join_lineages(rows)
    rows.sort(key=lambda row: (row["published_at"], row["incident_id"]), reverse=True)
    coverage = build_coverage(rows, args.as_of, args.minimum_lineages)
    if coverage["status"] != "complete":
        failed = [name for name, passed in coverage["quality_gates"].items() if not passed]
        raise ValueError(f"quality gates failed: {failed}")

    write_json({"status": "complete", "as_of": args.as_of, "incidents": rows}, args.json_output)
    write_csv(rows, args.csv_output)
    write_json(coverage, args.coverage_output)
    write_csv(unresolved_rows(rows), args.unresolved_output)
    print(
        f"validated {len(rows)} source-bound rows across "
        f"{coverage['deduplicated_lineage_count']} incident lineages"
    )


if __name__ == "__main__":
    main()
