#!/usr/bin/env python3
"""Quality-gated throughput sweep for dense and physical arithmetic models.

The harness follows the BFCL physical-runtime workflow while using the exact
standalone arithmetic contract: no counterfactual donor activations are loaded,
every row gets the same fixed generation cap, and generated text is scored as a
strict first-line integer.  Every candidate/batch attempt is appended to JSONL
before the sweep continues, including quality failures, unsupported
configurations, and runtime failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from evaluate_arithmetic_standalone import (
    DEFAULT_PROMPT_FORMATS,
    PROMPT_FORMATS,
    SCORER_CONTRACT,
    _special_token_ids,
    _trim_generated_ids,
    build_two_digit_records,
    carry_fields,
    score_prediction,
    sha256_json,
    summarize_predictions,
)
from load_arithmetic_physical_bundle import load_arithmetic_physical_bundle
from load_bfcl_physical_bundle import (
    DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    SUPPORTED_ACTIVATION_IMPLEMENTATIONS,
    SUPPORTED_ATTENTION_IMPLEMENTATIONS,
    SUPPORTED_MLP_IMPLEMENTATIONS,
    SUPPORTED_WIDTH_ALIGNMENTS,
    build_generation_compile_settings,
    observe_generation_compile_state,
    validate_activation_runtime_settings,
)


COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
CACHE_IMPLEMENTATIONS = ("dynamic", "static")
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class UnsupportedCandidate(RuntimeError):
    """Raised when a candidate is valid but unavailable in this environment."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    role: str
    model_kind: str
    attention_implementation: str = "eager"
    mlp_implementation: str = "separate"
    activation_implementation: str = "torch"
    hybrid_activation_threshold_rows: int = (
        DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
    )
    width_alignment: int = 1
    outer_compile_mode: str = "none"
    cache_implementation: str = "dynamic"
    generation_disable_compile: bool = False
    generation_compile_dynamic: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CandidateSpec":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown candidate fields: {unknown}")
        return cls(**raw)


@dataclass
class EncodedBatch:
    indices: list[int]
    records: list[dict[str, Any]]
    encoded: dict[str, torch.Tensor]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    normalized = [float(value) for value in values]
    return {
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
        "p50": percentile(normalized, 0.50),
        "p95": percentile(normalized, 0.95),
        "min": min(normalized),
        "max": max(normalized),
        "stdev": statistics.stdev(normalized)
        if len(normalized) > 1
        else 0.0,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([dict(record) for record in records])


def _normalize_record(raw: dict[str, Any], index: int) -> dict[str, Any]:
    source = raw.get("target") if isinstance(raw.get("target"), dict) else raw
    prompt = source.get("prompt")
    answer = source.get("answer")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"record {index} has no nonempty string prompt")
    try:
        numeric_answer = int(answer)
    except (TypeError, ValueError) as error:
        raise ValueError(f"record {index} has no integer answer") from error
    if isinstance(answer, float) and not answer.is_integer():
        raise ValueError(f"record {index} has a non-integral answer")
    record_id = raw.get("id", source.get("id", index))
    generation_prompt = source.get(
        "generation_prompt",
        raw.get("generation_prompt"),
    )
    if generation_prompt is not None and not isinstance(generation_prompt, str):
        raise ValueError(f"record {index} generation_prompt must be a string")

    a = source.get("a")
    b = source.get("b")
    if a is not None or b is not None:
        try:
            a = int(a)
            b = int(b)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"record {index} must provide integer a and b together"
            ) from error
        if a + b != numeric_answer:
            raise ValueError(
                f"record {index} answer {numeric_answer} != {a} + {b}"
            )
        carry = carry_fields(a, b)
    else:
        carry = {
            "carry": str(source.get("carry", "unknown")),
            "carry_class": str(source.get("carry_class", "unknown")),
            "ones_carry": source.get("ones_carry"),
            "leading_carry": source.get("leading_carry"),
        }

    normalized = {
        "id": str(record_id),
        "prompt": prompt,
        "prompt_format": str(source.get("prompt_format", "custom")),
        "answer": numeric_answer,
        "answer_text": str(source.get("answer_text", numeric_answer)),
        "result_length": int(
            source.get("result_length", len(str(numeric_answer)))
        ),
        **carry,
    }
    if a is not None and b is not None:
        normalized.update({"a": a, "b": b})
    if generation_prompt is not None:
        normalized["generation_prompt"] = generation_prompt
    if "source_position" in source:
        normalized["source_position"] = str(source["source_position"])
    if "source_index" in source:
        normalized["source_index"] = int(source["source_index"])
    return normalized


def load_arithmetic_records(
    path: Path | None,
    *,
    min_value: int = 10,
    max_value: int = 99,
    prompt_formats: Sequence[str] = DEFAULT_PROMPT_FORMATS,
    num_records: int = 1_500,
    seed: int = 123,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load frozen records or build the standalone evaluator's balanced set."""

    if path is None:
        records = build_two_digit_records(
            min_value=min_value,
            max_value=max_value,
            prompt_formats=prompt_formats,
            num_records=num_records,
            seed=seed,
        )
    elif path.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict):
            for key in ("records", "rows", "pairs"):
                if isinstance(payload.get(key), list):
                    raw_rows = payload[key]
                    break
            else:
                raise ValueError(
                    f"{path} must contain a list or records/rows/pairs list"
                )
        else:
            raise ValueError(f"{path} must contain JSON records")
        records = [
            _normalize_record(raw, index)
            for index, raw in enumerate(raw_rows)
            if isinstance(raw, dict)
        ]
        if len(records) != len(raw_rows):
            raise ValueError("every arithmetic record must be a JSON object")
    if limit is not None:
        if limit <= 0:
            raise ValueError("record limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError("arithmetic benchmark selection is empty")
    source_position_presence = [
        "source_position" in record for record in records
    ]
    if any(source_position_presence) and not all(source_position_presence):
        raise ValueError(
            "source_position must be present on either every record or none"
        )
    return records


def default_candidate_specs(
    *,
    include_dense: bool,
    include_physical: bool,
    include_default_optimized: bool = True,
) -> list[CandidateSpec]:
    """Return a deliberately diverse but bounded default runtime matrix."""

    candidates: list[CandidateSpec] = []
    if include_dense:
        candidates.append(
            CandidateSpec(
                name="dense_parent_sdpa_dynamic",
                role="dense_parent",
                model_kind="dense",
                attention_implementation="sdpa",
            )
        )
    if include_physical:
        candidates.append(
            CandidateSpec(
                name="canonical_eager_physical",
                role="canonical_physical",
                model_kind="physical",
            )
        )
    if include_physical and include_default_optimized:
        candidates.extend(
            [
                CandidateSpec(
                    name="physical_sdpa_packed128_dynamic_torch",
                    role="optimized_physical",
                    model_kind="physical",
                    attention_implementation="sdpa",
                    mlp_implementation="packed_gate_up",
                    width_alignment=128,
                ),
                CandidateSpec(
                    name="physical_sdpa_packed128_static_torch",
                    role="optimized_physical",
                    model_kind="physical",
                    attention_implementation="sdpa",
                    mlp_implementation="packed_gate_up",
                    width_alignment=128,
                    cache_implementation="static",
                ),
                CandidateSpec(
                    name="physical_sdpa_packed128_static_hybrid",
                    role="optimized_physical",
                    model_kind="physical",
                    attention_implementation="sdpa",
                    mlp_implementation="packed_gate_up",
                    activation_implementation="hybrid",
                    width_alignment=128,
                    cache_implementation="static",
                ),
            ]
        )
    return candidates


def validate_candidate_spec(candidate: CandidateSpec) -> None:
    if not candidate.name.strip():
        raise ValueError("candidate name must not be empty")
    if candidate.role not in {
        "dense_parent",
        "canonical_physical",
        "optimized_physical",
    }:
        raise ValueError(f"unsupported candidate role {candidate.role!r}")
    if candidate.model_kind not in {"dense", "physical"}:
        raise ValueError(f"unsupported model kind {candidate.model_kind!r}")
    if candidate.attention_implementation not in (
        SUPPORTED_ATTENTION_IMPLEMENTATIONS
    ):
        raise ValueError(
            "unsupported attention implementation "
            f"{candidate.attention_implementation!r}"
        )
    if candidate.mlp_implementation not in SUPPORTED_MLP_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported MLP implementation {candidate.mlp_implementation!r}"
        )
    if (
        candidate.activation_implementation
        not in SUPPORTED_ACTIVATION_IMPLEMENTATIONS
    ):
        raise ValueError(
            "unsupported activation implementation "
            f"{candidate.activation_implementation!r}"
        )
    if candidate.width_alignment not in SUPPORTED_WIDTH_ALIGNMENTS:
        raise ValueError(
            f"unsupported width alignment {candidate.width_alignment!r}"
        )
    if candidate.outer_compile_mode not in COMPILE_MODES:
        raise ValueError(
            f"unsupported outer compile mode {candidate.outer_compile_mode!r}"
        )
    if candidate.cache_implementation not in CACHE_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported cache implementation {candidate.cache_implementation!r}"
        )
    if candidate.model_kind == "dense" and (
        candidate.mlp_implementation != "separate"
        or candidate.activation_implementation != "torch"
        or candidate.width_alignment != 1
    ):
        raise ValueError("dense candidates cannot request physical MLP repacking")
    if (
        candidate.mlp_implementation == "separate"
        and candidate.width_alignment != 1
    ):
        raise ValueError("width alignment requires packed_gate_up")
    if (
        candidate.mlp_implementation == "separate"
        and candidate.activation_implementation != "torch"
    ):
        raise ValueError("non-Torch activation requires packed_gate_up")
    if (
        candidate.outer_compile_mode != "none"
        and candidate.cache_implementation == "static"
        and not candidate.generation_disable_compile
    ):
        raise ValueError(
            "outer compilation and automatic StaticCache compilation cannot "
            "both own generation"
        )
    validate_activation_runtime_settings(
        activation_implementation=candidate.activation_implementation,
        hybrid_activation_threshold_rows=(
            candidate.hybrid_activation_threshold_rows
        ),
    )
    build_generation_compile_settings(
        cache_implementation=candidate.cache_implementation,
        disable_compile=candidate.generation_disable_compile,
        compile_dynamic=candidate.generation_compile_dynamic,
    )


def load_candidate_config(path: Path) -> list[CandidateSpec]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("candidates")
    if not isinstance(payload, list):
        raise ValueError("candidate config must contain a candidates list")
    candidates = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"candidate config row {index} must be an object")
        candidates.append(CandidateSpec.from_mapping(raw))
    return candidates


def score_generated_sequences(
    sequences: torch.Tensor,
    *,
    prompt_width: int,
    batch: EncodedBatch,
    tokenizer: Any,
    stop_ids: set[int],
    pad_token_id: int | None,
) -> tuple[list[tuple[int, dict[str, Any]]], int, int]:
    """Decode and score one batch with the standalone evaluator contract."""

    if sequences.ndim != 2:
        raise TypeError("model.generate must return a rank-two token tensor")
    if int(sequences.shape[0]) != len(batch.records):
        raise ValueError("generation batch size does not match input records")
    suffix = sequences[:, prompt_width:]
    indexed_predictions = []
    generated_tokens = 0
    for index, record, token_ids in zip(
        batch.indices,
        batch.records,
        suffix.detach().cpu().tolist(),
    ):
        trimmed = _trim_generated_ids(
            token_ids,
            stop_ids=stop_ids,
            pad_token_id=pad_token_id,
        )
        generated_tokens += len(trimmed)
        prediction_text = tokenizer.decode(
            trimmed,
            skip_special_tokens=True,
        )
        indexed_predictions.append(
            (
                index,
                score_prediction(
                    record,
                    prediction_text=prediction_text,
                    prediction_token_ids=trimmed,
                ),
            )
        )
    return indexed_predictions, generated_tokens, int(suffix.numel())


def prediction_digest(
    indexed_predictions: Iterable[tuple[int, Mapping[str, Any]]],
) -> str:
    stable_rows = []
    for index, prediction in sorted(indexed_predictions):
        stable_rows.append(
            {
                "index": index,
                "id": prediction.get("id"),
                "prediction_text": prediction["prediction_text"],
                "prediction_token_ids": prediction["prediction_token_ids"],
                "parsed_answer": prediction["parsed_answer"],
                "exact_numeric_correct": prediction[
                    "exact_numeric_correct"
                ],
            }
        )
    return sha256_json(stable_rows)


def quality_gate(
    *,
    correct: int,
    examples: int,
    floor_accuracy: float,
    floor_correct: int | None,
) -> dict[str, Any]:
    accuracy = correct / examples
    required_correct = (
        floor_correct
        if floor_correct is not None
        else math.ceil(floor_accuracy * examples - 1e-12)
    )
    return {
        "status": "pass" if correct >= required_correct else "fail",
        "pass": correct >= required_correct,
        "correct": correct,
        "examples": examples,
        "accuracy": accuracy,
        "floor_accuracy": floor_accuracy
        if floor_correct is None
        else required_correct / examples,
        "floor_correct": required_correct,
    }


def prepare_batches(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> list[EncodedBatch]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    batches: list[EncodedBatch] = []
    for start in range(0, len(records), batch_size):
        batch_records = [
            dict(record) for record in records[start : start + batch_size]
        ]
        prompts = [
            str(record.get("generation_prompt", record["prompt"]))
            for record in batch_records
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded_tensors = {
            key: value.to(device)
            for key, value in dict(encoded).items()
            if isinstance(value, torch.Tensor)
        }
        batches.append(
            EncodedBatch(
                indices=list(range(start, start + len(batch_records))),
                records=batch_records,
                encoded=encoded_tensors,
            )
        )
    return batches


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generate_fixed_cap(
    model: Any,
    batch: EncodedBatch,
    *,
    max_new_tokens: int,
    generation_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    output = model.generate(
        **batch.encoded,
        **dict(generation_kwargs),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=False,
    )
    sequences = getattr(output, "sequences", output)
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise TypeError("model.generate must return a rank-two token tensor")
    return sequences


def run_generation_once(
    model: Any,
    batches: Sequence[EncodedBatch],
    *,
    tokenizer: Any,
    device: torch.device,
    generation_kwargs: dict[str, Any],
    max_new_tokens: int,
    stop_ids: set[int],
    pad_token_id: int | None,
) -> dict[str, Any]:
    """Measure synchronized generation only, then decode and score off-clock."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generation_seconds = 0.0
    generated_tokens = 0
    generated_token_slots = 0
    indexed_predictions: list[tuple[int, dict[str, Any]]] = []
    with torch.inference_mode():
        for batch in batches:
            _cuda_sync(device)
            started = time.perf_counter()
            sequences = _generate_fixed_cap(
                model,
                batch,
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
            )
            _cuda_sync(device)
            generation_seconds += time.perf_counter() - started
            prompt_width = int(batch.encoded["input_ids"].shape[-1])
            predictions, token_count, token_slots = (
                score_generated_sequences(
                    sequences,
                    prompt_width=prompt_width,
                    batch=batch,
                    tokenizer=tokenizer,
                    stop_ids=stop_ids,
                    pad_token_id=pad_token_id,
                )
            )
            generated_tokens += token_count
            generated_token_slots += token_slots
            indexed_predictions.extend(predictions)
    ordered_predictions = [
        prediction for _, prediction in sorted(indexed_predictions)
    ]
    quality_summary = summarize_predictions(ordered_predictions)
    examples = len(ordered_predictions)
    correct = int(quality_summary["correct"])
    return {
        "elapsed_seconds": generation_seconds,
        "measurement_contract": (
            "sum of synchronized pretokenized model.generate calls; "
            "decode and scoring excluded"
        ),
        "examples": examples,
        "correct": correct,
        "accuracy": float(quality_summary["accuracy"]),
        "quality_summary": quality_summary,
        "generated_tokens": generated_tokens,
        "generated_token_slots": generated_token_slots,
        "examples_per_second": examples / generation_seconds,
        "generated_tokens_per_second": (
            generated_tokens / generation_seconds
        ),
        "generated_token_slots_per_second": (
            generated_token_slots / generation_seconds
        ),
        "correct_examples_per_second": correct / generation_seconds,
        "prediction_sha256": prediction_digest(indexed_predictions),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else None,
    }


def measure_batch_latency(
    model: Any,
    batches: Sequence[EncodedBatch],
    *,
    device: torch.device,
    generation_kwargs: dict[str, Any],
    max_new_tokens: int,
    repeats: int,
) -> list[dict[str, Any]]:
    measurements = []
    for repeat in range(repeats):
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches):
                _cuda_sync(device)
                started = time.perf_counter()
                sequences = _generate_fixed_cap(
                    model,
                    batch,
                    max_new_tokens=max_new_tokens,
                    generation_kwargs=generation_kwargs,
                )
                _cuda_sync(device)
                elapsed = time.perf_counter() - started
                prompt_width = int(batch.encoded["input_ids"].shape[-1])
                generated_token_slots = int(
                    sequences[:, prompt_width:].numel()
                )
                measurements.append(
                    {
                        "repeat": repeat,
                        "batch_index": batch_index,
                        "examples": len(batch.indices),
                        "generated_token_slots": generated_token_slots,
                        "elapsed_seconds": elapsed,
                        "elapsed_milliseconds": elapsed * 1000.0,
                        "per_example_milliseconds": (
                            elapsed * 1000.0 / len(batch.indices)
                        ),
                    }
                )
    return measurements


def measure_ttft(
    model: Any,
    batches: Sequence[EncodedBatch],
    *,
    device: torch.device,
    generation_kwargs: dict[str, Any],
) -> dict[str, Any]:
    measurements = []
    contract = (
        "pretokenized synchronized model.generate(max_new_tokens=1), "
        "including first-token selection"
    )
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(batches):
                _cuda_sync(device)
                started = time.perf_counter()
                _generate_fixed_cap(
                    model,
                    batch,
                    max_new_tokens=1,
                    generation_kwargs=generation_kwargs,
                )
                _cuda_sync(device)
                elapsed = time.perf_counter() - started
                measurements.append(
                    {
                        "batch_index": batch_index,
                        "examples": len(batch.indices),
                        "elapsed_milliseconds": elapsed * 1000.0,
                        "per_example_milliseconds": (
                            elapsed * 1000.0 / len(batch.indices)
                        ),
                    }
                )
    except Exception as error:
        return {
            "status": "unsupported",
            "measurement_contract": contract,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "measurements": measurements,
            "summary": None,
        }
    return {
        "status": "pass",
        "measurement_contract": contract,
        "measurements": measurements,
        "summary": {
            "batch_milliseconds": summarize(
                [row["elapsed_milliseconds"] for row in measurements]
            ),
            "per_example_milliseconds": summarize(
                [row["per_example_milliseconds"] for row in measurements]
            ),
        },
    }


def _generation_settings(
    candidate: CandidateSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs, receipt = build_generation_compile_settings(
        cache_implementation=candidate.cache_implementation,
        disable_compile=candidate.generation_disable_compile,
        compile_dynamic=candidate.generation_compile_dynamic,
    )
    if candidate.cache_implementation != "dynamic":
        kwargs["cache_implementation"] = candidate.cache_implementation
    return kwargs, receipt


def resolve_generation_token_settings(
    model: Any,
    tokenizer: Any,
) -> tuple[set[int], int, Any]:
    """Mirror the standalone evaluator's EOS/padding resolution."""

    eos_value = getattr(
        getattr(model, "generation_config", None),
        "eos_token_id",
        None,
    )
    if eos_value is None:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    stop_ids = _special_token_ids(eos_value)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        if not stop_ids:
            raise ValueError("tokenizer needs a pad token or model/tokenizer EOS")
        pad_token_id = min(stop_ids)
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None and getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = pad_token_id
    return stop_ids, int(pad_token_id), eos_value


def load_candidate(
    candidate: CandidateSpec,
    *,
    bundle: Path | None,
    dense_model: str | None,
    dense_revision: str | None,
    dtype_name: str,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], float, float]:
    validate_candidate_spec(candidate)
    load_started = time.perf_counter()
    if candidate.model_kind == "physical":
        if bundle is None:
            raise UnsupportedCandidate("physical candidate requires --bundle")
        model, tokenizer, receipt = load_arithmetic_physical_bundle(
            bundle,
            device=str(device),
            attention_implementation=candidate.attention_implementation,
            mlp_implementation=candidate.mlp_implementation,
            activation_implementation=candidate.activation_implementation,
            hybrid_activation_threshold_rows=(
                candidate.hybrid_activation_threshold_rows
            ),
            width_alignment=candidate.width_alignment,
        )
    else:
        if dense_model is None:
            raise UnsupportedCandidate("dense candidate requires --dense-model")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            dense_model,
            revision=dense_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            dense_model,
            revision=dense_revision,
            dtype=DTYPES[dtype_name],
            attn_implementation=candidate.attention_implementation,
        ).to(device).eval()
        receipt = {
            "status": "pass",
            "kind": "dense_parent",
            "model": dense_model,
            "revision": dense_revision,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_dtype": str(DTYPES[dtype_name]),
            "donor_model_loaded": False,
        }
    if tokenizer is None:
        raise RuntimeError("candidate loader did not restore a tokenizer")
    tokenizer.padding_side = "left"
    _cuda_sync(device)
    load_seconds = time.perf_counter() - load_started

    compile_seconds = 0.0
    if candidate.outer_compile_mode != "none":
        compile_started = time.perf_counter()
        model = torch.compile(model, mode=candidate.outer_compile_mode)
        compile_seconds = time.perf_counter() - compile_started
    return model, tokenizer, receipt, load_seconds, compile_seconds


def _error_receipt(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback_tail": traceback.format_exc().splitlines()[-20:],
    }


def classify_error(error: BaseException) -> str:
    if isinstance(error, (UnsupportedCandidate, ImportError, ModuleNotFoundError)):
        return "unsupported"
    message = str(error).lower()
    unsupported_fragments = (
        "not available",
        "not installed",
        "unsupported",
        "requires flash",
        "requires triton",
        "no kernel image",
    )
    if any(fragment in message for fragment in unsupported_fragments):
        return "unsupported"
    return "failed"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def summarize_attempt(
    *,
    candidate: CandidateSpec,
    batch_size: int,
    records: Sequence[Mapping[str, Any]],
    batches: Sequence[EncodedBatch],
    warmups: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    batch_latency: list[dict[str, Any]],
    ttft: dict[str, Any],
    load_receipt: dict[str, Any],
    load_seconds: float,
    compile_seconds: float,
    after_load_memory: dict[str, int | None],
    floor_accuracy: float,
    floor_correct: int | None,
    contract: dict[str, Any],
) -> dict[str, Any]:
    correct_values = [int(row["correct"]) for row in measurements]
    conservative_correct = min(correct_values)
    gate = quality_gate(
        correct=conservative_correct,
        examples=len(records),
        floor_accuracy=floor_accuracy,
        floor_correct=floor_correct,
    )
    return {
        "status": "pass",
        "created_at": utc_now(),
        "candidate": asdict(candidate),
        "batch_size": batch_size,
        "contract": contract,
        "load_receipt": load_receipt,
        "setup_summary": {
            "load_seconds": load_seconds,
            "outer_compile_wrap_seconds": compile_seconds,
            "after_load_allocated_bytes": after_load_memory["allocated_bytes"],
            "after_load_reserved_bytes": after_load_memory["reserved_bytes"],
        },
        "encoded_batch_count": len(batches),
        "quality_gate": gate,
        "quality_stability": {
            "correct_values": correct_values,
            "prediction_sha256_values": [
                row["prediction_sha256"] for row in measurements
            ],
            "all_repeats_identical": len(
                {row["prediction_sha256"] for row in measurements}
            )
            == 1,
        },
        "warmup_measurements": warmups,
        "measurements": measurements,
        "summary": {
            key: summarize([float(row[key]) for row in measurements])
            for key in (
                "elapsed_seconds",
                "examples_per_second",
                "generated_tokens_per_second",
                "generated_token_slots_per_second",
                "correct_examples_per_second",
                "accuracy",
            )
        }
        | {
            key: summarize(
                [float(row[key]) for row in measurements if row[key] is not None]
            )
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")
            if any(row[key] is not None for row in measurements)
        },
        "batch_latency_measurements": batch_latency,
        "batch_latency_summary": {
            "batch_milliseconds": summarize(
                [row["elapsed_milliseconds"] for row in batch_latency]
            ),
            "per_example_milliseconds": summarize(
                [row["per_example_milliseconds"] for row in batch_latency]
            ),
        },
        "ttft": ttft,
        "generation_compile_observation": None,
    }


def failed_attempt(
    *,
    candidate: CandidateSpec,
    batch_size: int,
    contract: dict[str, Any],
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    return {
        "status": classify_error(error),
        "created_at": utc_now(),
        "candidate": asdict(candidate),
        "batch_size": batch_size,
        "contract": contract,
        "failure_stage": stage,
        "error": _error_receipt(error),
        "quality_gate": None,
    }


def select_fastest(attempts: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "pass"
        and attempt.get("quality_gate", {}).get("pass") is True
    ]
    if not eligible:
        return None
    winner = max(
        eligible,
        key=lambda attempt: attempt["summary"][
            "correct_examples_per_second"
        ]["median"],
    )
    return {
        "candidate": winner["candidate"]["name"],
        "role": winner["candidate"]["role"],
        "batch_size": winner["batch_size"],
        "median_correct_examples_per_second": winner["summary"][
            "correct_examples_per_second"
        ]["median"],
        "median_generated_tokens_per_second": winner["summary"][
            "generated_tokens_per_second"
        ]["median"],
        "accuracy": winner["quality_gate"]["accuracy"],
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    records = load_arithmetic_records(
        args.records,
        min_value=args.min_value,
        max_value=args.max_value,
        prompt_formats=args.prompt_formats,
        num_records=args.num_records,
        seed=args.seed,
        limit=args.limit,
    )
    workload_hash = records_sha256(records)
    run_id = str(uuid.uuid4())
    contract_base = {
        "run_id": run_id,
        "workload_sha256": workload_hash,
        "records_source": (
            str(args.records)
            if args.records
            else "generated_stratified_two_digit"
        ),
        "records_source_sha256": (
            sha256_file(args.records) if args.records is not None else None
        ),
        "examples": len(records),
        "min_value": args.min_value if args.records is None else None,
        "max_value": args.max_value if args.records is None else None,
        "num_records": args.num_records if args.records is None else None,
        "seed": args.seed if args.records is None else None,
        "prompt_formats": (
            list(args.prompt_formats) if args.records is None else None
        ),
        "record_limit": args.limit,
        "prompt_contract": (
            "generation_prompt_if_present_else_prompt_exact"
        ),
        "quality_metric": SCORER_CONTRACT["name"],
        "scorer": SCORER_CONTRACT,
        "isolation": "standalone_model_no_counterfactual_donor",
        "generation_mode": "fixed_cap_greedy_unguided",
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens_contract": (
            "non_stop_token_ids_after_eos_and_pad_trimming"
        ),
        "generated_token_slots_contract": (
            "returned_generation_suffix_slots_including_terminal_and_padding"
        ),
        "throughput_timing_contract": (
            "synchronized_pretokenized_generate_only_decode_scoring_excluded"
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_repeats": args.latency_repeats,
    }

    candidates = default_candidate_specs(
        include_dense=args.dense_model is not None,
        include_physical=args.bundle is not None,
        include_default_optimized=not args.no_default_optimized,
    )
    if args.candidate_config is not None:
        candidates.extend(load_candidate_config(args.candidate_config))
    if args.only_candidate:
        selected = set(args.only_candidate)
        candidates = [
            candidate for candidate in candidates if candidate.name in selected
        ]
        missing = sorted(selected - {candidate.name for candidate in candidates})
        if missing:
            raise ValueError(f"unknown --only-candidate values: {missing}")
    if not candidates:
        raise ValueError("no candidates were selected")
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        validate_candidate_spec(candidate)

    args.attempts_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.attempts_jsonl.write_text("")
    attempts: list[dict[str, Any]] = []
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("benchmark execution requires a CUDA device")
    torch.cuda.set_device(device)

    for candidate in candidates:
        torch.cuda.empty_cache()
        try:
            model, tokenizer, load_receipt, load_seconds, compile_seconds = (
                load_candidate(
                    candidate,
                    bundle=args.bundle,
                    dense_model=args.dense_model,
                    dense_revision=args.dense_revision,
                    dtype_name=args.dtype,
                    device=device,
                )
            )
            after_load_memory = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            generation_kwargs, generation_compile_receipt = (
                _generation_settings(candidate)
            )
            stop_ids, pad_token_id, eos_value = (
                resolve_generation_token_settings(model, tokenizer)
            )
            generation_kwargs["pad_token_id"] = pad_token_id
            if eos_value is not None:
                generation_kwargs["eos_token_id"] = eos_value
        except Exception as error:
            for batch_size in args.batch_sizes:
                attempt = failed_attempt(
                    candidate=candidate,
                    batch_size=batch_size,
                    contract=contract_base | {"batch_size": batch_size},
                    error=error,
                    stage="candidate_load",
                )
                attempts.append(attempt)
                append_jsonl(args.attempts_jsonl, attempt)
            continue

        for batch_size in args.batch_sizes:
            contract = contract_base | {
                "batch_size": batch_size,
                "generation_compile": generation_compile_receipt,
            }
            try:
                batches = prepare_batches(
                    records,
                    tokenizer,
                    batch_size=batch_size,
                    device=device,
                )
                warmups = [
                    run_generation_once(
                        model,
                        batches,
                        tokenizer=tokenizer,
                        device=device,
                        generation_kwargs=generation_kwargs,
                        max_new_tokens=args.max_new_tokens,
                        stop_ids=stop_ids,
                        pad_token_id=pad_token_id,
                    )
                    for _ in range(args.warmup)
                ]
                measurements = [
                    run_generation_once(
                        model,
                        batches,
                        tokenizer=tokenizer,
                        device=device,
                        generation_kwargs=generation_kwargs,
                        max_new_tokens=args.max_new_tokens,
                        stop_ids=stop_ids,
                        pad_token_id=pad_token_id,
                    )
                    for _ in range(args.repeats)
                ]
                latency = measure_batch_latency(
                    model,
                    batches,
                    device=device,
                    generation_kwargs=generation_kwargs,
                    max_new_tokens=args.max_new_tokens,
                    repeats=args.latency_repeats,
                )
                ttft = measure_ttft(
                    model,
                    batches,
                    device=device,
                    generation_kwargs=generation_kwargs,
                )
                attempt = summarize_attempt(
                    candidate=candidate,
                    batch_size=batch_size,
                    records=records,
                    batches=batches,
                    warmups=warmups,
                    measurements=measurements,
                    batch_latency=latency,
                    ttft=ttft,
                    load_receipt=load_receipt,
                    load_seconds=load_seconds,
                    compile_seconds=compile_seconds,
                    after_load_memory=after_load_memory,
                    floor_accuracy=args.quality_floor_accuracy,
                    floor_correct=args.quality_floor_correct,
                    contract=contract,
                )
                attempt["generation_compile_observation"] = (
                    observe_generation_compile_state(model)
                )
            except Exception as error:
                attempt = failed_attempt(
                    candidate=candidate,
                    batch_size=batch_size,
                    contract=contract,
                    error=error,
                    stage="batch_sweep",
                )
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            attempts.append(attempt)
            append_jsonl(args.attempts_jsonl, attempt)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    report = {
        "status": "pass"
        if any(attempt["status"] == "pass" for attempt in attempts)
        else "failed",
        "created_at": utc_now(),
        "run_id": run_id,
        "contract": contract_base,
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "candidate_specs": [asdict(candidate) for candidate in candidates],
        "batch_sizes": args.batch_sizes,
        "attempts_jsonl": str(args.attempts_jsonl),
        "attempts": attempts,
        "attempt_status_counts": {
            status: sum(attempt["status"] == status for attempt in attempts)
            for status in ("pass", "failed", "unsupported")
        },
        "fastest_quality_passing": select_fastest(attempts),
    }
    write_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--dense-model")
    parser.add_argument("--dense-revision")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts-jsonl", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--only-candidate", action="append")
    parser.add_argument("--no-default-optimized", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-value", type=int, default=10)
    parser.add_argument("--max-value", type=int, default=99)
    parser.add_argument("--num-records", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt-formats",
        nargs="+",
        choices=PROMPT_FORMATS,
        default=list(DEFAULT_PROMPT_FORMATS),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--quality-floor-accuracy", type=float, default=0.90)
    quality.add_argument("--quality-floor-correct", type=int)
    args = parser.parse_args()
    if args.attempts_jsonl is None:
        args.attempts_jsonl = args.output.with_suffix(".attempts.jsonl")
    if args.bundle is None and args.dense_model is None:
        parser.error("at least one of --bundle or --dense-model is required")
    if (
        args.records is None
        and not 10 <= args.min_value <= args.max_value <= 99
    ):
        parser.error(
            "generated records require 10 <= min-value <= max-value <= 99"
        )
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.latency_repeats <= 0
    ):
        parser.error("warmup must be nonnegative and repeats must be positive")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    if args.num_records <= 0:
        parser.error("--num-records must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0.0 <= args.quality_floor_accuracy <= 1.0:
        parser.error("--quality-floor-accuracy must be in [0, 1]")
    if args.quality_floor_correct is not None and args.quality_floor_correct < 0:
        parser.error("--quality-floor-correct must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    report = run_sweep(args)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
