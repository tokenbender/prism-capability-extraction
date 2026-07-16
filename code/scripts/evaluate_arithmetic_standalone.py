#!/usr/bin/env python3
"""Evaluate dense, logically isolated, or physical arithmetic substrates.

This evaluator is deliberately independent of the historical counterfactual
patching scorer.  In ``logical`` mode, unselected MLP intermediate channels are
zeroed; no donor activation from a dense or counterfactual forward is supplied.
In ``physical`` mode, the strict arithmetic bundle loader reconstructs only the
retained per-layer MLP widths before entering the same generation/scoring path.
Generation is greedy and unguided: every row receives the same fixed
``max_new_tokens`` safety cap, irrespective of the gold answer's length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn


SCHEMA_VERSION = "prism_arithmetic_standalone_eval_v1"
MASK_KEY = "mlp_final"
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_PROMPT_FORMATS = ("compact", "spaced")
HISTORICAL_SOURCE_POSITIONS = ("hundreds", "tens", "ones")
DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS = 2_048
PROMPT_FORMATS = (
    "compact",
    "spaced",
    "equation_compact",
    "equation_compact_spaced",
    "words",
    "question",
    "sum",
)
SCORER_CONTRACT = {
    "name": "strict_first_line_integer_v1",
    "description": (
        "Take the first non-empty generated line, allow one terminal punctuation "
        "mark, require the rest of that line to be empty, parse it as a base-10 "
        "integer, and compare numeric equality with the gold sum."
    ),
}
_INTEGER_LINE = re.compile(r"^[+-]?\d+[.!?]?$")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used for content receipts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def render_prompt(a: int, b: int, prompt_format: str) -> str:
    """Render formats shared with the arithmetic training surface where possible.

    ``compact`` and ``spaced`` intentionally retain the historical names from
    ``train_lora_2digit_kl.py``.  The former has spaces inside the equation but
    no space after ``=``; the latter adds that final generation-boundary space.
    """

    canonical = f"{a} + {b} ="
    templates = {
        "compact": canonical,
        "spaced": canonical + " ",
        "equation_compact": f"{a}+{b}=",
        "equation_compact_spaced": f"{a}+{b}= ",
        "words": f"{a} plus {b} equals ",
        "question": f"What is {a} + {b}? ",
        "sum": f"Sum: {a} + {b} = ",
    }
    try:
        return templates[prompt_format]
    except KeyError as error:
        raise ValueError(
            f"unknown prompt format {prompt_format!r}; expected one of "
            f"{PROMPT_FORMATS}"
        ) from error


def carry_fields(a: int, b: int) -> dict[str, Any]:
    """Return explicit ones-column and leading-carry labels."""

    ones_carry = (a % 10) + (b % 10) >= 10
    leading_carry = a + b >= 100
    names = []
    if ones_carry:
        names.append("ones")
    if leading_carry:
        names.append("leading")
    return {
        "carry": "carry" if ones_carry else "no_carry",
        "carry_class": "_and_".join(names) if names else "none",
        "ones_carry": ones_carry,
        "leading_carry": leading_carry,
    }


def _stable_rank(seed: int, *parts: object) -> str:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _round_robin_strata(
    strata: Mapping[tuple[str, int, str], Sequence[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    offsets = {key: 0 for key in strata}
    keys = sorted(strata)
    while keys:
        remaining = []
        for key in keys:
            offset = offsets[key]
            rows = strata[key]
            if offset < len(rows):
                yield rows[offset]
                offsets[key] += 1
            if offsets[key] < len(rows):
                remaining.append(key)
        keys = remaining


def build_two_digit_records(
    *,
    min_value: int = 10,
    max_value: int = 99,
    prompt_formats: Sequence[str] = DEFAULT_PROMPT_FORMATS,
    num_records: int | None = 1_500,
    seed: int = 123,
) -> list[dict[str, Any]]:
    """Build a deterministic, slice-balanced two-digit addition evaluation set."""

    if min_value < 10 or max_value > 99 or min_value > max_value:
        raise ValueError(
            "two-digit operands require 10 <= min_value <= max_value <= 99"
        )
    if num_records is not None and num_records <= 0:
        raise ValueError("num_records must be positive or None")
    formats = tuple(prompt_formats)
    if not formats:
        raise ValueError("at least one prompt format is required")
    if len(set(formats)) != len(formats):
        raise ValueError("prompt formats must be unique")
    for prompt_format in formats:
        render_prompt(min_value, min_value, prompt_format)

    strata: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for prompt_format in formats:
        for a in range(min_value, max_value + 1):
            for b in range(min_value, max_value + 1):
                answer = a + b
                carry = carry_fields(a, b)
                row = {
                    "id": f"arith-{a:02d}-{b:02d}-{prompt_format}",
                    "a": a,
                    "b": b,
                    "answer": answer,
                    "answer_text": str(answer),
                    "prompt": render_prompt(a, b, prompt_format),
                    "prompt_format": prompt_format,
                    "result_length": len(str(answer)),
                    **carry,
                }
                stratum = (
                    str(row["carry"]),
                    int(row["result_length"]),
                    prompt_format,
                )
                strata[stratum].append(row)

    for key, rows in strata.items():
        rows.sort(
            key=lambda row: _stable_rank(
                seed,
                key,
                row["a"],
                row["b"],
                row["prompt_format"],
            )
        )

    ordered = list(_round_robin_strata(strata))
    if num_records is not None:
        if num_records > len(ordered):
            raise ValueError(
                f"requested {num_records} records but only {len(ordered)} exist"
            )
        ordered = ordered[:num_records]
    return ordered


def parse_historical_pair_specs(values: Sequence[str]) -> dict[str, Path]:
    """Parse the three explicit ``SOURCE_POSITION=PATH`` pair inputs."""

    if len(values) != len(HISTORICAL_SOURCE_POSITIONS):
        raise ValueError(
            "historical input requires exactly three --historical-pair values: "
            + ", ".join(HISTORICAL_SOURCE_POSITIONS)
        )
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"invalid historical pair spec {value!r}; expected SOURCE=PATH"
            )
        source_position, raw_path = value.split("=", 1)
        if source_position not in HISTORICAL_SOURCE_POSITIONS:
            raise ValueError(
                f"unknown historical source position {source_position!r}; "
                f"expected one of {HISTORICAL_SOURCE_POSITIONS}"
            )
        if source_position in parsed:
            raise ValueError(
                f"duplicate historical source position {source_position!r}"
            )
        if not raw_path:
            raise ValueError(f"empty path for historical source {source_position!r}")
        parsed[source_position] = Path(raw_path)
    missing = set(HISTORICAL_SOURCE_POSITIONS) - set(parsed)
    if missing:
        raise ValueError(f"missing historical source positions: {sorted(missing)}")
    return {
        source_position: parsed[source_position]
        for source_position in HISTORICAL_SOURCE_POSITIONS
    }


def build_historical_pair_records(
    pair_files: Mapping[str, Path],
    *,
    final_n_per_source: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze the final target rows from each audited position-pair file.

    ``prompt`` preserves the source JSON string byte-for-byte.  The historical
    full-answer evaluator appended one ASCII space at the generation boundary,
    so ``generation_prompt`` records that exact evaluated input separately.
    """

    if final_n_per_source <= 0:
        raise ValueError("final_n_per_source must be positive")
    if set(pair_files) != set(HISTORICAL_SOURCE_POSITIONS):
        raise ValueError(
            "historical pair files must cover exactly "
            f"{HISTORICAL_SOURCE_POSITIONS}"
        )

    records: list[dict[str, Any]] = []
    source_receipts: dict[str, Any] = {}
    for source_position in HISTORICAL_SOURCE_POSITIONS:
        path = Path(pair_files[source_position])
        payload = json.loads(path.read_text())
        pairs = payload.get("pairs")
        if not isinstance(pairs, list):
            raise TypeError(f"{path} must contain a top-level list named 'pairs'")
        if len(pairs) < final_n_per_source:
            raise ValueError(
                f"{path} has {len(pairs)} pairs, fewer than requested final "
                f"{final_n_per_source}"
            )
        first_source_index = len(pairs) - final_n_per_source
        for source_index in range(first_source_index, len(pairs)):
            pair = pairs[source_index]
            if not isinstance(pair, dict) or not isinstance(
                pair.get("target"), dict
            ):
                raise TypeError(
                    f"{path} pair index {source_index} has no target record"
                )
            target = pair["target"]
            try:
                prompt = target["prompt"]
                a = int(target["a"])
                b = int(target["b"])
                answer_text = str(target["answer"])
                answer = int(answer_text)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid target at {path} pair index {source_index}"
                ) from error
            if not isinstance(prompt, str):
                raise TypeError(
                    f"target prompt at {path} pair index {source_index} "
                    "must be a string"
                )
            if answer != a + b:
                raise ValueError(
                    f"target answer at {path} pair index {source_index} is "
                    f"{answer}, expected {a + b}"
                )
            stable_payload = {
                "source_position": source_position,
                "source_index": source_index,
                "prompt": prompt,
                "a": a,
                "b": b,
                "answer_text": answer_text,
            }
            carry = carry_fields(a, b)
            records.append(
                {
                    "id": (
                        f"historical-{source_position}-{source_index:06d}-"
                        f"{sha256_json(stable_payload)[:12]}"
                    ),
                    "source_position": source_position,
                    "source_index": source_index,
                    "prompt": prompt,
                    "generation_prompt": prompt + " ",
                    "prompt_format": "historical_spaced",
                    "a": a,
                    "b": b,
                    "answer": answer,
                    "answer_text": answer_text,
                    "result_length": len(answer_text),
                    **carry,
                }
            )
        source_receipts[source_position] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "total_pairs": len(pairs),
            "selection": "final_n_targets",
            "selected": final_n_per_source,
            "first_source_index": first_source_index,
            "last_source_index": len(pairs) - 1,
        }
    if len({str(record["id"]) for record in records}) != len(records):
        raise RuntimeError("historical record IDs are not unique")
    receipt = {
        "kind": "historical_position_pair_targets",
        "source_positions": list(HISTORICAL_SOURCE_POSITIONS),
        "records_per_source": final_n_per_source,
        "num_records": len(records),
        "selection": "final_n_target_records_per_source_in_canonical_order",
        "prompt_preservation": (
            "prompt is the exact source string; generation_prompt appends the "
            "single ASCII space used by the historical full-answer evaluator"
        ),
        "sources": source_receipts,
    }
    return records, receipt


def parse_exact_integer(text: str) -> int | None:
    """Parse the strict first-line numeric scorer contract."""

    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if not line or _INTEGER_LINE.fullmatch(line) is None:
        return None
    if line[-1:] in ".!?":
        line = line[:-1]
    try:
        return int(line, 10)
    except ValueError:
        return None


def score_prediction(
    record: Mapping[str, Any],
    *,
    prediction_text: str,
    prediction_token_ids: Sequence[int],
) -> dict[str, Any]:
    parsed = parse_exact_integer(prediction_text)
    gold = int(record["answer"])
    return {
        **dict(record),
        "gold_answer": gold,
        "prediction_text": prediction_text,
        "prediction_token_ids": [int(value) for value in prediction_token_ids],
        "parsed_answer": parsed,
        "exact_numeric_correct": parsed == gold,
    }


def _finish_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int | float]]:
    return {
        key: {
            "correct": int(value["correct"]),
            "n": int(value["n"]),
            "accuracy": int(value["correct"]) / int(value["n"]),
        }
        for key, value in sorted(counts.items())
        if value["n"]
    }


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot summarize an empty prediction set")
    correct = sum(bool(row["exact_numeric_correct"]) for row in predictions)
    slice_fields = [
        "carry",
        "carry_class",
        "result_length",
        "prompt_format",
    ]
    has_source_position = [
        "source_position" in row for row in predictions
    ]
    if any(has_source_position):
        if not all(has_source_position):
            raise ValueError(
                "source_position must be present on either every prediction or none"
            )
        slice_fields.append("source_position")
    slices: dict[str, Any] = {}
    for field in slice_fields:
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "n": 0}
        )
        for row in predictions:
            key = str(row[field])
            counts[key]["correct"] += int(bool(row["exact_numeric_correct"]))
            counts[key]["n"] += 1
        slices[f"by_{field}"] = _finish_counts(counts)
    return {
        "correct": correct,
        "n": len(predictions),
        "accuracy": correct / len(predictions),
        **slices,
    }


def decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Locate decoder layers through common Transformers/PEFT wrappers."""

    queue: list[Any] = [model]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        layers = getattr(current, "layers", None)
        if layers is not None:
            return layers
        for name in ("model", "base_model", "module"):
            child = getattr(current, name, None)
            if child is not None and child is not current:
                queue.append(child)
    raise AttributeError("could not locate decoder layers")


def mlp_widths(model: nn.Module) -> list[int]:
    widths = []
    for layer_index, layer in enumerate(decoder_layers(model)):
        try:
            width = int(layer.mlp.down_proj.in_features)
        except AttributeError as error:
            raise AttributeError(
                f"layer {layer_index} has no Qwen-style mlp.down_proj"
            ) from error
        if width <= 0:
            raise ValueError(f"layer {layer_index} has invalid MLP width {width}")
        widths.append(width)
    if not widths:
        raise ValueError("model has no decoder layers")
    return widths


def load_selected_mlp_mask(
    path: Path,
    *,
    widths: Sequence[int],
    key: str = MASK_KEY,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Load and strictly validate ``(layer, channel)`` selected-channel pairs."""

    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{key!r} missing from {path}")
        pairs = np.asarray(data[key])
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"{key!r} must have shape [n, 2], got {tuple(pairs.shape)}"
        )
    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError(f"{key!r} must contain integer layer/channel pairs")

    selected = {
        layer_index: torch.zeros(width, dtype=torch.bool)
        for layer_index, width in enumerate(widths)
    }
    seen: set[tuple[int, int]] = set()
    for raw_layer, raw_channel in pairs.tolist():
        layer_index = int(raw_layer)
        channel_index = int(raw_channel)
        pair = (layer_index, channel_index)
        if pair in seen:
            raise ValueError(f"duplicate mask pair {pair} in {path}")
        seen.add(pair)
        if not 0 <= layer_index < len(widths):
            raise ValueError(
                f"mask layer {layer_index} outside [0, {len(widths)})"
            )
        if not 0 <= channel_index < widths[layer_index]:
            raise ValueError(
                f"mask channel {channel_index} outside layer {layer_index} "
                f"width {widths[layer_index]}"
            )
        selected[layer_index][channel_index] = True
    if not seen:
        raise ValueError("logical isolation mask selects zero MLP channels")

    kept_per_layer = {
        str(layer): int(mask.sum().item()) for layer, mask in selected.items()
    }
    kept_total = sum(kept_per_layer.values())
    available_total = sum(int(width) for width in widths)
    receipt = {
        "mode": "logical_zero_selected_mlp",
        "mask_path": str(path),
        "mask_key": key,
        "mask_sha256": sha256_file(path),
        "layers": len(widths),
        "kept_per_layer": kept_per_layer,
        "kept_total": kept_total,
        "available_total": available_total,
        "kept_fraction": kept_total / available_total,
        "unselected_operation": "multiply_down_proj_input_by_zero",
        "donor_activations": False,
    }
    return selected, receipt


def validate_selected_mlp_mask(
    selected: Mapping[int, torch.Tensor],
    *,
    widths: Sequence[int],
) -> None:
    expected_layers = set(range(len(widths)))
    if set(selected) != expected_layers:
        raise ValueError(
            "logical mask must cover every decoder layer exactly; "
            f"expected {sorted(expected_layers)}, got {sorted(selected)}"
        )
    for layer_index, width in enumerate(widths):
        keep = selected[layer_index]
        if keep.dtype != torch.bool or keep.ndim != 1:
            raise TypeError(
                f"layer {layer_index} mask must be a one-dimensional bool tensor"
            )
        if int(keep.numel()) != int(width):
            raise ValueError(
                f"layer {layer_index} mask width {keep.numel()} != {width}"
            )


@contextmanager
def logical_zero_isolation(
    model: nn.Module,
    selected: Mapping[int, torch.Tensor],
) -> Iterator[None]:
    """Zero every unselected MLP channel for every forward in the context."""

    layers = decoder_layers(model)
    widths = mlp_widths(model)
    validate_selected_mlp_mask(selected, widths=widths)
    handles = []
    for layer_index, layer in enumerate(layers):
        keep_cpu = selected[layer_index].detach().clone()

        def zero_unselected(
            _module: nn.Module,
            hook_args: tuple[Any, ...],
            *,
            keep: torch.Tensor = keep_cpu,
        ) -> tuple[Any, ...]:
            if not hook_args:
                raise RuntimeError("down_proj pre-hook received no activation")
            activation = hook_args[0]
            if not isinstance(activation, torch.Tensor):
                raise TypeError("down_proj input must be a tensor")
            if activation.shape[-1] != keep.numel():
                raise ValueError(
                    f"down_proj activation width {activation.shape[-1]} "
                    f"!= mask width {keep.numel()}"
                )
            broadcast_shape = [1] * (activation.ndim - 1) + [keep.numel()]
            multiplier = keep.to(
                device=activation.device,
                dtype=activation.dtype,
            ).view(*broadcast_shape)
            return (activation * multiplier, *hook_args[1:])

        handles.append(
            layer.mlp.down_proj.register_forward_pre_hook(zero_unselected)
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _special_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _model_input_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


def _move_batch_to_device(
    encoded: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def _trim_generated_ids(
    token_ids: Sequence[int],
    *,
    stop_ids: set[int],
    pad_token_id: int | None,
) -> list[int]:
    trimmed = []
    for raw_id in token_ids:
        token_id = int(raw_id)
        if token_id in stop_ids or (
            pad_token_id is not None and token_id == pad_token_id
        ):
            break
        trimmed.append(token_id)
    return trimmed


def generate_predictions(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run fixed-cap greedy generation and score records after generation."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not records:
        raise ValueError("no records to evaluate")

    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
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

    device = _model_input_device(model)
    predictions = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
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
        encoded = _move_batch_to_device(encoded, device)
        input_width = int(encoded["input_ids"].shape[1])
        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": int(pad_token_id),
            "return_dict_in_generate": False,
        }
        if eos_value is not None:
            generation_kwargs["eos_token_id"] = eos_value
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)
        sequences = getattr(output, "sequences", output)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise TypeError("model.generate must return a rank-two token tensor")
        if int(sequences.shape[0]) != len(batch_records):
            raise ValueError("generation batch size does not match input records")
        generated = sequences[:, input_width:].detach().cpu().tolist()
        for record, token_ids in zip(batch_records, generated):
            trimmed = _trim_generated_ids(
                token_ids,
                stop_ids=stop_ids,
                pad_token_id=int(pad_token_id),
            )
            text = tokenizer.decode(trimmed, skip_special_tokens=True)
            predictions.append(
                score_prediction(
                    record,
                    prediction_text=text,
                    prediction_token_ids=trimmed,
                )
            )
    return predictions


def software_receipt() -> dict[str, Any]:
    versions = {"numpy": np.__version__, "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception as error:  # pragma: no cover - only possible in partial envs
        versions["transformers"] = f"unavailable: {error}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def model_receipt(model: nn.Module, tokenizer: Any, requested: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else None
    parameters = list(model.parameters())
    return {
        "requested": requested,
        "resolved_name_or_path": getattr(config, "_name_or_path", None),
        "resolved_commit": getattr(config, "_commit_hash", None),
        "config_sha256": sha256_json(config_dict) if config_dict is not None else None,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }


def load_physical_evaluation_runtime(
    bundle: Path,
    *,
    device: str,
    attention_implementation: str,
    mlp_implementation: str,
    activation_implementation: str,
    hybrid_activation_threshold_rows: int,
    width_alignment: int,
    allow_mlp_fallback: bool,
    loader: Any = None,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load a physical bundle and adapt its strict receipt to this evaluator."""

    if loader is None:
        from load_arithmetic_physical_bundle import (
            load_arithmetic_physical_bundle,
        )

        loader = load_arithmetic_physical_bundle
    model, tokenizer, load_receipt = loader(
        bundle,
        device=device,
        attention_implementation=attention_implementation,
        mlp_implementation=mlp_implementation,
        activation_implementation=activation_implementation,
        hybrid_activation_threshold_rows=hybrid_activation_threshold_rows,
        width_alignment=width_alignment,
        allow_mlp_fallback=allow_mlp_fallback,
        restore_tokenizer=True,
    )
    if tokenizer is None:
        raise RuntimeError("physical bundle loader did not restore its tokenizer")
    if load_receipt.get("status") != "pass":
        raise RuntimeError(
            f"physical bundle load did not pass: {load_receipt.get('status')!r}"
        )
    isolation = {
        "mode": "physical_bundle",
        "bundle": str(Path(bundle)),
        "format": load_receipt.get("format"),
        "layers": load_receipt.get("layers"),
        "kept_total": load_receipt.get("kept_total"),
        "kept_per_layer": load_receipt.get("kept_per_layer"),
        "dense_mlp_allocated": load_receipt.get("dense_mlp_allocated"),
        "donor_activations": bool(load_receipt.get("donor_model_loaded", False)),
        "checkpoint_receipt": load_receipt.get("checkpoint_receipt"),
        "loader_receipt": load_receipt,
    }
    if isolation["donor_activations"]:
        raise RuntimeError("physical evaluator refuses a loader with a donor model")
    return model, tokenizer, isolation


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_run_artifacts(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write JSONL evidence, a manifest, and GNU-compatible hash receipts."""

    records_path = output_dir / "records.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "manifest.json"
    sums_path = output_dir / "SHA256SUMS"

    write_jsonl(records_path, records)
    write_jsonl(predictions_path, predictions)
    completed = {
        **dict(manifest),
        "artifacts": {
            "records": {
                "path": records_path.name,
                "sha256": sha256_file(records_path),
                "rows": len(records),
                "canonical_content_sha256": sha256_json(list(records)),
            },
            "predictions": {
                "path": predictions_path.name,
                "sha256": sha256_file(predictions_path),
                "rows": len(predictions),
                "canonical_content_sha256": sha256_json(list(predictions)),
            },
        },
    }
    write_json(manifest_path, completed)
    sums = {
        records_path.name: sha256_file(records_path),
        predictions_path.name: sha256_file(predictions_path),
        manifest_path.name: sha256_file(manifest_path),
    }
    _atomic_write(
        sums_path,
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
    )
    return completed


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_cli_contract(args: argparse.Namespace) -> None:
    """Reject ambiguous model, isolation, and dataset input combinations."""

    if args.mode == "physical":
        if args.bundle is None:
            raise ValueError("--mode physical requires --bundle")
        forbidden = {
            "--model": args.model,
            "--model-revision": args.model_revision,
            "--tokenizer": args.tokenizer,
            "--tokenizer-revision": args.tokenizer_revision,
            "--mask": args.mask,
            "--dtype": args.dtype,
        }
        active = [name for name, value in forbidden.items() if value is not None]
        if active:
            raise ValueError(
                "--mode physical forbids bundle-conflicting arguments: "
                + ", ".join(active)
            )
    else:
        if args.bundle is not None:
            raise ValueError(f"--mode {args.mode} forbids --bundle")
        if args.mode == "logical" and args.mask is None:
            raise ValueError("--mode logical requires --mask")
        if args.mode == "dense" and args.mask is not None:
            raise ValueError("--mask is only valid with --mode logical")
        non_default_physical = (
            args.mlp_implementation != "separate"
            or args.activation_implementation != "torch"
            or args.hybrid_activation_threshold_rows
            != DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
            or args.width_alignment != 1
            or args.allow_mlp_fallback
        )
        if non_default_physical:
            raise ValueError(
                "physical runtime selectors require --mode physical"
            )

    if args.historical_pair:
        parse_historical_pair_specs(args.historical_pair)
        generated_overrides = (
            args.min_value != 10
            or args.max_value != 99
            or args.num_records != 1_500
            or args.seed != 123
            or tuple(args.prompt_formats) != DEFAULT_PROMPT_FORMATS
        )
        if generated_overrides:
            raise ValueError(
                "generated-record controls cannot be combined with "
                "--historical-pair"
            )
    elif args.historical_final_n != 500:
        raise ValueError(
            "--historical-final-n requires the three --historical-pair inputs"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dense", "logical", "physical"),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-key", default=MASK_KEY)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
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
    parser.add_argument(
        "--historical-pair",
        action="append",
        default=[],
        metavar="SOURCE_POSITION=PATH",
        help=(
            "repeat exactly three times for hundreds, tens, and ones; select "
            "the final target rows from each file"
        ),
    )
    parser.add_argument("--historical-final-n", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=("separate", "packed_gate_up"),
        default="separate",
    )
    parser.add_argument(
        "--activation-implementation",
        choices=("torch", "triton", "hybrid"),
        default="torch",
    )
    parser.add_argument(
        "--hybrid-activation-threshold-rows",
        type=int,
        default=DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    )
    parser.add_argument(
        "--width-alignment",
        type=int,
        choices=(1, 16, 64, 128, 256),
        default=1,
    )
    parser.add_argument("--allow-mlp-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_cli_contract(args)
    prepare_output_directory(args.output_dir, overwrite=args.overwrite)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.historical_pair:
        historical_files = parse_historical_pair_specs(args.historical_pair)
        records, dataset_receipt = build_historical_pair_records(
            historical_files,
            final_n_per_source=args.historical_final_n,
        )
    else:
        records = build_two_digit_records(
            min_value=args.min_value,
            max_value=args.max_value,
            prompt_formats=args.prompt_formats,
            num_records=args.num_records,
            seed=args.seed,
        )
        dataset_receipt = {
            "kind": "deterministic_stratified_two_digit_addition",
            "min_value": args.min_value,
            "max_value": args.max_value,
            "seed": args.seed,
            "num_records": len(records),
            "prompt_formats": list(args.prompt_formats),
            "selection": "stable_hash_with_round_robin_over_slice_strata",
        }

    device = pick_device(args.device)
    if args.mode == "physical":
        assert args.bundle is not None
        model, tokenizer, isolation_receipt = load_physical_evaluation_runtime(
            args.bundle,
            device=device,
            attention_implementation=args.attention_implementation,
            mlp_implementation=args.mlp_implementation,
            activation_implementation=args.activation_implementation,
            hybrid_activation_threshold_rows=(
                args.hybrid_activation_threshold_rows
            ),
            width_alignment=args.width_alignment,
            allow_mlp_fallback=args.allow_mlp_fallback,
        )
        isolation_context = nullcontext()
        requested_model = str(args.bundle)
        runtime_dtype = (
            isolation_receipt["loader_receipt"].get("parameter_dtype")
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        requested_model = args.model or DEFAULT_MODEL
        dtype_name = args.dtype or "bfloat16"
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
        tokenizer_name = args.tokenizer or requested_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=args.tokenizer_revision or args.model_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            requested_model,
            revision=args.model_revision,
            dtype=dtype,
            attn_implementation=args.attention_implementation,
        ).to(device)
        model.eval()
        widths = mlp_widths(model)
        if args.mode == "logical":
            assert args.mask is not None
            selected, isolation_receipt = load_selected_mlp_mask(
                args.mask,
                widths=widths,
                key=args.mask_key,
            )
            isolation_context = logical_zero_isolation(model, selected)
        else:
            isolation_receipt = {
                "mode": "dense",
                "layers": len(widths),
                "available_total": sum(widths),
                "donor_activations": False,
            }
            isolation_context = nullcontext()
        runtime_dtype = dtype_name

    started = time.perf_counter()
    with isolation_context:
        predictions = generate_predictions(
            model,
            tokenizer,
            records,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    elapsed = time.perf_counter() - started
    summary = summarize_predictions(predictions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "mode": args.mode,
            "dataset": dataset_receipt,
            "generation": {
                "algorithm": "unguided_greedy",
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
                "gold_answer_length_used": False,
                "stopping": "model_eos_or_fixed_global_token_cap",
                "batch_size": args.batch_size,
            },
            "scorer": SCORER_CONTRACT,
            "isolation": isolation_receipt,
        },
        "model": model_receipt(model, tokenizer, requested_model),
        "runtime": {
            "device": device,
            "dtype": runtime_dtype,
            "attention_implementation": args.attention_implementation,
            "generation_seconds": elapsed,
        },
        "summary": summary,
        "software": software_receipt(),
        "source": {
            "script": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    completed = write_run_artifacts(
        args.output_dir,
        records=records,
        predictions=predictions,
        manifest=manifest,
    )
    print(json.dumps(completed, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
