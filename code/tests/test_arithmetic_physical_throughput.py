from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_arithmetic_physical_throughput import (  # noqa: E402
    CandidateSpec,
    EncodedBatch,
    append_jsonl,
    classify_error,
    default_candidate_specs,
    load_arithmetic_records,
    percentile,
    prepare_batches,
    quality_gate,
    resolve_generation_token_settings,
    run_generation_once,
    score_generated_sequences,
    select_fastest,
    summarize,
    validate_candidate_spec,
)


def arithmetic_record(
    record_id: str,
    answer: int,
    *,
    prompt: str = "arithmetic =",
    generation_prompt: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": record_id,
        "prompt": prompt,
        "prompt_format": "test",
        "answer": answer,
        "answer_text": str(answer),
        "result_length": len(str(answer)),
        "carry": "unknown",
        "carry_class": "unknown",
        "ones_carry": None,
        "leading_carry": None,
    }
    if generation_prompt is not None:
        record["generation_prompt"] = generation_prompt
    return record


class DecodeTokenizer:
    def __init__(self, decoded: dict[tuple[int, ...], str]) -> None:
        self.decoded = decoded

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens
        return self.decoded[tuple(token_ids)]


def test_summary_records_p50_p95_and_dispersion() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]

    assert percentile(values, 0.95) == pytest.approx(8.8)
    result = summarize(values)
    assert result["median"] == 3.0
    assert result["p50"] == 3.0
    assert result["p95"] == pytest.approx(8.8)
    assert result["stdev"] > 0.0


def test_generated_records_match_standalone_balanced_builder_and_nested_inputs(
    tmp_path: Path,
) -> None:
    generated = load_arithmetic_records(
        None,
        min_value=10,
        max_value=11,
        prompt_formats=("compact",),
        num_records=4,
        seed=123,
    )
    regenerated = load_arithmetic_records(
        None,
        min_value=10,
        max_value=11,
        prompt_formats=("compact",),
        num_records=4,
        seed=123,
    )
    assert generated == regenerated
    assert len(generated) == 4
    assert {row["id"] for row in generated} == {
        "arith-10-10-compact",
        "arith-10-11-compact",
        "arith-11-10-compact",
        "arith-11-11-compact",
    }
    assert all(row["prompt"].endswith("=") for row in generated)
    assert all(not row["prompt"].endswith("= ") for row in generated)

    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "id": "pair-1",
                        "target": {
                            "prompt": "12 + 13 =",
                            "generation_prompt": "12 + 13 = ",
                            "answer": 25,
                        },
                    }
                ]
            }
        )
    )
    loaded = load_arithmetic_records(path)
    assert loaded[0]["id"] == "pair-1"
    assert loaded[0]["prompt"] == "12 + 13 ="
    assert loaded[0]["generation_prompt"] == "12 + 13 = "
    assert loaded[0]["answer"] == 25


def test_jsonl_records_are_normalized_and_limited(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"pair-{index}",
                    "prompt": f"{index + 10} + 10 =",
                    "generation_prompt": f"{index + 10} + 10 = ",
                    "answer": index + 20,
                }
            )
            for index in range(2)
        )
        + "\n"
    )

    loaded = load_arithmetic_records(path, limit=1)

    assert loaded == [
        {
            "id": "pair-0",
            "prompt": "10 + 10 =",
            "generation_prompt": "10 + 10 = ",
            "prompt_format": "custom",
            "answer": 20,
            "answer_text": "20",
            "result_length": 2,
            "carry": "unknown",
            "carry_class": "unknown",
            "ones_carry": None,
            "leading_carry": None,
        }
    ]


def test_prepare_batches_preserves_prompt_boundary_and_input_order() -> None:
    records = [
        arithmetic_record("two-digit", 25, prompt="12 + 13 ="),
        arithmetic_record(
            "three-digit",
            105,
            prompt="52 + 53 =",
            generation_prompt="52 + 53 = ",
        ),
    ]

    class CaptureTokenizer:
        def __init__(self) -> None:
            self.prompts: list[list[str]] = []

        def __call__(self, prompts: list[str], **_: Any) -> dict[str, Any]:
            self.prompts.append(prompts)
            return {
                "input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
                "attention_mask": torch.ones(
                    (len(prompts), 3),
                    dtype=torch.long,
                ),
            }

    tokenizer = CaptureTokenizer()
    batches = prepare_batches(
        records,
        tokenizer,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert tokenizer.prompts == [["12 + 13 =", "52 + 53 = "]]
    assert batches[0].indices == [0, 1]
    assert [row["id"] for row in batches[0].records] == [
        "two-digit",
        "three-digit",
    ]


def test_default_matrix_covers_required_roles_and_runtime_diversity() -> None:
    candidates = default_candidate_specs(
        include_dense=True,
        include_physical=True,
    )

    assert {candidate.role for candidate in candidates} == {
        "dense_parent",
        "canonical_physical",
        "optimized_physical",
    }
    assert any(
        candidate.cache_implementation == "static" for candidate in candidates
    )
    assert any(
        candidate.activation_implementation == "hybrid"
        for candidate in candidates
    )
    assert any(
        candidate.mlp_implementation == "packed_gate_up"
        for candidate in candidates
    )
    for candidate in candidates:
        validate_candidate_spec(candidate)


def test_candidate_validation_rejects_ambiguous_compile_and_invalid_repacking() -> None:
    with pytest.raises(ValueError, match="both own generation"):
        validate_candidate_spec(
            CandidateSpec(
                name="double-compile",
                role="optimized_physical",
                model_kind="physical",
                outer_compile_mode="reduce-overhead",
                cache_implementation="static",
            )
        )

    with pytest.raises(ValueError, match="physical MLP repacking"):
        validate_candidate_spec(
            CandidateSpec(
                name="bad-dense",
                role="dense_parent",
                model_kind="dense",
                mlp_implementation="packed_gate_up",
            )
        )


def test_strict_decoded_scoring_trims_eos_and_counts_tokens_honestly() -> None:
    records = [
        arithmetic_record("correct", 25),
        arithmetic_record("wrong", 27),
    ]
    batch = EncodedBatch(
        indices=[0, 1],
        records=records,
        encoded={"input_ids": torch.ones((2, 2), dtype=torch.long)},
    )
    sequences = torch.tensor(
        [
            [10, 11, 2, 5, 99, 0],
            [10, 11, 2, 6, 99, 0],
        ]
    )

    predictions, generated_tokens, generated_slots = (
        score_generated_sequences(
            sequences,
            prompt_width=2,
            batch=batch,
            tokenizer=DecodeTokenizer(
                {
                    (2, 5): "25",
                    (2, 6): "26",
                }
            ),
            stop_ids={99},
            pad_token_id=0,
        )
    )

    assert [row["exact_numeric_correct"] for _, row in predictions] == [
        True,
        False,
    ]
    assert generated_tokens == 4
    assert generated_slots == 8


def test_generation_uses_one_fixed_cap_without_gold_length_guidance() -> None:
    records = [
        arithmetic_record("two-digit", 25),
        arithmetic_record("three-digit", 105),
    ]
    batch = EncodedBatch(
        indices=[0, 1],
        records=records,
        encoded={
            "input_ids": torch.tensor([[10, 11], [10, 11]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        },
    )

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(self, **kwargs: Any) -> torch.Tensor:
            self.calls.append(kwargs)
            suffix = torch.tensor(
                [
                    [25, 99, 0, 0],
                    [105, 99, 0, 0],
                ]
            )
            return torch.cat((kwargs["input_ids"], suffix), dim=1)

    model = FakeModel()
    result = run_generation_once(
        model,
        [batch],
        tokenizer=DecodeTokenizer({(25,): "25", (105,): "105"}),
        device=torch.device("cpu"),
        generation_kwargs={"pad_token_id": 0, "eos_token_id": 99},
        max_new_tokens=8,
        stop_ids={99},
        pad_token_id=0,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["max_new_tokens"] == 8
    assert "min_new_tokens" not in model.calls[0]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert model.calls[0]["use_cache"] is True
    assert result["examples"] == 2
    assert result["correct"] == 2
    assert result["accuracy"] == 1.0
    assert result["generated_tokens"] == 2
    assert result["generated_token_slots"] == 8
    assert result["examples_per_second"] > 0.0
    assert result["generated_tokens_per_second"] > 0.0
    assert result["generated_token_slots_per_second"] > 0.0
    assert result["correct_examples_per_second"] > 0.0
    assert result["peak_allocated_bytes"] is None


def test_quality_floor_and_generation_token_resolution() -> None:
    assert quality_gate(
        correct=91,
        examples=100,
        floor_accuracy=0.90,
        floor_correct=None,
    )["pass"]
    assert not quality_gate(
        correct=89,
        examples=100,
        floor_accuracy=0.90,
        floor_correct=None,
    )["pass"]
    assert quality_gate(
        correct=90,
        examples=100,
        floor_accuracy=0.99,
        floor_correct=90,
    )["pass"]

    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[99, 100])
    )
    tokenizer = SimpleNamespace(
        pad_token_id=None,
        pad_token=None,
        eos_token=None,
        eos_token_id=None,
    )
    stop_ids, pad_token_id, eos_value = resolve_generation_token_settings(
        model,
        tokenizer,
    )
    assert stop_ids == {99, 100}
    assert pad_token_id == 99
    assert tokenizer.pad_token_id == 99
    assert eos_value == [99, 100]


def test_fastest_selection_excludes_quality_failures_and_failed_candidates() -> None:
    attempts = [
        {
            "status": "pass",
            "candidate": {"name": "bad-fast", "role": "optimized_physical"},
            "batch_size": 64,
            "quality_gate": {"pass": False, "accuracy": 0.2},
            "summary": {
                "examples_per_second": {"median": 2000.0},
                "correct_examples_per_second": {"median": 1000.0},
                "generated_tokens_per_second": {"median": 2000.0},
            },
        },
        {
            "status": "failed",
            "candidate": {"name": "failed", "role": "optimized_physical"},
            "batch_size": 64,
            "quality_gate": None,
        },
        {
            "status": "pass",
            "candidate": {"name": "winner", "role": "optimized_physical"},
            "batch_size": 32,
            "quality_gate": {"pass": True, "accuracy": 0.91},
            "summary": {
                "examples_per_second": {"median": 35.0},
                "correct_examples_per_second": {"median": 30.0},
                "generated_tokens_per_second": {"median": 70.0},
            },
        },
        {
            "status": "pass",
            "candidate": {"name": "more-correct-slower", "role": "dense_parent"},
            "batch_size": 64,
            "quality_gate": {"pass": True, "accuracy": 0.99},
            "summary": {
                "examples_per_second": {"median": 34.0},
                "correct_examples_per_second": {"median": 33.0},
                "generated_tokens_per_second": {"median": 68.0},
            },
        },
    ]

    winner = select_fastest(attempts)
    assert winner is not None
    assert winner["candidate"] == "winner"
    assert winner["batch_size"] == 32
    assert winner["median_examples_per_second"] == 35.0


def test_failed_candidates_are_classified_and_jsonl_is_incremental(
    tmp_path: Path,
) -> None:
    assert classify_error(ModuleNotFoundError("flash_attn")) == "unsupported"
    assert classify_error(RuntimeError("CUDA out of memory")) == "failed"

    path = tmp_path / "attempts.jsonl"
    append_jsonl(path, {"status": "unsupported", "name": "flash"})
    append_jsonl(path, {"status": "failed", "name": "oom"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"status": "unsupported", "name": "flash"},
        {"status": "failed", "name": "oom"},
    ]
