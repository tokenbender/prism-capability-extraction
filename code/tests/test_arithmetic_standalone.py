from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_arithmetic_standalone import (  # noqa: E402
    DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    build_historical_pair_records,
    build_two_digit_records,
    generate_predictions,
    load_physical_evaluation_runtime,
    load_selected_mlp_mask,
    logical_zero_isolation,
    parse_historical_pair_specs,
    parse_exact_integer,
    score_prediction,
    sha256_file,
    summarize_predictions,
    validate_cli_contract,
    write_run_artifacts,
)


def test_record_builder_is_deterministic_unique_and_slice_complete() -> None:
    first = build_two_digit_records(
        min_value=10,
        max_value=99,
        prompt_formats=("compact", "spaced"),
        num_records=240,
        seed=123,
    )
    second = build_two_digit_records(
        min_value=10,
        max_value=99,
        prompt_formats=("compact", "spaced"),
        num_records=240,
        seed=123,
    )
    other_seed = build_two_digit_records(
        min_value=10,
        max_value=99,
        prompt_formats=("compact", "spaced"),
        num_records=240,
        seed=124,
    )

    assert first == second
    assert [row["id"] for row in first] != [row["id"] for row in other_seed]
    assert len({row["id"] for row in first}) == len(first)
    assert {row["prompt_format"] for row in first} == {"compact", "spaced"}
    assert {row["carry"] for row in first} == {"carry", "no_carry"}
    assert {row["result_length"] for row in first} == {2, 3}
    assert all(row["answer"] == row["a"] + row["b"] for row in first)
    assert all(10 <= row["a"] <= 99 and 10 <= row["b"] <= 99 for row in first)


def test_record_builder_rejects_invalid_or_overlarge_contracts() -> None:
    with pytest.raises(ValueError, match="two-digit operands"):
        build_two_digit_records(min_value=9)
    with pytest.raises(ValueError, match="prompt formats must be unique"):
        build_two_digit_records(prompt_formats=("compact", "compact"))
    with pytest.raises(ValueError, match="only 2 exist"):
        build_two_digit_records(
            min_value=10,
            max_value=10,
            prompt_formats=("compact", "spaced"),
            num_records=3,
        )


def _historical_pair_payload(offset: int, count: int = 4) -> dict[str, object]:
    pairs = []
    for index in range(count):
        a = 10 + offset + index
        b = 20 + index
        pairs.append(
            {
                "target": {
                    "prompt": f"{a} + {b} =",
                    "a": a,
                    "b": b,
                    "answer": str(a + b),
                },
                "cf": {
                    "prompt": f"{a} + {b + 1} =",
                    "a": a,
                    "b": b + 1,
                    "answer": str(a + b + 1),
                },
            }
        )
    return {"pairs": pairs}


def test_historical_records_take_final_targets_and_hash_every_source(
    tmp_path: Path,
) -> None:
    files = {}
    for offset, source in enumerate(("hundreds", "tens", "ones")):
        path = tmp_path / f"{source}_pairs.json"
        path.write_text(json.dumps(_historical_pair_payload(offset * 5)))
        files[source] = path

    records, receipt = build_historical_pair_records(
        files,
        final_n_per_source=2,
    )
    replay, replay_receipt = build_historical_pair_records(
        files,
        final_n_per_source=2,
    )

    assert records == replay
    assert receipt == replay_receipt
    assert len(records) == 6
    assert len({row["id"] for row in records}) == 6
    assert [row["source_position"] for row in records] == [
        "hundreds",
        "hundreds",
        "tens",
        "tens",
        "ones",
        "ones",
    ]
    assert [row["source_index"] for row in records] == [2, 3, 2, 3, 2, 3]
    for row in records:
        source_payload = json.loads(
            files[row["source_position"]].read_text()
        )
        target = source_payload["pairs"][row["source_index"]]["target"]
        assert row["prompt"] == target["prompt"]
        assert row["generation_prompt"] == target["prompt"] + " "
        assert row["a"] == target["a"]
        assert row["b"] == target["b"]
        assert row["answer_text"] == target["answer"]
        assert row["answer"] == int(target["answer"])
    for source, path in files.items():
        assert receipt["sources"][source]["sha256"] == sha256_file(path)
        assert receipt["sources"][source]["selected"] == 2
    assert receipt["num_records"] == 6

    specs = parse_historical_pair_specs(
        [f"{source}={path}" for source, path in reversed(list(files.items()))]
    )
    assert list(specs) == ["hundreds", "tens", "ones"]
    assert specs == files


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42),
        ("  42!  ", 42),
        ("\n42\nThe calculation is complete.", 42),
        ("-7.", -7),
        ("0042", 42),
        ("The answer is 42", None),
        ("42.0", None),
        ("42 + 1", None),
        ("", None),
    ],
)
def test_exact_integer_parser(text: str, expected: int | None) -> None:
    assert parse_exact_integer(text) == expected


def test_scoring_and_required_slices_use_numeric_equality() -> None:
    records = [
        {
            "id": "a",
            "a": 19,
            "b": 23,
            "answer": 42,
            "answer_text": "42",
            "prompt": "19 + 23 =",
            "prompt_format": "compact",
            "result_length": 2,
            "carry": "carry",
            "carry_class": "ones",
            "ones_carry": True,
            "leading_carry": False,
        },
        {
            "id": "b",
            "a": 60,
            "b": 50,
            "answer": 110,
            "answer_text": "110",
            "prompt": "60 + 50 = ",
            "prompt_format": "spaced",
            "result_length": 3,
            "carry": "no_carry",
            "carry_class": "leading",
            "ones_carry": False,
            "leading_carry": True,
        },
    ]
    predictions = [
        score_prediction(records[0], prediction_text="042", prediction_token_ids=[1]),
        score_prediction(records[1], prediction_text="111", prediction_token_ids=[2]),
    ]
    summary = summarize_predictions(predictions)

    assert predictions[0]["exact_numeric_correct"] is True
    assert predictions[1]["exact_numeric_correct"] is False
    assert summary["correct"] == 1
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["by_carry"]["carry"]["accuracy"] == 1.0
    assert summary["by_carry"]["no_carry"]["accuracy"] == 0.0
    assert summary["by_result_length"]["2"]["accuracy"] == 1.0
    assert summary["by_result_length"]["3"]["accuracy"] == 0.0
    assert summary["by_prompt_format"]["compact"]["n"] == 1
    assert summary["by_prompt_format"]["spaced"]["n"] == 1


class DummyMLP(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.down_proj = nn.Linear(width, 1, bias=False)
        self.down_proj.weight.data.fill_(1.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(hidden)


class DummyLayer(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.mlp = DummyMLP(width)


class DummyCore(nn.Module):
    def __init__(self, layers: int = 2, width: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(width) for _ in range(layers)])


class DummyIsolationModel(nn.Module):
    def __init__(self, layers: int = 2, width: int = 4) -> None:
        super().__init__()
        self.model = DummyCore(layers, width)


def test_logical_zero_isolation_zeros_unselected_and_removes_hooks() -> None:
    model = DummyIsolationModel(layers=2, width=4)
    selected = {
        0: torch.tensor([True, False, True, False]),
        1: torch.tensor([False, True, False, False]),
    }
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    dense = [layer.mlp(hidden).item() for layer in model.model.layers]
    with logical_zero_isolation(model, selected):
        isolated = [layer.mlp(hidden).item() for layer in model.model.layers]
    restored = [layer.mlp(hidden).item() for layer in model.model.layers]

    assert dense == [10.0, 10.0]
    assert isolated == [4.0, 2.0]
    assert restored == dense
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)


def test_mask_loader_is_strict_and_receipted(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.npz"
    np.savez_compressed(
        mask_path,
        mlp_final=np.array([[0, 1], [0, 3], [1, 0]], dtype=np.int32),
    )

    selected, receipt = load_selected_mlp_mask(mask_path, widths=[4, 2])

    assert selected[0].tolist() == [False, True, False, True]
    assert selected[1].tolist() == [True, False]
    assert receipt["kept_per_layer"] == {"0": 2, "1": 1}
    assert receipt["kept_total"] == 3
    assert receipt["available_total"] == 6
    assert receipt["kept_fraction"] == 0.5
    assert receipt["mask_sha256"] == sha256_file(mask_path)
    assert receipt["donor_activations"] is False

    duplicate_path = tmp_path / "duplicate.npz"
    np.savez_compressed(
        duplicate_path,
        mlp_final=np.array([[0, 1], [0, 1]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="duplicate mask pair"):
        load_selected_mlp_mask(duplicate_path, widths=[4])

    invalid_path = tmp_path / "invalid.npz"
    np.savez_compressed(
        invalid_path,
        mlp_final=np.array([[1, 0]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="outside"):
        load_selected_mlp_mask(invalid_path, widths=[4])


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    padding_side = "right"
    name_or_path = "fake-tokenizer"

    def __call__(
        self,
        prompts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert add_special_tokens is False
        assert padding is True
        assert return_tensors == "pt"
        widths = [len(prompt) % 3 + 1 for prompt in prompts]
        max_width = max(widths)
        rows = []
        masks = []
        for width in widths:
            rows.append([0] * (max_width - width) + [7] * width)
            masks.append([0] * (max_width - width) + [1] * width)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


class FakeTokenizerWithoutPad(FakeTokenizer):
    eos_token = "<eos>"
    pad_token = None
    pad_token_id = None


class FakeGenerationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(weight=self.anchor)

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        additions = torch.tensor(
            [[ord("4"), ord("2"), 99], [ord("1"), ord("1"), ord("0")]],
            dtype=torch.long,
        )
        return torch.cat([input_ids, additions], dim=1)


def test_generation_is_greedy_and_never_receives_gold_answer_length() -> None:
    records = [
        {
            "id": "short-gold",
            "prompt": "19 + 23 =",
            "answer": 42,
            "answer_text": "42",
            "a": 19,
            "b": 23,
            "prompt_format": "compact",
            "result_length": 2,
            "carry": "carry",
            "carry_class": "ones",
            "ones_carry": True,
            "leading_carry": False,
        },
        {
            "id": "long-gold",
            "prompt": "60 + 50 = ",
            "answer": 110,
            "answer_text": "110",
            "a": 60,
            "b": 50,
            "prompt_format": "spaced",
            "result_length": 3,
            "carry": "no_carry",
            "carry_class": "leading",
            "ones_carry": False,
            "leading_carry": True,
        },
    ]
    model = FakeGenerationModel()
    tokenizer = FakeTokenizer()
    predictions = generate_predictions(
        model,
        tokenizer,
        records,
        batch_size=2,
        max_new_tokens=8,
    )

    assert [row["prediction_text"] for row in predictions] == ["42", "110"]
    assert all(row["exact_numeric_correct"] for row in predictions)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["do_sample"] is False
    assert call["num_beams"] == 1
    assert call["max_new_tokens"] == 8
    assert "max_length" not in call
    assert "stopping_criteria" not in call
    assert not any("answer" in key or "gold" in key for key in call)
    assert tokenizer.padding_side == "left"


def test_historical_generation_uses_audited_boundary_not_gold_length() -> None:
    record = {
        "id": "historical-hundreds-000003",
        "source_position": "hundreds",
        "source_index": 3,
        "prompt": "19 + 23 =",
        "generation_prompt": "19 + 23 = ",
        "answer": 42,
        "answer_text": "42",
        "a": 19,
        "b": 23,
        "prompt_format": "historical_spaced",
        "result_length": 2,
        "carry": "carry",
        "carry_class": "ones",
        "ones_carry": True,
        "leading_carry": False,
    }

    class CapturingTokenizer(FakeTokenizer):
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
            self.prompts.extend(prompts)
            return super().__call__(prompts, **kwargs)

    class SingleRowModel(FakeGenerationModel):
        def generate(self, **kwargs: object) -> torch.Tensor:
            self.calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            addition = torch.tensor([[ord("4"), ord("2"), 99]])
            return torch.cat([input_ids, addition], dim=1)

    tokenizer = CapturingTokenizer()
    model = SingleRowModel()
    predictions = generate_predictions(
        model,
        tokenizer,
        [record],
        batch_size=1,
        max_new_tokens=8,
    )

    assert tokenizer.prompts == ["19 + 23 = "]
    assert predictions[0]["source_position"] == "hundreds"
    assert predictions[0]["exact_numeric_correct"] is True
    assert model.calls[0]["max_new_tokens"] == 8
    assert "max_length" not in model.calls[0]
    assert summarize_predictions(predictions)["by_source_position"][
        "hundreds"
    ] == {"correct": 1, "n": 1, "accuracy": 1.0}


def test_physical_runtime_feeds_the_same_prediction_and_scorer_path(
    tmp_path: Path,
) -> None:
    model = FakeGenerationModel()
    tokenizer = FakeTokenizer()
    calls = []

    def fake_loader(bundle: Path, **kwargs: object):
        calls.append((bundle, kwargs))
        return model, tokenizer, {
            "status": "pass",
            "format": "qwen_physical_mlp_substrate_v1",
            "layers": 2,
            "kept_total": 3,
            "kept_per_layer": [1, 2],
            "dense_mlp_allocated": False,
            "donor_model_loaded": False,
            "checkpoint_receipt": {"checkpoint_sha256": "abc"},
            "parameter_dtype": "torch.float32",
        }

    loaded_model, loaded_tokenizer, isolation = (
        load_physical_evaluation_runtime(
            tmp_path / "bundle",
            device="cpu",
            attention_implementation="eager",
            mlp_implementation="separate",
            activation_implementation="torch",
            hybrid_activation_threshold_rows=(
                DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
            ),
            width_alignment=1,
            allow_mlp_fallback=False,
            loader=fake_loader,
        )
    )
    records = [
        {
            "id": "physical-a",
            "prompt": "19 + 23 =",
            "answer": 42,
            "answer_text": "42",
            "a": 19,
            "b": 23,
            "prompt_format": "compact",
            "result_length": 2,
            "carry": "carry",
            "carry_class": "ones",
            "ones_carry": True,
            "leading_carry": False,
        },
        {
            "id": "physical-b",
            "prompt": "60 + 50 = ",
            "answer": 110,
            "answer_text": "110",
            "a": 60,
            "b": 50,
            "prompt_format": "spaced",
            "result_length": 3,
            "carry": "no_carry",
            "carry_class": "leading",
            "ones_carry": False,
            "leading_carry": True,
        },
    ]
    predictions = generate_predictions(
        loaded_model,
        loaded_tokenizer,
        records,
        batch_size=2,
        max_new_tokens=8,
    )

    assert all(row["exact_numeric_correct"] for row in predictions)
    assert isolation["mode"] == "physical_bundle"
    assert isolation["dense_mlp_allocated"] is False
    assert isolation["donor_activations"] is False
    assert isolation["checkpoint_receipt"]["checkpoint_sha256"] == "abc"
    assert calls[0][1]["restore_tokenizer"] is True


def _cli_namespace(**overrides: object) -> Namespace:
    values = {
        "mode": "dense",
        "bundle": None,
        "model": None,
        "model_revision": None,
        "tokenizer": None,
        "tokenizer_revision": None,
        "mask": None,
        "dtype": None,
        "mlp_implementation": "separate",
        "activation_implementation": "torch",
        "hybrid_activation_threshold_rows": (
            DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
        ),
        "width_alignment": 1,
        "allow_mlp_fallback": False,
        "historical_pair": [],
        "historical_final_n": 500,
        "min_value": 10,
        "max_value": 99,
        "num_records": 1_500,
        "seed": 123,
        "prompt_formats": ["compact", "spaced"],
    }
    values.update(overrides)
    return Namespace(**values)


def test_cli_contract_enforces_physical_and_historical_combinations() -> None:
    validate_cli_contract(
        _cli_namespace(mode="physical", bundle=Path("bundle"))
    )
    with pytest.raises(ValueError, match="requires --bundle"):
        validate_cli_contract(_cli_namespace(mode="physical"))
    with pytest.raises(ValueError, match="bundle-conflicting"):
        validate_cli_contract(
            _cli_namespace(
                mode="physical",
                bundle=Path("bundle"),
                mask=Path("mask.npz"),
            )
        )
    with pytest.raises(ValueError, match="forbids --bundle"):
        validate_cli_contract(
            _cli_namespace(mode="dense", bundle=Path("bundle"))
        )
    with pytest.raises(ValueError, match="requires --mask"):
        validate_cli_contract(_cli_namespace(mode="logical"))
    historical = [
        "hundreds=h.json",
        "tens=t.json",
        "ones=o.json",
    ]
    validate_cli_contract(_cli_namespace(historical_pair=historical))
    with pytest.raises(ValueError, match="generated-record controls"):
        validate_cli_contract(
            _cli_namespace(
                historical_pair=historical,
                num_records=12,
            )
        )


def test_generation_configures_tokenizer_padding_from_eos() -> None:
    record = {
        "id": "short-gold",
        "prompt": "19 + 23 =",
        "answer": 42,
        "answer_text": "42",
        "a": 19,
        "b": 23,
        "prompt_format": "compact",
        "result_length": 2,
        "carry": "carry",
        "carry_class": "ones",
        "ones_carry": True,
        "leading_carry": False,
    }
    tokenizer = FakeTokenizerWithoutPad()

    class SingleRowModel(FakeGenerationModel):
        def generate(self, **kwargs: object) -> torch.Tensor:
            self.calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            addition = torch.tensor([[ord("4"), ord("2"), 99]])
            return torch.cat([input_ids, addition], dim=1)

    predictions = generate_predictions(
        SingleRowModel(),
        tokenizer,
        [record],
        batch_size=1,
        max_new_tokens=8,
    )

    assert tokenizer.pad_token == tokenizer.eos_token
    assert predictions[0]["prediction_text"] == "42"


def test_artifact_writer_emits_verifiable_manifest_and_hashes(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "arith-19-23-compact",
            "a": 19,
            "b": 23,
            "answer": 42,
            "prompt": "19 + 23 =",
            "prompt_format": "compact",
            "result_length": 2,
            "carry": "carry",
            "carry_class": "ones",
        }
    ]
    predictions = [
        {
            **records[0],
            "gold_answer": 42,
            "prediction_text": "42",
            "prediction_token_ids": [4, 2],
            "parsed_answer": 42,
            "exact_numeric_correct": True,
        }
    ]
    completed = write_run_artifacts(
        tmp_path,
        records=records,
        predictions=predictions,
        manifest={"schema_version": "test", "status": "pass"},
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == completed
    assert manifest["artifacts"]["records"]["rows"] == 1
    assert manifest["artifacts"]["predictions"]["rows"] == 1
    sums = {}
    for line in (tmp_path / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        sums[filename] = digest
    assert sums == {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("manifest.json", "predictions.jsonl", "records.jsonl")
    }
