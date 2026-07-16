from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from verify_arithmetic_merge_provenance import (  # noqa: E402
    apply_lora_scale,
    capture_lora_scaling,
    compare_state_dicts,
    describe_source,
    local_file_manifest,
    scaling_receipt,
)


class ScalingLeaf(nn.Module):
    def __init__(self, scaling: dict[str, float]):
        super().__init__()
        self.scaling = dict(scaling)


class ScalingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ScalingLeaf({"default": 2.0})
        self.block = nn.Module()
        self.block.second = ScalingLeaf({"a": 0.5, "b": 4.0})


def test_captured_scaling_multiplier_matches_historical_semantics() -> None:
    model = ScalingModel()
    captured = capture_lora_scaling(model)
    receipt = scaling_receipt(captured, 0.55)
    apply_lora_scale(captured, 0.55)

    assert model.first.scaling == {"default": 1.1}
    assert model.block.second.scaling == {"a": 0.275, "b": 2.2}
    assert receipt["module_count"] == 2
    assert receipt["entry_count"] == 3
    assert len(receipt["canonical_sha256"]) == 64

    # Applying a different multiplier still uses the immutable captured values,
    # not values already mutated by the previous application.
    apply_lora_scale(captured, 0.25)
    assert model.first.scaling == {"default": 0.5}
    assert model.block.second.scaling == {"a": 0.125, "b": 1.0}


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -0.1])
def test_scaling_rejects_non_finite_or_negative_values(scale: float) -> None:
    captured = capture_lora_scaling(ScalingModel())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        apply_lora_scale(captured, scale)


def test_identical_states_are_the_only_hard_pass() -> None:
    reconstructed = {
        "float": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "integer": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    candidate = {key: value.clone() for key, value in reconstructed.items()}
    receipt = compare_state_dicts(reconstructed, candidate)

    assert receipt["status"] == "pass"
    assert receipt["exact_tensor_identity"] is True
    assert receipt["missing_key_count"] == 0
    assert receipt["shape_mismatch_count"] == 0
    assert receipt["dtype_mismatch_count"] == 0
    assert receipt["tensor_mismatch_count"] == 0
    assert receipt["mismatched_element_count"] == 0
    assert receipt["global_max_abs_error"] == 0.0


def test_value_mismatch_reports_exact_error_count_and_max_abs() -> None:
    reconstructed = {
        "weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }
    candidate = {
        "weight": torch.tensor([1.0, 2.5, -1.0], dtype=torch.float32),
    }
    receipt = compare_state_dicts(reconstructed, candidate)

    assert receipt["status"] == "fail"
    assert receipt["exact_tensor_identity"] is False
    assert receipt["tensor_mismatch_count"] == 1
    assert receipt["mismatched_element_count"] == 2
    assert receipt["global_max_abs_error"] == 4.0
    assert receipt["tensor_mismatches"] == [
        {
            "key": "weight",
            "shape": [3],
            "dtype": "torch.float32",
            "error_count": 2,
            "max_abs_error": 4.0,
        }
    ]
    assert receipt["mismatch_samples"][0]["kind"] == "tensor"


def test_key_shape_and_dtype_mismatches_are_separate_hard_failures() -> None:
    reconstructed = {
        "missing": torch.ones(1),
        "shape": torch.ones(2),
        "dtype": torch.ones(2, dtype=torch.float32),
    }
    candidate = {
        "unexpected": torch.ones(1),
        "shape": torch.ones(3),
        "dtype": torch.ones(2, dtype=torch.float16),
    }
    receipt = compare_state_dicts(
        reconstructed,
        candidate,
        mismatch_sample_limit=3,
    )

    assert receipt["status"] == "fail"
    assert receipt["missing_keys"] == ["missing"]
    assert receipt["unexpected_keys"] == ["unexpected"]
    assert receipt["shape_mismatch_count"] == 1
    assert receipt["dtype_mismatch_count"] == 1
    assert receipt["tensor_mismatch_count"] == 0
    assert len(receipt["mismatch_samples"]) == 3


def test_local_source_is_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "adapter"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"r": 32}\n')
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "run.log").write_text("local provenance")

    first = local_file_manifest(source)
    described = describe_source(str(source), requested_revision="local-note")
    assert first["file_count"] == 3
    assert described["kind"] == "local"
    assert described["requested_revision"] == "local-note"
    assert described["resolved_revision"] == first["manifest_sha256"]

    (source / "adapter_config.json").write_text('{"r": 16}\n')
    second = local_file_manifest(source)
    assert second["manifest_sha256"] != first["manifest_sha256"]


def test_remote_source_requires_an_explicit_revision() -> None:
    with pytest.raises(ValueError, match="requires an explicit revision"):
        describe_source("owner/not-a-local-path", requested_revision=None)
