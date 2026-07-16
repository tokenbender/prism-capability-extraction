from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from load_bfcl_physical_bundle import (  # noqa: E402
    PackedPhysicalQwenMLP,
    PhysicalQwenMLP,
    configure_mlp_runtime,
    install_physical_mlps,
    reference_parity,
    validate_packed_mlp,
)


class DummyMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DummyMLP()


class DummyDecoder(nn.Module):
    def __init__(self, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer() for _ in range(layers)])


class DummyModel(nn.Module):
    def __init__(self, layers: int = 2):
        super().__init__()
        self.model = DummyDecoder(layers)
        self.config = type("Config", (), {"intermediate_size": 6})()


def test_physical_mlp_matches_dense_keep_only_formula() -> None:
    torch.manual_seed(7)
    dense = DummyMLP()
    keep = torch.tensor([0, 2, 5])
    physical = PhysicalQwenMLP(dense, len(keep))
    with torch.no_grad():
        physical.gate_proj.weight.copy_(dense.gate_proj.weight[keep])
        physical.up_proj.weight.copy_(dense.up_proj.weight[keep])
        physical.down_proj.weight.copy_(dense.down_proj.weight[:, keep])

    inputs = torch.randn(2, 3, 4)
    hidden = dense.act_fn(dense.gate_proj(inputs)) * dense.up_proj(inputs)
    mask = torch.zeros_like(hidden)
    mask[..., keep] = hidden[..., keep]
    expected = dense.down_proj(mask)
    torch.testing.assert_close(physical(inputs), expected)


@pytest.mark.parametrize("width_alignment", [1, 16, 64, 128, 256])
def test_packed_mlp_is_lossless_and_zero_padded(width_alignment: int) -> None:
    torch.manual_seed(11)
    dense = DummyMLP()
    physical = PhysicalQwenMLP(dense, intermediate_size=3)
    with torch.no_grad():
        physical.gate_proj.weight.copy_(dense.gate_proj.weight[:3])
        physical.up_proj.weight.copy_(dense.up_proj.weight[:3])
        physical.down_proj.weight.copy_(dense.down_proj.weight[:, :3])

    packed = PackedPhysicalQwenMLP(physical, width_alignment)
    validation = validate_packed_mlp(physical, packed)
    assert validation == {
        "status": "pass",
        "active_width": 3,
        "aligned_width": max(3, width_alignment),
        "padding_channels": max(3, width_alignment) - 3,
    }

    inputs = torch.randn(2, 3, 4)
    torch.testing.assert_close(
        packed(inputs), physical(inputs), rtol=1e-6, atol=1e-7
    )


def test_runtime_repack_is_atomic_and_receipted() -> None:
    torch.manual_seed(13)
    model = DummyModel()
    metadata = {
        "isolation": {
            "kept_total": 7,
            "kept_per_layer": {"0": 2, "1": 5},
        }
    }
    widths = install_physical_mlps(model, metadata)
    inputs = torch.randn(2, 3, 4)
    expected = [layer.mlp(inputs) for layer in model.model.layers]

    receipt = configure_mlp_runtime(
        model,
        widths,
        implementation="packed_gate_up",
        width_alignment=16,
    )

    assert receipt["active_implementation"] == "packed_gate_up"
    assert receipt["validation"] == {
        "status": "pass",
        "method": "exact_active_tensors_and_zero_padding",
        "layers": 2,
    }
    assert receipt["active_channels"] == 7
    assert receipt["runtime_channels"] == 32
    assert receipt["padding_channels"] == 25
    assert receipt["aligned_per_layer"] == [16, 16]
    assert all(
        isinstance(layer.mlp, PackedPhysicalQwenMLP) for layer in model.model.layers
    )
    for layer, expected_output in zip(model.model.layers, expected):
        torch.testing.assert_close(
            layer.mlp(inputs), expected_output, rtol=1e-6, atol=1e-7
        )


def test_runtime_repack_can_explicitly_fallback_without_partial_mutation() -> None:
    model = DummyModel()
    metadata = {
        "isolation": {
            "kept_total": 7,
            "kept_per_layer": {"0": 2, "1": 5},
        }
    }
    widths = install_physical_mlps(model, metadata)
    first = model.model.layers[0].mlp
    model.model.layers[1].mlp = DummyMLP()
    second = model.model.layers[1].mlp

    receipt = configure_mlp_runtime(
        model,
        widths,
        implementation="packed_gate_up",
        width_alignment=16,
        allow_fallback=True,
    )

    assert receipt["active_implementation"] == "separate"
    assert receipt["fallback_used"] is True
    assert receipt["validation"] == {"status": "fail"}
    assert "expected PhysicalQwenMLP" in receipt["fallback_reason"]
    assert model.model.layers[0].mlp is first
    assert model.model.layers[1].mlp is second


def test_separate_runtime_rejects_misleading_alignment() -> None:
    model = DummyModel(layers=1)
    with pytest.raises(ValueError, match="only meaningful for packed_gate_up"):
        configure_mlp_runtime(
            model,
            [3],
            implementation="separate",
            width_alignment=16,
        )


def test_runtime_selector_rejects_an_unavailable_implementation() -> None:
    model = DummyModel(layers=1)

    with pytest.raises(ValueError, match="unsupported MLP implementation"):
        configure_mlp_runtime(
            model,
            [3],
            implementation="flashinfer",
            width_alignment=1,
        )


def test_install_physical_mlps_uses_every_recorded_width() -> None:
    model = DummyModel()
    metadata = {
        "isolation": {
            "kept_total": 7,
            "kept_per_layer": {"0": 2, "1": 5},
        }
    }
    assert install_physical_mlps(model, metadata) == [2, 5]
    assert model.model.layers[0].mlp.intermediate_size == 2
    assert model.model.layers[1].mlp.intermediate_size == 5


def test_reference_parity_is_prediction_boundary_exact(tmp_path: Path) -> None:
    generated = [
        {
            "id": "a",
            "prediction_text": "x",
            "prediction_calls": [{"name": "f", "arguments": {}}],
            "raw_correct": True,
            "normalized_correct": True,
        }
    ]
    reference = tmp_path / "reference.jsonl"
    reference.write_text(json.dumps(generated[0]) + "\n")
    assert reference_parity(generated, reference)["status"] == "pass"

    generated[0]["prediction_text"] = "y"
    receipt = reference_parity(generated, reference)
    assert receipt["status"] == "fail"
    assert receipt["field_diff_counts"]["prediction_text"] == 1
