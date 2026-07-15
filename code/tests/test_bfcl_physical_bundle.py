from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from load_bfcl_physical_bundle import (  # noqa: E402
    PhysicalQwenMLP,
    install_physical_mlps,
    reference_parity,
)


class DummyMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu


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


def test_install_physical_mlps_uses_every_recorded_width() -> None:
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = DummyMLP()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()
            self.config = type("Config", (), {"intermediate_size": 6})()

    model = Model()
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
