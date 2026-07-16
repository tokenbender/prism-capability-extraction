from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_arithmetic_physical_bundle import (  # noqa: E402
    build_physical_state_dict,
    load_mask_indices,
    parameter_accounting,
    write_physical_bundle,
)
from load_arithmetic_physical_bundle import (  # noqa: E402
    PackedPhysicalQwenMLP,
    PhysicalQwenMLP,
    configure_mlp_runtime,
    load_arithmetic_physical_bundle,
)


class DummyMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


def test_mask_loader_canonicalizes_and_rejects_invalid_masks(
    tmp_path: Path,
) -> None:
    mask = tmp_path / "mask.npz"
    np.savez(
        mask,
        mlp_final=np.asarray([[1, 4], [0, 3], [1, 1], [0, 0]], dtype=np.int32),
    )
    selected = load_mask_indices(mask, num_layers=2, dense_width=6)
    assert [values.tolist() for values in selected] == [[0, 3], [1, 4]]

    duplicate = tmp_path / "duplicate.npz"
    np.savez(
        duplicate,
        mlp_final=np.asarray([[0, 1], [0, 1], [1, 2]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="duplicate mask selection"):
        load_mask_indices(duplicate, num_layers=2, dense_width=6)

    empty_layer = tmp_path / "empty.npz"
    np.savez(
        empty_layer,
        mlp_final=np.asarray([[0, 1]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="empty layers"):
        load_mask_indices(empty_layer, num_layers=2, dense_width=6)


def test_state_builder_slices_rows_and_matching_columns() -> None:
    hidden = 4
    width = 6
    dense = {
        "model.embed_tokens.weight": torch.arange(40).reshape(10, hidden),
    }
    for layer in range(2):
        offset = layer * 1000
        dense[f"model.layers.{layer}.mlp.gate_proj.weight"] = (
            torch.arange(width * hidden).reshape(width, hidden) + offset
        )
        dense[f"model.layers.{layer}.mlp.up_proj.weight"] = (
            torch.arange(width * hidden).reshape(width, hidden) + offset + 100
        )
        dense[f"model.layers.{layer}.mlp.down_proj.weight"] = (
            torch.arange(hidden * width).reshape(hidden, width) + offset + 200
        )
    selected = [torch.tensor([0, 3]), torch.tensor([1, 4, 5])]
    physical, _ = build_physical_state_dict(dense, selected)

    torch.testing.assert_close(
        physical["model.layers.0.mlp.gate_proj.weight"],
        dense["model.layers.0.mlp.gate_proj.weight"][[0, 3]],
    )
    torch.testing.assert_close(
        physical["model.layers.1.mlp.up_proj.weight"],
        dense["model.layers.1.mlp.up_proj.weight"][[1, 4, 5]],
    )
    torch.testing.assert_close(
        physical["model.layers.1.mlp.down_proj.weight"],
        dense["model.layers.1.mlp.down_proj.weight"][:, [1, 4, 5]],
    )
    assert physical["model.embed_tokens.weight"].data_ptr() != (
        dense["model.embed_tokens.weight"].data_ptr()
    )


def test_physical_and_packed_mlps_match_logical_zero_isolation() -> None:
    torch.manual_seed(7)
    dense = DummyMLP()
    keep = torch.tensor([0, 2, 5])
    physical = PhysicalQwenMLP(dense, intermediate_size=len(keep))
    with torch.no_grad():
        physical.gate_proj.weight.copy_(dense.gate_proj.weight[keep])
        physical.up_proj.weight.copy_(dense.up_proj.weight[keep])
        physical.down_proj.weight.copy_(dense.down_proj.weight[:, keep])

    inputs = torch.randn(2, 3, 4)
    dense_hidden = (
        dense.act_fn(dense.gate_proj(inputs)) * dense.up_proj(inputs)
    )
    zero_isolated = torch.zeros_like(dense_hidden)
    zero_isolated[..., keep] = dense_hidden[..., keep]
    expected = dense.down_proj(zero_isolated)
    torch.testing.assert_close(physical(inputs), expected)

    packed = PackedPhysicalQwenMLP(physical, width_alignment=16)
    torch.testing.assert_close(
        packed(inputs), expected, rtol=1e-6, atol=1e-7
    )
    assert packed.intermediate_size == 3
    assert packed.aligned_intermediate_size == 16
    assert torch.count_nonzero(packed.gate_up_proj.weight[3:16]) == 0
    assert torch.count_nonzero(packed.gate_up_proj.weight[19:]) == 0
    assert torch.count_nonzero(packed.down_proj.weight[:, 3:]) == 0


def _write_mask(path: Path, selected: list[list[int]]) -> None:
    pairs = [
        (layer, channel)
        for layer, channels in enumerate(selected)
        for channel in channels
    ]
    np.savez(path, mlp_final=np.asarray(pairs, dtype=np.int32))


def _logical_zero_hooks(
    model: nn.Module,
    selected: list[list[int]],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    for layer, channels in zip(model.model.layers, selected):
        keep = torch.tensor(channels)

        def patch(_module, args, keep=keep):
            hidden = args[0]
            isolated = torch.zeros_like(hidden)
            isolated[..., keep] = hidden[..., keep]
            return (isolated,) + args[1:]

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(patch))
    return handles


def test_tiny_qwen2_bundle_strict_loads_without_dense_mlps(
    tmp_path: Path,
) -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    safetensors = pytest.importorskip("safetensors.torch")

    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    selected_lists = [[0, 3], [1, 4, 5]]
    selected = [torch.tensor(values) for values in selected_lists]
    mask = tmp_path / "mask.npz"
    _write_mask(mask, selected_lists)
    output = tmp_path / "physical"

    metadata = write_physical_bundle(
        model=model,
        selected=selected,
        output=output,
        source_model="tiny-qwen2-test",
        source_revision="test",
        mask_path=mask,
        candidate_id="tiny",
        include_scripts=True,
    )
    assert metadata["isolation"]["kept_per_layer"] == {"0": 2, "1": 3}
    assert metadata["isolation"]["kept_total"] == 5
    assert (
        metadata["physicalization"]["physical_parameters"]
        < metadata["physicalization"]["dense_parameters"]
    )
    checkpoint = safetensors.load_file(output / "model.safetensors")
    assert checkpoint["model.layers.0.mlp.gate_proj.weight"].shape == (2, 8)
    assert checkpoint["model.layers.1.mlp.up_proj.weight"].shape == (3, 8)
    assert checkpoint["model.layers.0.mlp.down_proj.weight"].shape == (8, 2)
    assert {
        path.name for path in (output / "scripts").iterdir()
    } == {
        "bfcl_direct_qwen3.py",
        "load_arithmetic_physical_bundle.py",
        "load_bfcl_physical_bundle.py",
        "triton_silu_mul.py",
    }

    input_ids = torch.tensor([[1, 5, 7, 2]])
    hooks = _logical_zero_hooks(model, selected_lists)
    with torch.inference_mode():
        expected = model(input_ids).logits
    for handle in hooks:
        handle.remove()

    loaded, tokenizer, receipt = load_arithmetic_physical_bundle(
        output,
        device="cpu",
        restore_tokenizer=False,
    )
    assert tokenizer is None
    assert receipt["canonical_checkpoint_load"] == {
        "strict": True,
        "status": "pass",
    }
    assert receipt["dense_mlp_allocated"] is False
    assert receipt["donor_model_loaded"] is False
    assert [layer.mlp.intermediate_size for layer in loaded.model.layers] == [2, 3]
    assert all(
        layer.mlp.intermediate_size != config.intermediate_size
        for layer in loaded.model.layers
    )
    with torch.inference_mode():
        actual = loaded(input_ids).logits
    torch.testing.assert_close(actual, expected)

    loaded_packed, _, packed_receipt = load_arithmetic_physical_bundle(
        output,
        device="cpu",
        mlp_implementation="packed_gate_up",
        width_alignment=16,
        restore_tokenizer=False,
    )
    assert all(
        isinstance(layer.mlp, PackedPhysicalQwenMLP)
        for layer in loaded_packed.model.layers
    )
    assert packed_receipt["mlp_runtime"]["padding_channels"] == 27
    with torch.inference_mode():
        packed_actual = loaded_packed(input_ids).logits
    torch.testing.assert_close(packed_actual, expected, rtol=1e-5, atol=1e-6)

    # Run only from the staged artifact directory.  This proves the copied
    # loader dependency closure does not need the PRISM source checkout.
    standalone = subprocess.run(
        [
            sys.executable,
            str(output / "scripts" / "load_arithmetic_physical_bundle.py"),
            "--bundle",
            str(output),
            "--device",
            "cpu",
            "--no-tokenizer",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    standalone_receipt = json.loads(standalone.stdout)
    assert standalone_receipt["canonical_checkpoint_load"]["status"] == "pass"
    assert standalone_receipt["dense_mlp_allocated"] is False


def test_parameter_accounting_separates_parameters_and_tensor_bytes() -> None:
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Parameter(torch.ones(6, 4))
            self.up = nn.Parameter(torch.ones(6, 4))
            self.down = nn.Parameter(torch.ones(4, 6))
            self.other = nn.Parameter(torch.ones(3))

    model = Tiny()
    dense = {
        "model.layers.0.mlp.gate_proj.weight": torch.ones(6, 4),
        "model.layers.0.mlp.up_proj.weight": torch.ones(6, 4),
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 6),
        "other": torch.ones(3),
    }
    physical = {
        "model.layers.0.mlp.gate_proj.weight": torch.ones(2, 4),
        "model.layers.0.mlp.up_proj.weight": torch.ones(2, 4),
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 2),
        "other": torch.ones(3),
    }
    receipt = parameter_accounting(model, dense, physical)
    assert receipt["dense_parameters"] == 75
    assert receipt["physical_parameters"] == 27
    assert receipt["physical_serialized_tensor_bytes"] == 27 * 4
