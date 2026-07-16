from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from load_bfcl_physical_bundle import (  # noqa: E402
    DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    PackedPhysicalQwenMLP,
    PhysicalQwenMLP,
    configure_mlp_runtime,
    install_physical_mlps,
    probe_triton_activation_runtime,
    reference_parity,
    validate_activation_runtime_settings,
    validate_packed_mlp,
    validate_triton_silu_mul_input,
)
import load_bfcl_physical_bundle as physical_loader  # noqa: E402


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

    assert probe_triton_activation_runtime(model) == {
        "status": "not_requested",
        "layers": 0,
        "unique_widths": [],
    }


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


def test_triton_activation_requires_packed_gate_up() -> None:
    model = DummyModel(layers=1)

    with pytest.raises(ValueError, match="require packed_gate_up"):
        configure_mlp_runtime(
            model,
            [3],
            implementation="separate",
            activation_implementation="triton",
        )


def test_triton_selector_is_lazy_and_receipted(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DummyModel(layers=1)
    metadata = {
        "isolation": {
            "kept_total": 3,
            "kept_per_layer": {"0": 3},
        }
    }
    widths = install_physical_mlps(model, metadata)
    sentinel_kernel = lambda gate_up: gate_up[..., : gate_up.shape[-1] // 2]
    monkeypatch.setattr(
        physical_loader,
        "_load_triton_silu_and_mul",
        lambda: sentinel_kernel,
    )

    receipt = configure_mlp_runtime(
        model,
        widths,
        implementation="packed_gate_up",
        activation_implementation="triton",
        width_alignment=16,
    )

    packed = model.model.layers[0].mlp
    assert isinstance(packed, PackedPhysicalQwenMLP)
    assert packed.activation_implementation == "triton"
    assert packed._triton_silu_and_mul is sentinel_kernel
    assert receipt["requested_activation_implementation"] == "triton"
    assert receipt["active_activation_implementation"] == "triton"


def test_hybrid_selector_routes_only_large_flattened_rows_to_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(17)
    physical = PhysicalQwenMLP(DummyMLP(), intermediate_size=3)
    triton_calls: list[tuple[int, ...]] = []

    def sentinel_kernel(gate_up: torch.Tensor) -> torch.Tensor:
        triton_calls.append(tuple(gate_up.shape))
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    monkeypatch.setattr(
        physical_loader,
        "_load_triton_silu_and_mul",
        lambda: sentinel_kernel,
    )
    monkeypatch.setattr(
        physical_loader,
        "validate_triton_silu_mul_input",
        lambda gate_up, *, expected_width: None,
    )
    packed = PackedPhysicalQwenMLP(
        physical,
        width_alignment=1,
        activation_implementation="hybrid",
        hybrid_activation_threshold_rows=6,
    )

    small = torch.randn(1, 5, 4)
    large = torch.randn(2, 3, 4)
    small_gate_up = packed.gate_up_proj(small)
    small_gate, small_up = small_gate_up.chunk(2, dim=-1)
    small_expected = packed.down_proj(
        torch.nn.functional.silu(small_gate) * small_up
    )
    large_gate_up = packed.gate_up_proj(large)
    large_gate, large_up = large_gate_up.chunk(2, dim=-1)
    large_expected = packed.down_proj(
        torch.nn.functional.silu(large_gate) * large_up
    )

    torch.testing.assert_close(packed(small), small_expected)
    assert triton_calls == []
    torch.testing.assert_close(packed(large), large_expected)
    assert triton_calls == [(2, 3, 6)]


def test_hybrid_selector_is_lazy_atomic_and_receipted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DummyModel(layers=1)
    metadata = {
        "isolation": {
            "kept_total": 3,
            "kept_per_layer": {"0": 3},
        }
    }
    widths = install_physical_mlps(model, metadata)
    sentinel_kernel = lambda gate_up: gate_up[..., : gate_up.shape[-1] // 2]
    monkeypatch.setattr(
        physical_loader,
        "_load_triton_silu_and_mul",
        lambda: sentinel_kernel,
    )

    receipt = configure_mlp_runtime(
        model,
        widths,
        implementation="packed_gate_up",
        activation_implementation="hybrid",
        width_alignment=16,
    )

    packed = model.model.layers[0].mlp
    assert isinstance(packed, PackedPhysicalQwenMLP)
    assert packed.activation_implementation == "hybrid"
    assert packed._triton_silu_and_mul is sentinel_kernel
    assert (
        packed.hybrid_activation_threshold_rows
        == DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
    )
    assert (
        receipt["requested_hybrid_activation_threshold_rows"]
        == DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
    )
    assert (
        receipt["active_hybrid_activation_threshold_rows"]
        == DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
    )
    assert receipt["activation_dispatch"] == {
        "mode": "hybrid_row_threshold",
        "flattened_rows": "product_of_all_dimensions_except_last",
        "torch_when_rows_below": DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
        "triton_when_rows_at_least": DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    }

    with pytest.raises(RuntimeError, match="requires loading the model on CUDA"):
        probe_triton_activation_runtime(model)


def test_hybrid_small_row_path_remains_fullgraph_compilable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = PhysicalQwenMLP(DummyMLP(), intermediate_size=3)

    def forbidden_kernel(gate_up: torch.Tensor) -> torch.Tensor:
        raise AssertionError(f"unexpected Triton call for shape {tuple(gate_up.shape)}")

    monkeypatch.setattr(
        physical_loader,
        "_load_triton_silu_and_mul",
        lambda: forbidden_kernel,
    )
    packed = PackedPhysicalQwenMLP(
        physical,
        width_alignment=1,
        activation_implementation="hybrid",
    )
    inputs = torch.randn(2, 3, 4)
    expected = packed(inputs)

    compiled = torch.compile(packed, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs), expected)


@pytest.mark.parametrize(
    ("implementation", "threshold", "message"),
    [
        ("hybrid", 0, "must be positive"),
        ("torch", 1024, "only meaningful"),
        ("triton", 4096, "only meaningful"),
    ],
)
def test_hybrid_threshold_rejects_ambiguous_settings(
    implementation: str,
    threshold: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_activation_runtime_settings(
            activation_implementation=implementation,
            hybrid_activation_threshold_rows=threshold,
        )


def test_hybrid_missing_dependency_preserves_canonical_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DummyModel(layers=2)
    metadata = {
        "isolation": {
            "kept_total": 7,
            "kept_per_layer": {"0": 2, "1": 5},
        }
    }
    widths = install_physical_mlps(model, metadata)
    original_modules = [layer.mlp for layer in model.model.layers]

    def unavailable() -> None:
        raise RuntimeError(
            "activation implementation 'hybrid' requires the optional Triton package"
        )

    monkeypatch.setattr(physical_loader, "_load_triton_silu_and_mul", unavailable)

    with pytest.raises(RuntimeError, match="requires the optional Triton package"):
        configure_mlp_runtime(
            model,
            widths,
            implementation="packed_gate_up",
            activation_implementation="hybrid",
            width_alignment=16,
        )
    assert [layer.mlp for layer in model.model.layers] == original_modules


def test_triton_selector_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DummyModel(layers=1)
    metadata = {
        "isolation": {
            "kept_total": 3,
            "kept_per_layer": {"0": 3},
        }
    }
    widths = install_physical_mlps(model, metadata)

    def unavailable() -> None:
        raise RuntimeError(
            "activation implementation 'triton' requires the optional Triton package"
        )

    monkeypatch.setattr(physical_loader, "_load_triton_silu_and_mul", unavailable)

    with pytest.raises(RuntimeError, match="requires the optional Triton package"):
        configure_mlp_runtime(
            model,
            widths,
            implementation="packed_gate_up",
            activation_implementation="triton",
            width_alignment=16,
        )
    assert isinstance(model.model.layers[0].mlp, PhysicalQwenMLP)


def test_triton_input_validation_is_cuda_low_precision_and_inference_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_input = torch.empty(2, 6, dtype=torch.bfloat16)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="requires a CUDA"):
        validate_triton_silu_mul_input(cpu_input, expected_width=3)

    fake_cuda = Mock()
    fake_cuda.device = torch.device("cuda")
    fake_cuda.dtype = torch.bfloat16
    fake_cuda.ndim = 2
    fake_cuda.shape = (2, 6)
    fake_cuda.is_contiguous.return_value = True
    fake_cuda.numel.return_value = 12

    with torch.inference_mode():
        validate_triton_silu_mul_input(fake_cuda, expected_width=3)

    fake_cuda.dtype = torch.float32
    with torch.inference_mode(), pytest.raises(TypeError, match="only CUDA BF16/FP16"):
        validate_triton_silu_mul_input(fake_cuda, expected_width=3)

    fake_cuda.dtype = torch.float16
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="inference-only"):
        validate_triton_silu_mul_input(fake_cuda, expected_width=3)


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
