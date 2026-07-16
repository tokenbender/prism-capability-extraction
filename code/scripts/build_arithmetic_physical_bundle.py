#!/usr/bin/env python3
"""Build a physically jagged Qwen2 arithmetic substrate from a merged model.

The input mask is the PRISM ``mlp_final`` NPZ format: an ``[N, 2]`` array of
``(layer, intermediate_channel)`` pairs.  Selected rows of ``gate_proj`` and
``up_proj`` and the matching columns of ``down_proj`` are serialized.  The
discarded channels are therefore absent rather than zero-filled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SUPPORTED_FORMAT = "qwen_physical_mlp_substrate_v1"
SUPPORTED_MODEL_TYPE = "qwen2"
CHECKPOINT_NAME = "model.safetensors"
METADATA_NAME = "substrate_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def state_tensor_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(tensor) for tensor in state.values())


def canonical_selection_sha256(selected: Sequence[torch.Tensor]) -> str:
    payload = [
        [int(index) for index in layer.detach().cpu().tolist()]
        for layer in selected
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_mask_indices(
    path: Path,
    *,
    num_layers: int,
    dense_width: int,
) -> list[torch.Tensor]:
    """Load, validate, and canonically order a PRISM ``mlp_final`` mask."""

    with np.load(path, allow_pickle=False) as payload:
        if "mlp_final" not in payload:
            raise KeyError(f"mlp_final missing from {path}")
        pairs = np.asarray(payload["mlp_final"])
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(
            f"mlp_final must have shape [N, 2], got {tuple(pairs.shape)}"
        )
    if not np.issubdtype(pairs.dtype, np.integer):
        raise TypeError(f"mlp_final must be integral, got {pairs.dtype}")

    per_layer: list[list[int]] = [[] for _ in range(num_layers)]
    seen: set[tuple[int, int]] = set()
    for raw_layer, raw_channel in pairs.tolist():
        layer = int(raw_layer)
        channel = int(raw_channel)
        if not 0 <= layer < num_layers:
            raise ValueError(f"mask layer {layer} is outside [0, {num_layers})")
        if not 0 <= channel < dense_width:
            raise ValueError(
                f"mask channel {channel} is outside [0, {dense_width})"
            )
        pair = (layer, channel)
        if pair in seen:
            raise ValueError(f"duplicate mask selection {pair}")
        seen.add(pair)
        per_layer[layer].append(channel)

    empty_layers = [index for index, values in enumerate(per_layer) if not values]
    if empty_layers:
        raise ValueError(
            "physical Qwen MLPs require at least one retained channel per layer; "
            f"empty layers: {empty_layers}"
        )
    return [
        torch.tensor(sorted(values), dtype=torch.long) for values in per_layer
    ]


def _slice_projection(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    selected = tensor.index_select(dimension, indices.to(tensor.device))
    # Safetensors rejects shared/non-contiguous views.  A detached CPU copy
    # also guarantees the output no longer aliases the dense source model.
    return selected.detach().cpu().contiguous()


def build_physical_state_dict(
    dense_state: Mapping[str, torch.Tensor],
    selected: Sequence[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, list[int]]]]:
    """Return a state dict in which every Qwen2 MLP is physically sliced."""

    physical: dict[str, torch.Tensor] = {}
    expected: set[str] = set()
    shapes: dict[str, dict[str, list[int]]] = {}
    for layer, indices in enumerate(selected):
        prefix = f"model.layers.{layer}.mlp"
        names = {
            "gate_weight": f"{prefix}.gate_proj.weight",
            "up_weight": f"{prefix}.up_proj.weight",
            "down_weight": f"{prefix}.down_proj.weight",
        }
        expected.update(names.values())
        for key in names.values():
            if key not in dense_state:
                raise KeyError(f"dense model is missing required tensor {key}")
        shapes[str(layer)] = {}

    for name, tensor in dense_state.items():
        replacement: torch.Tensor | None = None
        for layer, indices in enumerate(selected):
            prefix = f"model.layers.{layer}.mlp"
            if name in (
                f"{prefix}.gate_proj.weight",
                f"{prefix}.up_proj.weight",
            ):
                replacement = _slice_projection(tensor, indices, dimension=0)
            elif name == f"{prefix}.down_proj.weight":
                replacement = _slice_projection(tensor, indices, dimension=1)
            elif name in (
                f"{prefix}.gate_proj.bias",
                f"{prefix}.up_proj.bias",
            ):
                replacement = _slice_projection(tensor, indices, dimension=0)
            if replacement is not None:
                tensor_name = name.removeprefix(prefix + ".")
                shapes[str(layer)][tensor_name] = list(replacement.shape)
                break
        if replacement is None:
            # Copy every tensor so tied/shared dense storage cannot leak into
            # the standalone safetensors checkpoint.
            replacement = tensor.detach().cpu().clone().contiguous()
        physical[name] = replacement

    missing = expected - physical.keys()
    if missing:
        raise RuntimeError(f"physical state construction missed tensors: {missing}")
    return physical, shapes


def parameter_accounting(
    model: torch.nn.Module,
    dense_state: Mapping[str, torch.Tensor],
    physical_state: Mapping[str, torch.Tensor],
) -> dict[str, int | float]:
    """Count real model parameters separately from serialized tensor bytes."""

    if dense_state.keys() != physical_state.keys():
        raise ValueError("dense and physical state dictionaries have different keys")
    dense_parameters = sum(parameter.numel() for parameter in model.parameters())
    dense_mlp = 0
    physical_mlp = 0
    for name, tensor in dense_state.items():
        if ".mlp." in name:
            dense_mlp += tensor.numel()
            physical_mlp += physical_state[name].numel()
    physical_parameters = dense_parameters - dense_mlp + physical_mlp
    if not 0 < physical_parameters <= dense_parameters:
        raise ValueError(
            "state dictionaries are inconsistent with the supplied model: "
            f"dense={dense_parameters}, dense_mlp={dense_mlp}, "
            f"physical_mlp={physical_mlp}"
        )
    dense_bytes = state_tensor_bytes(dense_state)
    physical_bytes = state_tensor_bytes(physical_state)
    return {
        "dense_parameters": int(dense_parameters),
        "physical_parameters": int(physical_parameters),
        "physical_parameter_fraction": physical_parameters / dense_parameters,
        "dense_serialized_tensor_bytes": int(dense_bytes),
        "physical_serialized_tensor_bytes": int(physical_bytes),
        "physical_serialized_tensor_fraction": physical_bytes / dense_bytes,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_sha256sums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in files
    ]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def write_physical_bundle(
    *,
    model: torch.nn.Module,
    selected: Sequence[torch.Tensor],
    output: Path,
    source_model: str,
    source_revision: str | None,
    mask_path: Path,
    candidate_id: str,
    overwrite: bool = False,
    tokenizer: Any = None,
    include_scripts: bool = True,
) -> dict[str, Any]:
    """Serialize a loaded merged Qwen2 model as a strict jagged bundle."""

    from safetensors.torch import save_file

    if getattr(model.config, "model_type", None) != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"expected model_type {SUPPORTED_MODEL_TYPE!r}, got "
            f"{getattr(model.config, 'model_type', None)!r}"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != len(selected):
        raise ValueError("selection count does not match Qwen2 decoder layers")
    dense_width = int(model.config.intermediate_size)
    for layer, indices in enumerate(selected):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError(f"layer {layer} indices must be a 1D torch.long tensor")
        if indices.numel() <= 0 or indices.numel() > dense_width:
            raise ValueError(f"invalid retained width in layer {layer}")
        if not torch.equal(indices, torch.unique(indices, sorted=True)):
            raise ValueError(f"layer {layer} indices are not sorted and unique")

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass overwrite=True")
        shutil.rmtree(output)
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    dense_state = model.state_dict()
    physical_state, tensor_shapes = build_physical_state_dict(
        dense_state, selected
    )
    accounting = parameter_accounting(model, dense_state, physical_state)

    model.config.save_pretrained(staging)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(staging)
    if tokenizer is not None:
        tokenizer.save_pretrained(staging)
    save_file(
        physical_state,
        staging / CHECKPOINT_NAME,
        metadata={"format": "pt"},
    )

    widths = [int(indices.numel()) for indices in selected]
    dense_total_channels = len(widths) * dense_width
    dtype = next(model.parameters()).dtype
    metadata: dict[str, Any] = {
        "format": SUPPORTED_FORMAT,
        "task": "two_digit_addition",
        "model_type": SUPPORTED_MODEL_TYPE,
        "dtype": str(dtype).removeprefix("torch."),
        "source_model": {
            "repo_or_path": source_model,
            "revision": source_revision,
        },
        "source_mask": {
            "candidate_id": candidate_id,
            "file": mask_path.name,
            "sha256": sha256(mask_path),
            "canonical_selection_sha256": canonical_selection_sha256(selected),
        },
        "isolation": {
            "operator": "zero_complement_no_donor_activations",
            "dense_width": dense_width,
            "dense_total_channels": dense_total_channels,
            "kept_total": sum(widths),
            "kept_fraction": sum(widths) / dense_total_channels,
            "kept_per_layer": {
                str(index): width for index, width in enumerate(widths)
            },
            "attention_pruned": False,
        },
        "physicalization": {
            "gate_proj": "selected_rows",
            "up_proj": "selected_rows",
            "down_proj": "matching_selected_columns",
            "tensor_shapes": tensor_shapes,
            **accounting,
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_file_bytes": (staging / CHECKPOINT_NAME).stat().st_size,
            "checkpoint_sha256": sha256(staging / CHECKPOINT_NAME),
        },
        "loader_contract": {
            "strict": True,
            "dense_mlp_tensors_required": False,
            "runtime_mask_required": False,
            "donor_model_required": False,
            "supported_mlp_implementations": ["separate", "packed_gate_up"],
            "supported_activation_implementations": ["torch", "triton", "hybrid"],
            "supported_width_alignments": [1, 16, 64, 128, 256],
        },
    }
    _write_json(staging / METADATA_NAME, metadata)

    if include_scripts:
        scripts = staging / "scripts"
        scripts.mkdir()
        here = Path(__file__).resolve().parent
        for name in (
            "load_arithmetic_physical_bundle.py",
            "load_bfcl_physical_bundle.py",
            "bfcl_direct_qwen3.py",
            "triton_silu_mul.py",
        ):
            source = here / name
            if not source.is_file():
                raise FileNotFoundError(f"missing required bundle script: {source}")
            shutil.copy2(source, scripts / name)

    _write_sha256sums(staging)
    staging.replace(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="merged dense model ID/path")
    parser.add_argument("--revision")
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    if model.config.model_type != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"arithmetic physicalization supports Qwen2, got {model.config.model_type}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )
    selected = load_mask_indices(
        args.mask,
        num_layers=int(model.config.num_hidden_layers),
        dense_width=int(model.config.intermediate_size),
    )
    metadata = write_physical_bundle(
        model=model,
        selected=selected,
        output=args.output,
        source_model=args.model,
        source_revision=args.revision,
        mask_path=args.mask,
        candidate_id=args.candidate_id,
        overwrite=args.overwrite,
        tokenizer=tokenizer,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
