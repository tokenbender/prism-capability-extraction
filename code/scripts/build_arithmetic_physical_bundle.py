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
TIED_WEIGHT_SOURCE = "model.embed_tokens.weight"
TIED_WEIGHT_ALIAS = "lm_head.weight"


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


def tensors_share_exact_storage(
    source: torch.Tensor,
    alias: torch.Tensor,
) -> bool:
    """Return whether two tensors are the same exact view of one storage."""

    return (
        source.device == alias.device
        and source.dtype == alias.dtype
        and source.shape == alias.shape
        and source.stride() == alias.stride()
        and source.storage_offset() == alias.storage_offset()
        and source.untyped_storage().data_ptr()
        == alias.untyped_storage().data_ptr()
    )


def prove_model_tied_weight_alias(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> bool:
    """Prove the config-declared embedding/LM-head parameter alias."""

    tie_word_embeddings = bool(
        getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    )
    if not tie_word_embeddings:
        return False
    missing = [
        key for key in (TIED_WEIGHT_SOURCE, TIED_WEIGHT_ALIAS) if key not in state
    ]
    if missing:
        raise ValueError(f"tied state is missing required tensors: {missing}")
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise ValueError("model cannot expose its declared tied embeddings")
    input_embeddings = get_input()
    output_embeddings = get_output()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model is missing a declared tied embedding module")
    source_parameter = getattr(input_embeddings, "weight", None)
    alias_parameter = getattr(output_embeddings, "weight", None)
    if not isinstance(source_parameter, torch.Tensor) or not isinstance(
        alias_parameter, torch.Tensor
    ):
        raise ValueError("declared tied embedding modules have no tensor weights")
    if not tensors_share_exact_storage(source_parameter, alias_parameter):
        raise ValueError(
            "config declares tied word embeddings but the model parameters "
            "are not an actual storage alias"
        )
    if not tensors_share_exact_storage(
        state[TIED_WEIGHT_SOURCE],
        state[TIED_WEIGHT_ALIAS],
    ):
        raise ValueError(
            "config declares tied word embeddings but state_dict aliases "
            "do not share exact storage"
        )
    return True


def prepare_state_for_serialization(
    state: Mapping[str, torch.Tensor],
    *,
    tie_word_embeddings: bool,
    tied_weight_alias_proven: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Remove a proven tied-weight alias from the on-disk state.

    ``state_dict()`` exposes both embedding and LM-head keys even when the
    parameters are tied. Safetensors does not preserve aliases, so serializing
    both keys silently duplicates the largest tensor in this model. The strict
    loader reconstructs the omitted alias from this explicit contract.
    """

    serialized = dict(state)
    receipt: dict[str, Any] = {
        "enabled": tie_word_embeddings,
        "alias_proven": tied_weight_alias_proven,
        "source": None,
        "omitted_aliases": [],
    }
    if not tie_word_embeddings:
        if tied_weight_alias_proven:
            raise ValueError(
                "a tied-weight alias cannot be proven when tying is disabled"
            )
        return serialized, receipt
    if not tied_weight_alias_proven:
        raise ValueError(
            "config declares tied word embeddings but an actual parameter "
            "alias was not proven"
        )
    missing = [
        key
        for key in (TIED_WEIGHT_SOURCE, TIED_WEIGHT_ALIAS)
        if key not in serialized
    ]
    if missing:
        raise ValueError(f"tied state is missing required tensors: {missing}")
    if not torch.equal(
        serialized[TIED_WEIGHT_SOURCE],
        serialized[TIED_WEIGHT_ALIAS],
    ):
        raise ValueError("declared tied embedding and LM-head tensors differ")
    serialized.pop(TIED_WEIGHT_ALIAS)
    receipt.update(
        {
            "source": TIED_WEIGHT_SOURCE,
            "omitted_aliases": [TIED_WEIGHT_ALIAS],
        }
    )
    return serialized, receipt


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
    *,
    tied_weight_alias_proven: bool | None = None,
) -> dict[str, Any]:
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
    dense_expanded_bytes = state_tensor_bytes(dense_state)
    physical_expanded_bytes = state_tensor_bytes(physical_state)
    tie_word_embeddings = bool(
        getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    )
    if tied_weight_alias_proven is None:
        tied_weight_alias_proven = prove_model_tied_weight_alias(
            model,
            dense_state,
        )
    dense_serialized, dense_tied = prepare_state_for_serialization(
        dense_state,
        tie_word_embeddings=tie_word_embeddings,
        tied_weight_alias_proven=tied_weight_alias_proven,
    )
    physical_serialized, physical_tied = prepare_state_for_serialization(
        physical_state,
        tie_word_embeddings=tie_word_embeddings,
        tied_weight_alias_proven=tied_weight_alias_proven,
    )
    if dense_tied != physical_tied:
        raise ValueError("dense and physical tied-weight contracts differ")
    dense_bytes = state_tensor_bytes(dense_serialized)
    physical_bytes = state_tensor_bytes(physical_serialized)
    return {
        "dense_parameters": int(dense_parameters),
        "physical_parameters": int(physical_parameters),
        "physical_parameter_fraction": physical_parameters / dense_parameters,
        "dense_expanded_state_tensor_bytes": int(dense_expanded_bytes),
        "physical_expanded_state_tensor_bytes": int(physical_expanded_bytes),
        "dense_serialized_tensor_bytes": int(dense_bytes),
        "physical_serialized_tensor_bytes": int(physical_bytes),
        "physical_serialized_tensor_fraction": physical_bytes / dense_bytes,
        "tied_weight_serialization": physical_tied,
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


def write_safetensors_checkpoint(
    state: Mapping[str, torch.Tensor],
    output: Path,
    *,
    max_shard_bytes: int | None = None,
) -> dict[str, Any]:
    """Write one safetensors file or a deterministic standard shard set."""

    from safetensors.torch import save_file

    if max_shard_bytes is not None and max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    items = list(state.items())
    if not items:
        raise ValueError("cannot serialize an empty state dictionary")
    total_tensor_bytes = state_tensor_bytes(state)
    if max_shard_bytes is None or total_tensor_bytes <= max_shard_bytes:
        checkpoint = output / CHECKPOINT_NAME
        save_file(dict(items), checkpoint, metadata={"format": "pt"})
        return {
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_file_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
            "sharding": {
                "enabled": False,
                "algorithm": "state_dict_insertion_order_greedy_max_bytes_v1",
                "max_shard_bytes": max_shard_bytes,
                "oversize_tensor_policy": "single_tensor_may_exceed_limit",
            },
            "checkpoint_files": [
                {
                    "file": CHECKPOINT_NAME,
                    "file_bytes": checkpoint.stat().st_size,
                    "tensor_bytes": total_tensor_bytes,
                    "sha256": sha256(checkpoint),
                }
            ],
        }

    shards: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    current_bytes = 0
    for name, tensor in items:
        size = tensor_bytes(tensor)
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append((name, tensor))
        current_bytes += size
    if current:
        shards.append(current)

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        path = output / filename
        shard_state = dict(shard)
        save_file(shard_state, path, metadata={"format": "pt"})
        for name in shard_state:
            weight_map[name] = filename
        files.append(
            {
                "file": filename,
                "file_bytes": path.stat().st_size,
                "tensor_bytes": state_tensor_bytes(shard_state),
                "sha256": sha256(path),
            }
        )
    index_name = f"{CHECKPOINT_NAME}.index.json"
    index_path = output / index_name
    _write_json(
        index_path,
        {
            "metadata": {"total_size": total_tensor_bytes},
            "weight_map": weight_map,
        },
    )
    return {
        "checkpoint": index_name,
        "checkpoint_index_file_bytes": index_path.stat().st_size,
        "checkpoint_index_sha256": sha256(index_path),
        "sharding": {
            "enabled": True,
            "algorithm": "state_dict_insertion_order_greedy_max_bytes_v1",
            "max_shard_bytes": max_shard_bytes,
            "oversize_tensor_policy": "single_tensor_may_exceed_limit",
            "shard_count": shard_count,
        },
        "checkpoint_files": files,
    }


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
    max_shard_bytes: int | None = None,
) -> dict[str, Any]:
    """Serialize a loaded merged Qwen2 model as a strict jagged bundle."""

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
    tied_weight_alias_proven = prove_model_tied_weight_alias(
        model,
        dense_state,
    )
    physical_state, tensor_shapes = build_physical_state_dict(
        dense_state, selected
    )
    accounting = parameter_accounting(
        model,
        dense_state,
        physical_state,
        tied_weight_alias_proven=tied_weight_alias_proven,
    )
    serialized_state, tied_weight_serialization = prepare_state_for_serialization(
        physical_state,
        tie_word_embeddings=bool(
            getattr(model.config, "tie_word_embeddings", False)
        ),
        tied_weight_alias_proven=tied_weight_alias_proven,
    )
    if (
        accounting["tied_weight_serialization"]
        != tied_weight_serialization
    ):
        raise RuntimeError("tied-weight serialization receipt drifted")

    model.config.save_pretrained(staging)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(staging)
    if tokenizer is not None:
        tokenizer.save_pretrained(staging)
    checkpoint_receipt = write_safetensors_checkpoint(
        serialized_state,
        staging,
        max_shard_bytes=max_shard_bytes,
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
            **checkpoint_receipt,
        },
        "loader_contract": {
            "strict": True,
            "dense_mlp_tensors_required": False,
            "runtime_mask_required": False,
            "donor_model_required": False,
            "tied_weight_serialization": tied_weight_serialization,
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
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        help="write a standard safetensors shard set when the state exceeds this size",
    )
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
        max_shard_bytes=args.max_shard_bytes,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
