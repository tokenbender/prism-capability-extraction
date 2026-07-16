#!/usr/bin/env python3
"""Strictly load a jagged Qwen2 arithmetic physical-substrate bundle.

Dense Qwen2 MLP modules are created only on the meta device.  They are replaced
with the recorded per-layer physical widths before the safetensors checkpoint
is assigned, so strict loading never allocates discarded dense MLP parameters.

The runtime repacking primitives are shared with the proven BFCL Issue #19
loader.  The canonical checkpoint remains separate gate/up/down tensors;
packing, alignment padding, and optional Triton activation dispatch happen
only after strict load and are fully receipted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from load_bfcl_physical_bundle import (
    DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS,
    SUPPORTED_ACTIVATION_IMPLEMENTATIONS,
    SUPPORTED_ATTENTION_IMPLEMENTATIONS,
    SUPPORTED_MLP_IMPLEMENTATIONS,
    SUPPORTED_WIDTH_ALIGNMENTS,
    PackedPhysicalQwenMLP,
    PhysicalQwenMLP,
    configure_mlp_runtime,
    decoder_layers,
    install_physical_mlps,
    probe_triton_activation_runtime,
    validate_packed_mlp,
    validate_triton_silu_mul_input,
)


SUPPORTED_FORMAT = "qwen_physical_mlp_substrate_v1"
SUPPORTED_MODEL_TYPE = "qwen2"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
TIED_WEIGHT_SOURCE = "model.embed_tokens.weight"
TIED_WEIGHT_ALIAS = "lm_head.weight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle_metadata(bundle: Path) -> dict[str, Any]:
    path = bundle / "substrate_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing bundle metadata: {path}")
    metadata = json.loads(path.read_text())
    if metadata.get("format") != SUPPORTED_FORMAT:
        raise ValueError(
            f"unsupported format {metadata.get('format')!r}; "
            f"expected {SUPPORTED_FORMAT!r}"
        )
    if metadata.get("model_type") != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"unsupported model_type {metadata.get('model_type')!r}; "
            f"expected {SUPPORTED_MODEL_TYPE!r}"
        )
    if metadata.get("isolation", {}).get("operator") != (
        "zero_complement_no_donor_activations"
    ):
        raise ValueError("bundle is not a zero-isolated arithmetic substrate")
    return metadata


def _validate_checkpoint_against_metadata(
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    checkpoint_files: list[Path],
    checkpoint_index: Path | None,
    weight_map: dict[str, str] | None,
    shard_inventory: dict[str, set[str]],
    index_total_size: int | None,
) -> dict[str, Any]:
    physicalization = metadata.get("physicalization", {})
    expected_files = physicalization.get("checkpoint_files")
    file_receipts = [
        {
            "file": path.name,
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in checkpoint_files
    ]
    if expected_files:
        expected_by_name = {
            str(item["file"]): item for item in expected_files
        }
        if set(expected_by_name) != {item["file"] for item in file_receipts}:
            raise ValueError("checkpoint shard inventory does not match metadata")
        for actual in file_receipts:
            expected = expected_by_name[actual["file"]]
            if actual["sha256"] != expected.get("sha256"):
                raise ValueError(
                    f"checkpoint SHA-256 mismatch for {actual['file']}: "
                    f"{actual['sha256']} != {expected.get('sha256')}"
                )
            if actual["file_bytes"] != int(expected.get("file_bytes", -1)):
                raise ValueError(
                    f"checkpoint byte-size mismatch for {actual['file']}"
                )
    elif len(checkpoint_files) == 1:
        expected_hash = physicalization.get("checkpoint_sha256")
        if expected_hash and file_receipts[0]["sha256"] != expected_hash:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"{file_receipts[0]['sha256']} != {expected_hash}"
            )
    else:
        raise ValueError("sharded checkpoint is missing an inventory contract")

    index_receipt = None
    if checkpoint_index is not None:
        if weight_map is None or index_total_size is None:
            raise ValueError("sharded checkpoint is missing its parsed index contract")
        if set(weight_map) != set(state):
            missing_from_index = sorted(set(state) - set(weight_map))
            missing_from_state = sorted(set(weight_map) - set(state))
            raise ValueError(
                "checkpoint index tensor inventory does not match shard contents: "
                f"unindexed={missing_from_index}, missing={missing_from_state}"
            )
        expected_by_shard: dict[str, set[str]] = {
            path.name: set() for path in checkpoint_files
        }
        for tensor_name, filename in weight_map.items():
            if filename not in expected_by_shard:
                raise ValueError(
                    f"checkpoint index assigns {tensor_name} to unknown shard "
                    f"{filename}"
                )
            expected_by_shard[filename].add(tensor_name)
        for filename, actual_names in shard_inventory.items():
            expected_names = expected_by_shard.get(filename)
            if expected_names != actual_names:
                raise ValueError(
                    f"checkpoint index assignments do not match {filename}: "
                    f"expected={sorted(expected_names or set())}, "
                    f"actual={sorted(actual_names)}"
                )
        index_receipt = {
            "file": checkpoint_index.name,
            "file_bytes": checkpoint_index.stat().st_size,
            "sha256": sha256(checkpoint_index),
        }
        expected_index_hash = physicalization.get("checkpoint_index_sha256")
        if (
            expected_index_hash
            and index_receipt["sha256"] != expected_index_hash
        ):
            raise ValueError(
                "checkpoint index SHA-256 mismatch: "
                f"{index_receipt['sha256']} != {expected_index_hash}"
            )
    actual_tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state.values()
    )
    expected_tensor_bytes = int(
        physicalization.get("physical_serialized_tensor_bytes", -1)
    )
    if actual_tensor_bytes != expected_tensor_bytes:
        raise ValueError(
            "serialized tensor-byte mismatch: "
            f"{actual_tensor_bytes} != {expected_tensor_bytes}"
        )
    if index_total_size is not None and index_total_size != actual_tensor_bytes:
        raise ValueError(
            "checkpoint index total_size mismatch: "
            f"{index_total_size} != {actual_tensor_bytes}"
        )
    return {
        "checkpoint_files": file_receipts,
        "checkpoint_index": index_receipt,
        "serialized_tensor_bytes": actual_tensor_bytes,
        "tensor_count": len(state),
    }


def resolve_checkpoint_files(
    bundle: Path,
    metadata: dict[str, Any],
) -> tuple[list[Path], Path | None, dict[str, str] | None, int | None]:
    """Resolve and validate the checkpoint inventory named by metadata."""

    bundle = bundle.resolve()

    def require_contained_regular_file(path: Path, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
        resolved = path.resolve(strict=True)
        if resolved.parent != bundle:
            raise ValueError(f"{label} escapes the bundle root: {path}")

    physicalization = metadata.get("physicalization", {})
    checkpoint_name = physicalization.get("checkpoint", "model.safetensors")
    if checkpoint_name not in {
        "model.safetensors",
        "model.safetensors.index.json",
    }:
        raise ValueError(
            f"unsupported or unsafe checkpoint contract {checkpoint_name!r}"
        )
    checkpoint = bundle / checkpoint_name
    require_contained_regular_file(checkpoint, "checkpoint contract file")
    if checkpoint_name.endswith(".index.json"):
        index = json.loads(checkpoint.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        if not all(
            isinstance(name, str)
            and name
            and isinstance(filename, str)
            and filename
            for name, filename in weight_map.items()
        ):
            raise ValueError("checkpoint index weight_map must contain strings")
        filenames = sorted(set(weight_map.values()))
        for filename in filenames:
            path = Path(filename)
            if path.name != filename or path.suffix != ".safetensors":
                raise ValueError(
                    f"checkpoint index contains unsafe shard path {filename!r}"
                )
        index_metadata = index.get("metadata")
        if not isinstance(index_metadata, dict):
            raise ValueError("checkpoint index has no metadata object")
        total_size = index_metadata.get("total_size")
        if (
            not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size <= 0
        ):
            raise ValueError("checkpoint index total_size must be a positive integer")
        files = [bundle / filename for filename in filenames]
        for path in files:
            require_contained_regular_file(path, "checkpoint shard")
        return files, checkpoint, dict(weight_map), total_size
    return [checkpoint], None, None, None


def restore_tied_weight_alias(
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """Reconstruct an explicitly omitted tied alias before strict assignment."""

    receipt: dict[str, Any] = {
        "enabled": tie_word_embeddings,
        "source": None,
        "alias": None,
        "alias_serialized": None,
        "alias_reconstructed": False,
    }
    if not tie_word_embeddings:
        return receipt
    if TIED_WEIGHT_SOURCE not in state:
        raise ValueError(
            f"tied checkpoint is missing source tensor {TIED_WEIGHT_SOURCE}"
        )
    receipt.update(
        {
            "source": TIED_WEIGHT_SOURCE,
            "alias": TIED_WEIGHT_ALIAS,
            "alias_serialized": TIED_WEIGHT_ALIAS in state,
        }
    )
    if TIED_WEIGHT_ALIAS in state:
        if not torch.equal(
            state[TIED_WEIGHT_SOURCE],
            state[TIED_WEIGHT_ALIAS],
        ):
            raise ValueError(
                "config requires tied word embeddings but serialized embedding "
                "and LM-head tensors differ"
            )
        return receipt

    contract = metadata.get("loader_contract", {}).get(
        "tied_weight_serialization", {}
    )
    physical_contract = metadata.get("physicalization", {}).get(
        "tied_weight_serialization", {}
    )
    if contract != physical_contract:
        raise ValueError(
            "loader and physicalization tied-weight contracts differ"
        )
    if (
        contract.get("enabled") is not True
        or contract.get("alias_proven") is not True
        or contract.get("source") != TIED_WEIGHT_SOURCE
        or TIED_WEIGHT_ALIAS not in (contract.get("omitted_aliases") or [])
    ):
        raise ValueError(
            "tied checkpoint omitted lm_head.weight without an explicit "
            "proven-alias reconstruction contract"
        )
    state[TIED_WEIGHT_ALIAS] = state[TIED_WEIGHT_SOURCE]
    receipt["alias_reconstructed"] = True
    return receipt


def load_arithmetic_physical_bundle(
    bundle: Path,
    *,
    device: str = "cuda:0",
    attention_implementation: str = "eager",
    mlp_implementation: str = "separate",
    activation_implementation: str = "torch",
    hybrid_activation_threshold_rows: int = (
        DEFAULT_HYBRID_ACTIVATION_THRESHOLD_ROWS
    ),
    width_alignment: int = 1,
    allow_mlp_fallback: bool = False,
    restore_tokenizer: bool = True,
) -> tuple[nn.Module, Any | None, dict[str, Any]]:
    """Reconstruct the physical Qwen2 model and strictly assign its tensors."""

    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )

    started = time.perf_counter()
    bundle = bundle.resolve()
    metadata = read_bundle_metadata(bundle)
    if attention_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"unsupported attention implementation {attention_implementation!r}"
        )
    (
        checkpoint_files,
        checkpoint_index,
        weight_map,
        index_total_size,
    ) = resolve_checkpoint_files(bundle, metadata)
    dtype_name = str(metadata.get("dtype"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported bundle dtype {dtype_name!r}")
    expected_dtype = DTYPES[dtype_name]

    construction_started = time.perf_counter()
    config = AutoConfig.from_pretrained(bundle, local_files_only=True)
    if config.model_type != SUPPORTED_MODEL_TYPE:
        raise ValueError(
            f"config model_type {config.model_type!r} is not Qwen2"
        )
    config.dtype = expected_dtype
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attention_implementation,
        )
        widths = install_physical_mlps(model, metadata)
    construction_seconds = time.perf_counter() - construction_started

    checkpoint_started = time.perf_counter()
    state: dict[str, torch.Tensor] = {}
    shard_inventory: dict[str, set[str]] = {}
    for checkpoint_file in checkpoint_files:
        shard = load_file(checkpoint_file, device="cpu")
        shard_inventory[checkpoint_file.name] = set(shard)
        duplicates = set(state).intersection(shard)
        if duplicates:
            raise ValueError(
                f"checkpoint shards contain duplicate tensors: {sorted(duplicates)}"
            )
        state.update(shard)
    checkpoint_receipt = _validate_checkpoint_against_metadata(
        state,
        metadata,
        checkpoint_files,
        checkpoint_index,
        weight_map,
        shard_inventory,
        index_total_size,
    )
    dtype_mismatches = {
        name: str(tensor.dtype)
        for name, tensor in state.items()
        if torch.is_floating_point(tensor) and tensor.dtype != expected_dtype
    }
    if dtype_mismatches:
        raise ValueError(
            f"checkpoint floating dtypes do not match {expected_dtype}: "
            f"{dtype_mismatches}"
        )
    tied_weight_serialization = restore_tied_weight_alias(
        state,
        metadata,
        tie_word_embeddings=bool(
            getattr(config, "tie_word_embeddings", False)
        ),
    )
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict checkpoint mismatch: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state
    # assign=True materializes the two serialized aliases as independent
    # Parameters. Restore the config-declared tying only after proving the
    # checkpoint tensors are identical, so parameter accounting and runtime
    # semantics match the source model without weakening strict loading.
    if bool(getattr(config, "tie_word_embeddings", False)):
        model.tie_weights()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("one or more physical parameters remained on meta")
    canonical_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    expected_parameters = int(
        metadata.get("physicalization", {}).get("physical_parameters", -1)
    )
    if canonical_parameters != expected_parameters:
        raise RuntimeError(
            f"physical parameter mismatch: {canonical_parameters} "
            f"!= {expected_parameters}"
        )
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    repack_started = time.perf_counter()
    mlp_runtime = configure_mlp_runtime(
        model,
        widths,
        implementation=mlp_implementation,
        activation_implementation=activation_implementation,
        hybrid_activation_threshold_rows=hybrid_activation_threshold_rows,
        width_alignment=width_alignment,
        allow_fallback=allow_mlp_fallback,
    )
    repack_seconds = time.perf_counter() - repack_started
    runtime_parameters = sum(parameter.numel() for parameter in model.parameters())

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested_device}")
    move_started = time.perf_counter()
    model = model.to(requested_device).eval()
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    move_seconds = time.perf_counter() - move_started

    probe_started = time.perf_counter()
    activation_probe = probe_triton_activation_runtime(model)
    probe_seconds = time.perf_counter() - probe_started
    mlp_runtime["activation_runtime_probe"] = activation_probe

    tokenizer = None
    restore_started = time.perf_counter()
    generation_config_path = bundle / "generation_config.json"
    if generation_config_path.is_file():
        model.generation_config = GenerationConfig.from_pretrained(
            bundle, local_files_only=True
        )
    elif restore_tokenizer:
        raise FileNotFoundError(
            "bundle is missing generation_config.json required for generation"
        )
    if restore_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            bundle,
            local_files_only=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    restore_seconds = time.perf_counter() - restore_started

    layer_widths = [
        int(layer.mlp.intermediate_size) for layer in decoder_layers(model)
    ]
    if layer_widths != widths:
        raise RuntimeError(
            f"active-width mismatch after runtime configuration: "
            f"{layer_widths} != {widths}"
        )
    receipt = {
        "status": "pass",
        "format": metadata["format"],
        "model_type": config.model_type,
        "bundle": str(bundle),
        "device": str(requested_device),
        "attention_implementation": getattr(
            model.config, "_attn_implementation", attention_implementation
        ),
        "canonical_checkpoint_load": {"strict": True, "status": "pass"},
        "dense_mlp_allocated": False,
        "donor_model_loaded": False,
        "layers": len(widths),
        "kept_total": sum(widths),
        "kept_per_layer": widths,
        "checkpoint": metadata.get("physicalization", {}).get("checkpoint"),
        "checkpoint_receipt": checkpoint_receipt,
        "tied_weight_serialization": tied_weight_serialization,
        "parameter_dtype": str(expected_dtype),
        "parameter_accounting": {
            "canonical_physical_parameters": canonical_parameters,
            "runtime_parameters": runtime_parameters,
            "runtime_added_parameters": runtime_parameters
            - canonical_parameters,
            "runtime_added_parameter_bytes": (
                runtime_parameters - canonical_parameters
            )
            * torch.empty((), dtype=expected_dtype).element_size(),
        },
        "mlp_runtime": mlp_runtime,
        "tokenizer_restored": tokenizer is not None,
        "timings_seconds": {
            "config_and_meta_construction": construction_seconds,
            "strict_checkpoint_load": checkpoint_seconds,
            "runtime_repack": repack_seconds,
            "device_move": move_seconds,
            "activation_runtime_probe": probe_seconds,
            "generation_and_tokenizer_restore": restore_seconds,
            "loader_total": time.perf_counter() - started,
        },
    }
    return model, tokenizer, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-implementation",
        choices=SUPPORTED_ATTENTION_IMPLEMENTATIONS,
        default="eager",
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=SUPPORTED_MLP_IMPLEMENTATIONS,
        default="separate",
    )
    parser.add_argument(
        "--activation-implementation",
        choices=SUPPORTED_ACTIVATION_IMPLEMENTATIONS,
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
        choices=SUPPORTED_WIDTH_ALIGNMENTS,
        default=1,
    )
    parser.add_argument("--allow-mlp-fallback", action="store_true")
    parser.add_argument("--no-tokenizer", action="store_true")
    args = parser.parse_args()

    model, _, receipt = load_arithmetic_physical_bundle(
        args.bundle,
        device=args.device,
        attention_implementation=args.attention_implementation,
        mlp_implementation=args.mlp_implementation,
        activation_implementation=args.activation_implementation,
        hybrid_activation_threshold_rows=args.hybrid_activation_threshold_rows,
        width_alignment=args.width_alignment,
        allow_mlp_fallback=args.allow_mlp_fallback,
        restore_tokenizer=not args.no_tokenizer,
    )
    receipt["parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
