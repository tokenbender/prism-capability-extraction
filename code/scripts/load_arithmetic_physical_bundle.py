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
    checkpoint: Path,
) -> dict[str, Any]:
    physicalization = metadata.get("physicalization", {})
    expected_hash = physicalization.get("checkpoint_sha256")
    actual_hash = sha256(checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {actual_hash} != {expected_hash}"
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
    return {
        "checkpoint_sha256": actual_hash,
        "serialized_tensor_bytes": actual_tensor_bytes,
        "tensor_count": len(state),
    }


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
    checkpoint_files = sorted(bundle.glob("*.safetensors"))
    if len(checkpoint_files) != 1:
        raise ValueError(
            f"expected exactly one safetensors checkpoint, found {len(checkpoint_files)}"
        )
    checkpoint = checkpoint_files[0]
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
    state = load_file(checkpoint, device="cpu")
    checkpoint_receipt = _validate_checkpoint_against_metadata(
        state, metadata, checkpoint
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
    tied_weight_keys = ("model.embed_tokens.weight", "lm_head.weight")
    if bool(getattr(config, "tie_word_embeddings", False)):
        missing_tied = [key for key in tied_weight_keys if key not in state]
        if missing_tied:
            raise ValueError(
                f"tied-embedding checkpoint is missing tensors: {missing_tied}"
            )
        if not torch.equal(state[tied_weight_keys[0]], state[tied_weight_keys[1]]):
            raise ValueError(
                "config requires tied word embeddings but serialized embedding "
                "and LM-head tensors differ"
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
        "checkpoint": checkpoint.name,
        "checkpoint_receipt": checkpoint_receipt,
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
