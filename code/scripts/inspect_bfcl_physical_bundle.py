#!/usr/bin/env python3
"""Validate and account for an exported jagged Qwen BFCL substrate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_inventory(root: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    inventory: dict[str, dict[str, Any]] = {}
    total_parameters = 0
    serialized_bytes = 0
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files under {root}")
    for path in files:
        serialized_bytes += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in inventory:
                    raise ValueError(f"duplicate tensor key: {key}")
                tensor = handle.get_slice(key)
                shape = list(tensor.get_shape())
                numel = math.prod(shape)
                inventory[key] = {
                    "shape": shape,
                    "dtype": tensor.get_dtype(),
                    "numel": numel,
                    "file": path.name,
                }
                total_parameters += numel
    return inventory, total_parameters, serialized_bytes


def expected_widths(mask_path: Path) -> list[int]:
    with np.load(mask_path) as archive:
        scores = archive["mlp_scores"]
    if scores.ndim != 2:
        raise ValueError(f"expected rank-2 mlp_scores, got {scores.shape}")
    return [int(np.count_nonzero(layer > 0)) for layer in scores]


def validate_layer_shapes(
    inventory: dict[str, dict[str, Any]],
    widths: list[int],
    hidden_size: int,
    dense_intermediate_size: int,
) -> list[dict[str, Any]]:
    layers = []
    for layer, width in enumerate(widths):
        prefix = f"model.layers.{layer}.mlp"
        expected = {
            f"{prefix}.gate_proj.weight": [width, hidden_size],
            f"{prefix}.up_proj.weight": [width, hidden_size],
            f"{prefix}.down_proj.weight": [hidden_size, width],
        }
        for key, shape in expected.items():
            observed = inventory.get(key)
            if observed is None:
                raise ValueError(f"missing physical MLP tensor: {key}")
            if observed["shape"] != shape:
                raise ValueError(f"shape mismatch for {key}: {observed['shape']} != {shape}")
            if dense_intermediate_size in observed["shape"] and width != dense_intermediate_size:
                raise ValueError(f"dense intermediate dimension survived in {key}: {observed['shape']}")
        layers.append({"layer": layer, "retained_width": width, "tensors": expected})
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.bundle / "config.json").read_text())
    metadata = json.loads((args.bundle / "substrate_metadata.json").read_text())
    hidden_size = int(config["hidden_size"])
    dense_intermediate_size = int(config["intermediate_size"])
    widths = expected_widths(args.mask)

    bundle_tensors, bundle_parameters, bundle_bytes = tensor_inventory(args.bundle)
    _, full_parameters, full_bytes = tensor_inventory(args.full_model)
    layer_shapes = validate_layer_shapes(
        bundle_tensors,
        widths,
        hidden_size,
        dense_intermediate_size,
    )

    metadata_widths = [int(metadata["isolation"]["kept_per_layer"][str(i)]) for i in range(len(widths))]
    if metadata_widths != widths:
        raise ValueError("substrate metadata widths do not match mask widths")

    retained_mlp_parameters = 3 * hidden_size * sum(widths)
    observed_mlp_parameters = sum(
        value["numel"]
        for key, value in bundle_tensors.items()
        if ".mlp." in key
    )
    if observed_mlp_parameters != retained_mlp_parameters:
        raise ValueError(
            f"MLP parameter mismatch: {observed_mlp_parameters} != {retained_mlp_parameters}"
        )

    report = {
        "status": "pass",
        "bundle": str(args.bundle),
        "bundle_format": metadata.get("format"),
        "mask": str(args.mask),
        "mask_sha256": sha256(args.mask),
        "model_safetensors_sha256": {
            path.name: sha256(path) for path in sorted(args.bundle.glob("*.safetensors"))
        },
        "layers": len(widths),
        "hidden_size": hidden_size,
        "dense_intermediate_size": dense_intermediate_size,
        "retained_channels": sum(widths),
        "dense_channels": len(widths) * dense_intermediate_size,
        "retained_channel_fraction": sum(widths) / (len(widths) * dense_intermediate_size),
        "retained_widths": widths,
        "retained_mlp_parameters": retained_mlp_parameters,
        "observed_mlp_parameters": observed_mlp_parameters,
        "bundle_total_parameters": bundle_parameters,
        "full_model_total_parameters": full_parameters,
        "retained_total_parameter_fraction": bundle_parameters / full_parameters,
        "bundle_safetensors_bytes": bundle_bytes,
        "full_model_safetensors_bytes": full_bytes,
        "retained_serialized_tensor_fraction": bundle_bytes / full_bytes,
        "dense_mlp_dimensions_absent": True,
        "metadata_widths_match_mask": True,
        "layer_shapes": layer_shapes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "layer_shapes"}, indent=2))


if __name__ == "__main__":
    main()
