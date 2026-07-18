#!/usr/bin/env python3
"""Train a BFCL LoRA while zero-isolating an explicit MLP-channel mask."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

import train_bfcl_masked_lora as trainer


def skip_mean_cache(
    model: Any,
    rows: Any,
    tokenizer: Any,
    args: Any,
    *,
    n_layers: int,
    d_ffn: int,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    """Zero isolation has no donor or mean-activation cache."""
    del model, rows, tokenizer, args, n_layers, d_ffn, dtype
    return {}


def install_zero_isolation_hooks(
    model: Any,
    mask: dict[int, torch.Tensor],
    means: dict[int, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> list[Any]:
    """Zero every unselected channel immediately before each MLP down projection."""
    del means, dtype
    hooks: list[Any] = []
    for layer_index, layer in enumerate(trainer.decoder_root(model).layers):
        keep = mask[layer_index]

        def hook(module: Any, args: tuple[torch.Tensor, ...], *, keep_mask: torch.Tensor = keep):
            del module
            hidden = args[0]
            selected = keep_mask.to(device=hidden.device)
            return (hidden * selected.to(dtype=hidden.dtype),)

        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    return hooks


def argument_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def main() -> None:
    trainer.build_mean_cache = skip_mean_cache
    trainer.install_mean_ablation_hooks = install_zero_isolation_hooks
    trainer.main()

    out_dir_value = argument_value("--out-dir")
    if out_dir_value is None:
        return
    summary_path = Path(out_dir_value) / "train_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    summary["intervention"] = {
        "kind": "zero_isolation",
        "hook": "decoder_layer.mlp.down_proj forward input",
        "donor_activations": False,
        "mean_activations": False,
        "gold_output_length_input": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
