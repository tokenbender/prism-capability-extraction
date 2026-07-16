#!/usr/bin/env python3
"""Prove that an arithmetic dense checkpoint is the exact scaled LoRA merge.

This gate deliberately compares model state, not predictions.  It loads a
pinned Qwen base and pinned PEFT adapter, captures the adapter's native scaling
dictionaries, multiplies those captured values by the historical scale, and
uses the same ``merge_and_unload(safe_merge=True)`` semantics as
``merge_lora_kl_sweep.py``.  Every reconstructed tensor must then be exactly
equal to the candidate dense checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import torch


RECEIPT_SCHEMA = "prism_arithmetic_merge_provenance_v1"
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for name in ("transformers", "peft", "accelerate", "safetensors", "huggingface_hub"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - environment receipt
            versions[name] = f"unavailable: {type(error).__name__}: {error}"
    return versions


def local_file_manifest(source: Path) -> dict[str, Any]:
    """Hash every model-relevant local file and bind the ordered manifest."""

    source = source.resolve()
    if source.is_file():
        files = [source]
        root = source.parent
    elif source.is_dir():
        files = sorted(
            path
            for path in source.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and ".cache" not in path.parts
            and "__pycache__" not in path.parts
        )
        root = source
    else:
        raise FileNotFoundError(source)
    if not files:
        raise ValueError(f"no model-relevant files found under {source}")

    rows = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = sha256(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
        total_bytes += size
    return {
        "root": str(source),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "manifest_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def describe_source(
    source: str,
    requested_revision: str | None,
) -> dict[str, Any]:
    """Resolve a local source by content or an HF source by immutable commit."""

    local = Path(source).expanduser()
    if local.exists():
        manifest = local_file_manifest(local)
        return {
            "kind": "local",
            "requested": source,
            "requested_revision": requested_revision,
            "resolved_revision": manifest["manifest_sha256"],
            "local_manifest": manifest,
        }
    if not requested_revision:
        raise ValueError(
            f"remote Hugging Face source {source!r} requires an explicit revision"
        )
    from huggingface_hub import HfApi

    info = HfApi().model_info(source, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not resolve a commit for {source}")
    return {
        "kind": "huggingface_model",
        "requested": source,
        "requested_revision": requested_revision,
        "resolved_revision": str(info.sha),
        "url": f"https://huggingface.co/{source}/tree/{info.sha}",
    }


def capture_lora_scaling(
    model: torch.nn.Module,
) -> list[tuple[str, Any, dict[str, float]]]:
    """Capture immutable native PEFT scaling values, including module names."""

    captured = []
    for module_name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and scaling:
            captured.append(
                (
                    module_name,
                    module,
                    {
                        adapter_name: float(value)
                        for adapter_name, value in scaling.items()
                    },
                )
            )
    if not captured:
        raise RuntimeError("no LoRA scaling dictionaries found")
    return captured


def apply_lora_scale(
    captured: list[tuple[str, Any, dict[str, float]]],
    scale: float,
) -> None:
    """Apply the historical captured-scaling multiplier exactly once."""

    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"scale must be finite and nonnegative, got {scale}")
    for _module_name, module, base_scaling in captured:
        for adapter_name, value in base_scaling.items():
            module.scaling[adapter_name] = value * scale


def scaling_receipt(
    captured: list[tuple[str, Any, dict[str, float]]],
    scale: float,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    rows = []
    digest = hashlib.sha256()
    for module_name, _module, base_scaling in captured:
        for adapter_name, value in sorted(base_scaling.items()):
            row = {
                "module": module_name,
                "adapter": adapter_name,
                "captured_scaling": value,
                "applied_scaling": value * scale,
            }
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(encoded.encode())
            digest.update(b"\n")
            rows.append(row)
    return {
        "module_count": len(captured),
        "entry_count": len(rows),
        "canonical_sha256": digest.hexdigest(),
        "samples": rows[:sample_limit],
        "sample_limit": sample_limit,
    }


def _max_abs_and_error_count(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[int, float]:
    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()
    unequal = expected_cpu != actual_cpu
    error_count = int(torch.count_nonzero(unequal).item())
    if error_count == 0:
        return 0, 0.0
    if expected_cpu.is_complex() or actual_cpu.is_complex():
        difference = (expected_cpu - actual_cpu).abs()
    elif expected_cpu.is_floating_point() or actual_cpu.is_floating_point():
        difference = (
            expected_cpu.to(torch.float64) - actual_cpu.to(torch.float64)
        ).abs()
    elif expected_cpu.dtype == torch.bool:
        difference = unequal.to(torch.float64)
    else:
        difference = (
            expected_cpu.to(torch.int64) - actual_cpu.to(torch.int64)
        ).abs()
    # A non-finite error is still a hard mismatch and should remain visible.
    maximum = float(difference.max().item())
    return error_count, maximum


def compare_state_dicts(
    reconstructed: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    mismatch_sample_limit: int = 50,
) -> dict[str, Any]:
    """Compare key, shape, dtype, and tensor identity without tolerances."""

    if mismatch_sample_limit < 0:
        raise ValueError("mismatch_sample_limit must be nonnegative")
    reconstructed_keys = set(reconstructed)
    candidate_keys = set(candidate)
    missing_keys = sorted(reconstructed_keys - candidate_keys)
    unexpected_keys = sorted(candidate_keys - reconstructed_keys)
    common_keys = sorted(reconstructed_keys & candidate_keys)

    shape_mismatches = []
    dtype_mismatches = []
    tensor_mismatches = []
    mismatch_samples = []
    compared_tensors = 0
    compared_elements = 0
    mismatched_elements = 0
    global_max_abs_error = 0.0

    def sample(row: dict[str, Any]) -> None:
        if len(mismatch_samples) < mismatch_sample_limit:
            mismatch_samples.append(row)

    for key in missing_keys:
        sample({"kind": "missing_key", "key": key})
    for key in unexpected_keys:
        sample({"kind": "unexpected_key", "key": key})

    for key in common_keys:
        expected = reconstructed[key]
        actual = candidate[key]
        if tuple(expected.shape) != tuple(actual.shape):
            row = {
                "key": key,
                "expected_shape": list(expected.shape),
                "candidate_shape": list(actual.shape),
            }
            shape_mismatches.append(row)
            sample({"kind": "shape", **row})
            continue
        if expected.dtype != actual.dtype:
            row = {
                "key": key,
                "expected_dtype": str(expected.dtype),
                "candidate_dtype": str(actual.dtype),
                "shape": list(expected.shape),
            }
            dtype_mismatches.append(row)
            sample({"kind": "dtype", **row})
            continue

        compared_tensors += 1
        compared_elements += expected.numel()
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        if torch.equal(expected_cpu, actual_cpu):
            continue
        error_count, max_abs_error = _max_abs_and_error_count(
            expected_cpu, actual_cpu
        )
        mismatched_elements += error_count
        if math.isnan(max_abs_error):
            global_max_abs_error = float("nan")
        elif not math.isnan(global_max_abs_error):
            global_max_abs_error = max(global_max_abs_error, max_abs_error)
        row = {
            "key": key,
            "shape": list(expected.shape),
            "dtype": str(expected.dtype),
            "error_count": error_count,
            "max_abs_error": max_abs_error,
        }
        tensor_mismatches.append(row)
        sample({"kind": "tensor", **row})

    exact = not (
        missing_keys
        or unexpected_keys
        or shape_mismatches
        or dtype_mismatches
        or tensor_mismatches
    )
    return {
        "status": "pass" if exact else "fail",
        "exact_tensor_identity": exact,
        "reconstructed_key_count": len(reconstructed_keys),
        "candidate_key_count": len(candidate_keys),
        "common_key_count": len(common_keys),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch_count": len(shape_mismatches),
        "dtype_mismatch_count": len(dtype_mismatches),
        "tensor_mismatch_count": len(tensor_mismatches),
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "tensor_mismatches": tensor_mismatches,
        "compared_tensor_count": compared_tensors,
        "compared_element_count": compared_elements,
        "mismatched_element_count": mismatched_elements,
        "global_max_abs_error": global_max_abs_error,
        "mismatch_samples": mismatch_samples,
        "mismatch_sample_limit": mismatch_sample_limit,
    }


def verify_merge_provenance(
    *,
    base_model: str,
    base_revision: str | None,
    adapter: str,
    adapter_revision: str | None,
    candidate: str,
    candidate_revision: str | None,
    scale: float,
    dtype: torch.dtype,
    device: str,
    mismatch_sample_limit: int,
) -> dict[str, Any]:
    """Reconstruct, merge, compare, and return the complete provenance receipt."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    sources = {
        "base": describe_source(base_model, base_revision),
        "adapter": describe_source(adapter, adapter_revision),
        "candidate": describe_source(candidate, candidate_revision),
    }
    # Resolve branch/tag names once, then load the immutable commit returned by
    # Hugging Face.  Local sources are already pinned by their content manifest.
    load_revisions = {
        name: (
            receipt["resolved_revision"]
            if receipt["kind"] == "huggingface_model"
            else None
        )
        for name, receipt in sources.items()
    }
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested_device}")

    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=load_revisions["base"],
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(requested_device).eval()
    adapted = PeftModel.from_pretrained(
        base,
        adapter,
        revision=load_revisions["adapter"],
    ).to(requested_device).eval()
    captured = capture_lora_scaling(adapted)
    applied_scaling = scaling_receipt(captured, scale)
    apply_lora_scale(captured, scale)
    load_seconds = time.perf_counter() - load_started

    merge_started = time.perf_counter()
    safe_merge_requested = True
    try:
        reconstructed = adapted.merge_and_unload(safe_merge=True)
        safe_merge_supported = True
    except TypeError:
        reconstructed = adapted.merge_and_unload()
        safe_merge_supported = False
    reconstructed.eval()
    merge_seconds = time.perf_counter() - merge_started

    candidate_started = time.perf_counter()
    candidate_model = AutoModelForCausalLM.from_pretrained(
        candidate,
        revision=load_revisions["candidate"],
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(requested_device).eval()
    candidate_load_seconds = time.perf_counter() - candidate_started

    compare_started = time.perf_counter()
    comparison = compare_state_dicts(
        reconstructed.state_dict(),
        candidate_model.state_dict(),
        mismatch_sample_limit=mismatch_sample_limit,
    )
    compare_seconds = time.perf_counter() - compare_started

    return {
        "schema": RECEIPT_SCHEMA,
        "status": "pass" if comparison["exact_tensor_identity"] else "fail",
        "hard_gate": {
            "criterion": (
                "identical state keys, shapes, dtypes, and torch.equal tensors"
            ),
            "exact_tensor_identity": comparison["exact_tensor_identity"],
        },
        "sources": sources,
        "load_revisions": load_revisions,
        "merge": {
            "scale": scale,
            "scaling_semantics": (
                "capture each native PEFT scaling value, multiply the captured "
                "value by scale exactly once, then merge_and_unload"
            ),
            "safe_merge_requested": safe_merge_requested,
            "safe_merge_supported": safe_merge_supported,
            "applied_scaling": applied_scaling,
            "reconstructed_model_class": type(reconstructed).__name__,
            "candidate_model_class": type(candidate_model).__name__,
        },
        "runtime": {
            "device": str(requested_device),
            "dtype": str(dtype),
            "attention_implementation": "eager",
        },
        "package_versions": package_versions(),
        "comparison": comparison,
        "timings_seconds": {
            "base_and_adapter_load": load_seconds,
            "merge": merge_seconds,
            "candidate_load": candidate_load_seconds,
            "comparison": compare_seconds,
            "total": time.perf_counter() - started,
        },
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-revision")
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mismatch-sample-limit", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    initial = {
        "schema": RECEIPT_SCHEMA,
        "status": "running",
        "requested": {
            "base_model": args.base_model,
            "base_revision": args.base_revision,
            "adapter": args.adapter,
            "adapter_revision": args.adapter_revision,
            "candidate": args.candidate,
            "candidate_revision": args.candidate_revision,
            "scale": args.scale,
            "dtype": args.dtype,
            "device": args.device,
        },
        "package_versions": package_versions(),
    }
    try:
        receipt = verify_merge_provenance(
            base_model=args.base_model,
            base_revision=args.base_revision,
            adapter=args.adapter,
            adapter_revision=args.adapter_revision,
            candidate=args.candidate,
            candidate_revision=args.candidate_revision,
            scale=args.scale,
            dtype=dtype,
            device=args.device,
            mismatch_sample_limit=args.mismatch_sample_limit,
        )
        receipt["requested"] = initial["requested"]
    except Exception as error:
        initial.update(
            {
                "status": "error",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _write_receipt(args.output, initial)
        print(json.dumps(initial, indent=2))
        raise

    _write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
