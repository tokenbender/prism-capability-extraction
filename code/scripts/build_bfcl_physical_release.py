#!/usr/bin/env python3
"""Assemble the private, checksum-complete Issue #18 physical BFCL release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


ARTIFACT_ID = "prism-bfcl-mace-140875-physical-v1"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_REVISION = "cdebd3e01fab1d886f169ed87470a4e83d10a5b5"
ADAPTER_SHA256 = "5586cdd56acc4b0eac3bf34896dd85d235433227f8693ee912fae47f78447892"
MASK_SHA256 = "f83816319ec7d39cd4d95e8c8f44cecc139f23657233ccdd061bda1f71350567"
MASK_SET_SHA1 = "7492d1944be3320c59a49eaf2b57189badd3524a"
PAIRS_SHA256 = "f5466f7d1c17744bcfc2b6bc1e1647e2095c98762e5afd6fe7641dfcb6c1276e"
SCORER_SHA256 = "17ce35b8ca1633e6cf8a5da5de032d5671b33b184f52daf5b8f9a63f9b76bc12"
SOURCE_COMMIT = "07b46cdd41648dd83fbe8180752085bd59adafb5"
MODEL_SHA256 = "d0e78a244b01e0d0cce027f9bc65cea304315f522dc4359de2fc9fc4b5f2fb64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def portable(value: Any, experiment_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: portable(item, experiment_root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, experiment_root) for item in value]
    if isinstance(value, str) and value.startswith(str(experiment_root)):
        suffix = value.removeprefix(str(experiment_root)).lstrip("/")
        return f"<experiment-root>/{suffix}"
    return value


def sanitize_predictions(
    source: Path,
    destination: Path,
    category_by_id: dict[str, str],
) -> None:
    allowed = (
        "id",
        "prediction_text",
        "prediction_calls",
        "raw_correct",
        "normalized_correct",
    )
    rows = []
    for row in read_jsonl(source):
        clean = {key: row.get(key) for key in allowed if key in row}
        clean["category"] = category_by_id.get(str(row["id"]))
        rows.append(clean)
    write_jsonl(destination, rows)


def package_versions() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "accelerate", "safetensors", "peft", "numpy"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[name] = f"unavailable: {error}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def hardware_receipt() -> dict[str, Any]:
    receipt: dict[str, Any] = {}
    try:
        import torch

        receipt["cuda_available"] = torch.cuda.is_available()
        receipt["cuda_runtime"] = torch.version.cuda
        receipt["gpu_count"] = torch.cuda.device_count()
        receipt["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception as error:
        receipt["torch_probe_error"] = str(error)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        receipt["nvidia_smi"] = result.stdout.strip().splitlines()
    except Exception as error:
        receipt["nvidia_smi_error"] = str(error)
    return receipt


def model_card() -> str:
    return f"""# PRISM BFCL MACE physical substrate

This is the **private, verification-stage** physical artifact for
[`tokenbender/prism-capability-extraction` issue #18](https://github.com/tokenbender/prism-capability-extraction/issues/18).
It is not yet authorized for public redistribution; see `LICENSE_STATUS.md`.

## What is physically smaller

The artifact keeps full attention, embeddings, normalization, residual, rotary,
and LM-head components. It physically removes unselected Qwen3 MLP intermediate
channels from `gate_proj`, `up_proj`, and `down_proj` in every layer.

- selected MLP channels: **140,875 / 442,368 (31.846%)**
- physical parameters: **4,485,989,376 / 8,190,735,360 (54.769%)**
- physical safetensors bytes: **8,972,024,720 / 16,381,516,824 (54.769%)**
- physical weight SHA-256: `{MODEL_SHA256}`

`config.json` intentionally retains the dense parent's global
`intermediate_size=12288`; the actual non-uniform widths live in
`substrate_metadata.json`. Generic `AutoModel.from_pretrained` and config-only
FLOP estimators are therefore invalid. Use the included strict loader.

## Measured BFCL result

All counts use the frozen PRISM internal normalized structured matcher, not the
official BFCL leaderboard scorer.

| comparison | numerator | denominator | ratio |
|---|---:|---:|---:|
| same-environment dense merged parent | 598 | 671 | **89.1207%** |
| frozen logical mask | 598 | 600 | **99.6667%** |
| preserved live-adapter anchor | 598 | 672 | 88.9881% |
| historical base-only anchor | 598 | 664 | 90.0602% |

The first row is the primary physical-compression comparison. The latter two
are cross-run or cross-state provenance and must not be relabelled as same-state.

Physical category scores are deliberately preserved: `exec_simple 87/100`,
`java 58/100`, `javascript 31/50`, `live_simple 96/258`, `simple 284/400`,
and `sql 42/99`.

## Same-harness B200 benchmark

Five measured repeats followed one warmup on the same B200, BF16, eager
attention, batch size 8, and the same 24-example workload.

| metric | dense | physical | change |
|---|---:|---:|---:|
| load seconds | 4.634 | 4.322 | -6.72% |
| after-load allocated VRAM | 16.384 GB | 9.002 GB | -45.06% |
| peak allocated VRAM | 18.135 GB | 10.754 GB | -40.70% |
| prefill tokens/s | 32,787.5 | 23,040.6 | **-29.73%** |
| cached decode tokens/s | 257.28 | 277.45 | **+7.84%** |
| end-to-end generated tokens/s | 239.85 | 251.07 | **+4.68%** |

Performance is mixed: the irregular skinny MLPs reduce memory and help decode
on this workload, but hurt prefill. This is not a blanket speedup claim.

## Strict load check

Create a Python 3.12 environment with the versions in
`environment/requirements.txt`, then run:

```bash
python scripts/load_bfcl_physical_bundle.py load-check \\
  --bundle . \\
  --device cuda:0
```

The loader fails closed on format, Qwen model type, per-layer coverage, BF16
metadata/config/checkpoint/parameters, tensor keys and shapes, multi-EOS
generation configuration, and the frozen legacy-tokenizer contract.

The full scorer inputs are not redistributed in this private model package.
`environment/commands.md` records how to obtain the pinned BFCL source and run
evaluation after its terms have been reviewed.

## Provenance

- base: `Qwen/Qwen3-8B@{BASE_REVISION}`
- adapter snapshot: `TokenBender/circuit-discovery@{ADAPTER_REVISION}`
- candidate: `category_repair_java_r500_protect_tail_b140875_p10000`
- mask SHA-256: `{MASK_SHA256}`
- PRISM source commit: `{SOURCE_COMMIT}`
- frozen scorer SHA-256: `{SCORER_SHA256}`

See `MANIFEST.json`, `SHA256SUMS`, `ATTRIBUTION.md`, and the evaluation and
benchmark directories for the complete evidence ledger.
"""


def license_status() -> str:
    return """# License status

This repository is intentionally private while redistribution rights are
resolved.

- The Qwen3-8B base snapshot is distributed under Apache License 2.0; its
  license text is preserved under `licenses/`.
- The PRISM-authored loader, evaluator, and release tooling are Apache-2.0;
  the PRISM license text is preserved separately.
- The `b007` adapter snapshot used to create these merged weights does not yet
  carry complete public license/developer/training-provenance metadata.
- Prediction exports omit prompts, tool schemas, targets, reference answers,
  and raw BFCL rows while the data redistribution boundary is reviewed.

Do not make this artifact public or redistribute its weights until adapter
rights and the intended public data evidence have been documented explicitly.
"""


def attribution() -> str:
    return f"""# Attribution and modification notice

This physical model derives from `Qwen/Qwen3-8B` at revision
`{BASE_REVISION}` and from the PRISM `b007` adapter snapshot at revision
`{ADAPTER_REVISION}`.

PRISM modifications:

1. Merge the rank-32 BFCL adapter into the pinned BF16 base.
2. Retain the exact 140,875-channel Issue #12 selection.
3. Slice `gate_proj` and `up_proj` rows and matching `down_proj` columns in
   every MLP layer.
4. Serialize the resulting jagged per-layer MLPs with the full transformer
   scaffold unchanged.
5. Add an independently implemented PRISM loader and verification receipts.

No private `circuit-shotting` source code is included in this package.
"""


def commands() -> str:
    return f"""# Reproduction commands

The evaluation input is the pinned 1,007-row PRISM BFCL single-call slice with
SHA-256 `{PAIRS_SHA256}`. It is not bundled here pending a data-terms review.

Strict artifact load:

```bash
python scripts/load_bfcl_physical_bundle.py load-check --bundle . --device cuda:0
```

Full evaluation, once the pinned pairs file is available:

```bash
python scripts/load_bfcl_physical_bundle.py eval \\
  --bundle . \\
  --pairs /path/to/pairs.jsonl \\
  --output physical_predictions.jsonl \\
  --device cuda:0 \\
  --batch-size 8 \\
  --max-new-tokens 512 \\
  --bfcl-canonicalization-prompt \\
  --normalized
```

Benchmark contract:

```bash
python scripts/benchmark_bfcl_physical_bundle.py \\
  --mode bundle --bundle . --pairs /path/to/smoke24.jsonl \\
  --output physical.json --device cuda:0 --batch-size 8 \\
  --max-new-tokens 128 --decode-steps 32 --warmup 1 --repeats 5 \\
  --bfcl-canonicalization-prompt
```

Pinned software: Python 3.12, PyTorch 2.12.0+cu130, Transformers 4.57.6,
PEFT 0.19.1, Accelerate 1.14.0, Safetensors 0.8.0. Eager attention,
`fix_mistral_regex=False`, thinking disabled, and deterministic generation are
part of the experiment contract.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--frozen-logical", type=Path, required=True)
    parser.add_argument("--frozen-dense", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    root_files = (
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    for name in root_files:
        copy(args.bundle / name, args.output / name)
    link_or_copy(args.bundle / "model.safetensors", args.output / "model.safetensors")
    if sha256(args.output / "model.safetensors") != MODEL_SHA256:
        raise ValueError("physical model checksum does not match the frozen receipt")

    metadata = read_json(args.bundle / "substrate_metadata.json")
    metadata["artifact_id"] = ARTIFACT_ID
    metadata["visibility"] = "private_verification_stage"
    metadata["source_model"] = {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": BASE_REVISION,
        "revision_note": "reconstructed replay pin; original Issue #12 used a floating ID",
    }
    metadata["adapter"] = {
        "repo_id": "TokenBender/circuit-discovery",
        "repo_type": "dataset",
        "revision": ADAPTER_REVISION,
        "path": "bfcl/issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter",
        "weights_sha256": ADAPTER_SHA256,
    }
    metadata["candidate"] = {
        "id": "category_repair_java_r500_protect_tail_b140875_p10000",
        "mask_sha256": MASK_SHA256,
        "selected_set_sha1": MASK_SET_SHA1,
        "selected_channels": 140875,
        "dense_channels": 442368,
    }
    metadata["evaluation_contract"] = {
        "pairs_sha256": PAIRS_SHA256,
        "scorer_sha256": SCORER_SHA256,
        "batch_size": 8,
        "max_new_tokens": 512,
        "attention": "eager",
        "dtype": "bfloat16",
        "thinking": False,
        "bfcl_canonicalization_prompt": True,
        "metric": "PRISM internal normalized exact; not official BFCL leaderboard",
        "tokenizer_fix_mistral_regex": False,
    }
    metadata["source_repository"] = {
        "repo": "tokenbender/prism-capability-extraction",
        "branch": "issue18-camera-ready-physical-mace",
        "source_commit": SOURCE_COMMIT,
    }
    metadata.pop("attribution", None)
    write_json(args.output / "substrate_metadata.json", metadata)

    script_names = (
        "bfcl_direct_qwen3.py",
        "load_bfcl_physical_bundle.py",
        "inspect_bfcl_physical_bundle.py",
        "summarize_bfcl_physical_parity.py",
        "benchmark_bfcl_physical_bundle.py",
        "compare_bfcl_physical_benchmarks.py",
    )
    for name in script_names:
        copy(args.repo_root / "code" / "scripts" / name, args.output / "scripts" / name)

    copy(args.bundle / "LICENSE", args.output / "licenses" / "QWEN3-APACHE-2.0.txt")
    copy(args.repo_root / "LICENSE", args.output / "licenses" / "PRISM-APACHE-2.0.txt")
    (args.output / "README.md").write_text(model_card())
    (args.output / "LICENSE_STATUS.md").write_text(license_status())
    (args.output / "ATTRIBUTION.md").write_text(attribution())

    categories = {str(row["id"]): str(row["category"]) for row in read_jsonl(args.pairs)}
    parity = args.outputs_root / "parity"
    dense_merge = args.outputs_root / "dense_merge_parity"
    same_state = args.outputs_root / "same_state_physical_vs_dense"
    sanitize_predictions(
        parity / "physical_predictions.jsonl",
        args.output / "evaluation" / "physical_predictions.public.jsonl",
        categories,
    )
    sanitize_predictions(
        args.frozen_logical,
        args.output / "evaluation" / "logical_mask_predictions.public.jsonl",
        categories,
    )
    sanitize_predictions(
        dense_merge / "physical_predictions.jsonl",
        args.output / "evaluation" / "dense_replay_predictions.public.jsonl",
        categories,
    )
    sanitize_predictions(
        args.frozen_dense,
        args.output / "evaluation" / "live_adapter_dense_predictions.public.jsonl",
        categories,
    )

    evidence_files = {
        parity / "physical_summary.json": "evaluation/logical_vs_physical_summary.json",
        parity / "prediction_diff.jsonl": "evaluation/logical_vs_physical_diff.jsonl",
        parity / "category_scores.csv": "evaluation/logical_vs_physical_category_scores.csv",
        parity / "split_scores.csv": "evaluation/logical_vs_physical_split_scores.csv",
        dense_merge / "physical_summary.json": "evaluation/merge_vs_live_adapter_summary.json",
        dense_merge / "prediction_diff.jsonl": "evaluation/merge_vs_live_adapter_diff.jsonl",
        same_state / "physical_summary.json": "evaluation/dense_vs_physical_summary.json",
        same_state / "prediction_diff.jsonl": "evaluation/dense_vs_physical_diff.jsonl",
        same_state / "category_scores.csv": "evaluation/dense_vs_physical_category_scores.csv",
        same_state / "split_scores.csv": "evaluation/dense_vs_physical_split_scores.csv",
        args.outputs_root / "bundle_inspection.json": "evaluation/bundle_inspection.json",
        args.outputs_root / "smoke_prism_native_loader_final.jsonl": "evaluation/smoke_predictions.public.jsonl",
        args.outputs_root / "smoke_prism_native_loader_final.summary.json": "evaluation/smoke_summary.json",
        args.outputs_root / "final_phase_benchmark_dense.json": "benchmarks/dense.json",
        args.outputs_root / "final_phase_benchmark_physical.json": "benchmarks/physical.json",
        args.outputs_root / "final_phase_benchmark_comparison.json": "benchmarks/comparison.json",
    }
    json_suffixes = {".json"}
    for source, relative in evidence_files.items():
        destination = args.output / relative
        if source.suffix in json_suffixes:
            write_json(destination, portable(read_json(source), args.experiment_root))
        elif source.name.endswith("smoke_prism_native_loader_final.jsonl"):
            sanitize_predictions(source, destination, categories)
        else:
            copy(source, destination)

    environment = package_versions()
    environment["hardware"] = hardware_receipt()
    write_json(args.output / "environment" / "software_and_hardware.json", environment)
    (args.output / "environment" / "requirements.txt").write_text(
        "torch==2.12.0\n"
        "transformers==4.57.6\n"
        "peft==0.19.1\n"
        "accelerate==1.14.0\n"
        "safetensors==0.8.0\n"
        "numpy>=1.24\n"
    )
    (args.output / "environment" / "commands.md").write_text(commands())

    comparisons = [
        {
            "label": "same_environment_dense_parent_recovery",
            "numerator_state": "physical_bundle",
            "denominator_state": "same_environment_dense_merged_b007_replay",
            "numerator": 598,
            "denominator": 671,
            "rows": 1007,
            "ratio": 598 / 671,
            "primary": True,
        },
        {
            "label": "frozen_logical_mask_score_retention",
            "numerator_state": "physical_bundle",
            "denominator_state": "frozen_live_adapter_logical_mask",
            "numerator": 598,
            "denominator": 600,
            "rows": 1007,
            "ratio": 598 / 600,
            "primary": False,
        },
        {
            "label": "preserved_live_adapter_anchor_cross_run",
            "numerator_state": "physical_bundle",
            "denominator_state": "preserved_live_adapter_dense",
            "numerator": 598,
            "denominator": 672,
            "rows": 1007,
            "ratio": 598 / 672,
            "primary": False,
        },
        {
            "label": "historical_base_only_cross_state",
            "numerator_state": "physical_bundle",
            "denominator_state": "historical_unadapted_base",
            "numerator": 598,
            "denominator": 664,
            "rows": 1007,
            "ratio": 598 / 664,
            "primary": False,
        },
    ]

    payload_files = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            payload_files.append(
                {
                    "path": path.relative_to(args.output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "status": "private_verification_stage",
        "model_sha256": MODEL_SHA256,
        "comparisons": comparisons,
        "parameter_accounting": {
            "selected_mlp_channels": 140875,
            "dense_mlp_channels": 442368,
            "selected_mlp_channel_fraction": 140875 / 442368,
            "physical_parameters": 4485989376,
            "dense_parameters": 8190735360,
            "physical_parameter_fraction": 4485989376 / 8190735360,
            "physical_safetensors_bytes": 8972024720,
            "dense_safetensors_bytes": 16381516824,
            "physical_serialized_tensor_fraction": 8972024720 / 16381516824,
        },
        "exclusions": [
            "dense merged parent weights",
            "LoRA adapter",
            "runtime mask NPZ",
            "raw BFCL pairs, prompts, schemas, targets, and reference answers",
            "private or unlicensed reference runtime code",
            "credentials and caches",
        ],
        "files": payload_files,
    }
    write_json(args.output / "MANIFEST.json", manifest)

    checksum_paths = [
        path
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(args.output).as_posix()}\n"
            for path in checksum_paths
        )
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "files": len(checksum_paths) + 1,
                "bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
                "model_sha256": sha256(args.output / "model.safetensors"),
                "sha256sums_sha256": sha256(args.output / "SHA256SUMS"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
