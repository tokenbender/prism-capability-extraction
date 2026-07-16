# Prism: Unlocking Language Model Capability Extraction

Authors: Abhishek Mishra, Krishna Pagare

A trained language model holds many capabilities at once, while a deployment
usually asks it to exercise only one. Prism asks whether a named capability can
be made to run through a sparse set of MLP channels while the surrounding
transformer scaffold stays fixed and the rest of the MLP is switched off.

This repository is the public code and artifact index for the paper. It
contains intervention scripts and small receipts, plus strict loaders and
evidence ledgers for standalone physical arithmetic and function-calling
substrates. The model weights remain in private verification mirrors while
their publication receipts and redistribution boundaries are completed.

<img src="paper/assets/figures/f1.svg" alt="Prism attribution and sparse-mask extraction diagram" width="100%">

## The extraction contract

Attribution is not enough. A channel can correlate with a behavior without
being sufficient to run it. Prism tests sufficiency by keeping a candidate mask,
disabling the complement, and measuring how much of the original behavior
survives under that intervention.

```mermaid
flowchart LR
    A["target behavior"] --> B["score MLP channels"]
    B --> C["keep top-k mask"]
    C --> D["disable complement"]
    D --> E["measure recovery"]
    F["collimation adapter"] --> B
    F --> C
```

Collimation is the behavior-preserving step that changes where a capability is
carried internally. The key distinction is timing. If collimation runs before
attribution, it can move the sparsity frontier because the mask is found after
the representation changes. If it runs after attribution, the mask is already
fixed, so the adapter can only route more behavior through the surviving
channels.

| regime | order | what it can change | primary camera-ready use |
|---|---|---|---|
| Pre-attribution collimation | collimate -> attribute -> mask | which channels carry the capability | arithmetic |
| Post-attribution collimation | attribute -> mask -> collimate | how much behavior fits through fixed channels | function calling |

## Main release results

| capability | model | intervention claim | released evidence |
|---|---|---|---|
| Two-digit addition — historical causal intervention | `Qwen2.5-Math-1.5B` | Under the historical donor-patched evaluator, which restores discarded prompt-token MLP activations from a dense counterfactual pass and generates for the gold-answer length, the 12,661-channel mask reaches `1,370/1,500` (`91.33%`) autoregressive exact. This is causal-intervention evidence, not standalone physical compression. | historical arithmetic masks, training summaries, and evaluator receipts |
| Two-digit addition — standalone physical substrate | `Qwen2.5-Math-1.5B` | The bounded Issue #22 search finds an independently loadable physical MACE with 250,237/250,880 MLP channels. It scores `1,386/1,500` (`92.40%`), recovers `1,378/1,406` (`98.01%`) of its matching dense parent, and retains `1,376/1,388` (`99.14%`) of its logical-zero correctness. Its fastest tested quality-valid one-B200 setup reaches 10,208.96 examples/s, `3.4417x` the canonical eager physical setup and `0.3474%` slower than the equally optimized dense parent. | standalone predictions/diffs, physical-loader and size receipts, bounded search ledger, repeated B200 benchmarks, profiler estimates, verified W&B evidence, and a private ModelScope artifact with fresh-download checksum and clean-load verification |
| Function calling | `Qwen3-8B` | A built and privately verified jagged-MLP substrate retains 140,875 channels, 54.77% of total parameters, and `598/671` (`89.12%`) of its same-environment dense parent's normalized-exact score. | physical weights, strict loader, full predictions/diffs, size accounting, and repeated B200 benchmark |

The two arithmetic rows are different experiments and should not be collapsed
into one compression claim. The historical 12,661-channel result depends on a
dense donor pass and gold-answer-length generation. The Issue #22 artifact is
standalone and donor-free, but its parity-valid physical form removes only 643
of 250,880 MLP channels (`0.2563%`), so global minimality and substantial
compression are not claimed. The Issue #22 runtime number is pretokenized,
synchronized generation throughput for the frozen 1,500-row workload and
recorded stack, not a universal serving or dense-speedup claim.

The function-calling count uses PRISM's pinned internal normalized structured
matcher, not the official BFCL leaderboard. The physical artifact is
`598/1007`; its same-environment merged parent is `671/1007`. The historical
`598/664 = 90.06%` comparison crosses model states and is not the primary
compression ratio.

EN-PT translation artifacts remain available as historical provenance, but
translation is not part of the primary camera-ready claim or contribution
surface.

The paper PDF and figure assets are in `paper/`. The claim-to-artifact index is
`docs/ARTIFACT_MANIFEST.json`.

## Repository map

| path | role |
|---|---|
| `code/` | runnable scripts for arithmetic, translation, and BFCL/function-calling experiments |
| `code/src/circuit_tracing/` | shared arithmetic circuit-tracing utilities |
| `code/scripts/` | BFCL filtering, scoring, conditioning, and evaluation scripts |
| `code/configs/` | sanitized configs for released runs |
| `results/` | small result receipts included in git |
| `docs/` | release boundary, data-source notes, terminology, and artifact manifest |
| `paper/` | paper PDF and figure/image assets |

Public legacy checkpoints, generated datasets, raw attribution arrays, adapters,
and full-model outputs are hosted on Hugging Face rather than committed to git.
The Issue #18 BFCL and Issue #22 arithmetic physical substrates are documented
exceptions: their weights are privately preserved on ModelScope rather than
committed to this repository.

The Issue #18 physical BFCL verification weights are mirrored privately at
[`tokenbender/prism-bfcl-mace-140875-physical-restricted-v2`](https://modelscope.ai/models/tokenbender/prism-bfcl-mace-140875-physical-restricted-v2).
Their public release remains gated on adapter provenance and a verified Hugging
Face publication receipt. The private repository's `master` revision is mutable;
the ModelScope verification receipt plus the model SHA-256 form the proof surface.
ModelScope exposes no license grant for that restricted repository. The git-sized evidence is in
`results/bfcl/issue18_physical_mace_v1/`.

The Issue #22 arithmetic physical artifact has been uploaded privately to
[`tokenbender/prism-arithmetic-mace-physical-v1`](https://modelscope.ai/models/tokenbender/prism-arithmetic-mace-physical-v1).
The immutable publication tag is `issue22-physical-v1.2`.
Its published transport is locally bound to checkpoint-index SHA-256
`9af6cd176de17c96c1f75d2c66c19d320d0d849c6d5dc61443c9973f1db30095`
and shard-manifest SHA-256
`571a1e50bdddb5dc289367663f24090c4225725cbac4006c0c009bc08cbc5521`.
The fresh tag download exactly matches all 44 uploaded payload files: 43 files
passed the payload's `SHA256SUMS`, and that checksum manifest is separately
bound by SHA-256. The downloaded model passed a strict CPU clean load with no
donor model or dense MLP allocation. The W&B evidence is verified under group
[`issue22-arithmetic-physical-throughput`](https://wandb.ai/ahm-rimer/prism-arithmetic/groups/issue22-arithmetic-physical-throughput),
artifact `issue22-arithmetic-physical-throughput-evidence:v0`. The git-sized
evidence is in `results/arithmetic/issue22_physical_mace_throughput/`.

## Install

Use Python 3.12+.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./code
```

The standard-library venv path also works.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./code
```

Most experiment reruns require GPU hardware and public model downloads. The
default circuit-tracing tests use `Qwen/Qwen3-0.6B` and CUDA when available.

```bash
cd code
python -m pytest src/circuit_tracing/test_pipeline.py -v
```

Slow edge-attribution and end-to-end tests are opt-in.

```bash
PRISM_RUN_SLOW_TESTS=1 python -m pytest src/circuit_tracing/test_pipeline.py -v
```

## Reproduce

Start with `REPRODUCE.md` for task-level commands. The short version is:

| task family | entry point |
|---|---|
| Arithmetic extraction | `code/src/circuit_tracing/`, `code/train_lora_2digit_kl.py`, `code/scripts/run_issue22_b200_pipeline.sh`, and the Issue #22 physical loader/evaluation scripts |
| Translation rescue | `code/build_ntrex_en2pt_jsonl.py`, `code/train_masked_kl_conditioning.py`, and translation evaluation scripts |
| Function calling | `code/scripts/load_bfcl_physical_bundle.py`, `code/scripts/bfcl_direct_qwen3.py`, and the Issue #18 receipts |

The repository keeps small receipts in `results/`. Public Hugging Face artifacts
are pinned by immutable revision SHA in the manifest. The private Issue #18
ModelScope mirror is bound by its completed verification receipt and model
hash. The private Issue #22 ModelScope artifact is bound by its immutable tag,
fresh-download receipt, and checkpoint shard manifest.

## Public artifacts

| artifact | repository | revision |
|---|---|---|
| Arithmetic generated data, masks, and run receipts | [TokenBender/circuit-discovery](https://huggingface.co/datasets/TokenBender/circuit-discovery) | `4b9fb53fef92550042d8576fe011e99270fdca8b` |
| Arithmetic checkpoints and adapters | [TokenBender/circuit-discovery](https://huggingface.co/TokenBender/circuit-discovery) | `f75ca2f123ce6aaca0e8096918df1ddb34b5d546` |
| EN-PT generated data and run artifacts | [TokenBender/synth-data-en-pt-circuit](https://huggingface.co/datasets/TokenBender/synth-data-en-pt-circuit) | `36ee2512bcabf32f224c34792db4fb1907d711c3` |
| EN-PT adapters and masks | [Occupying-Mars/hy-lora-conditions](https://huggingface.co/Occupying-Mars/hy-lora-conditions) | `8139bf31538727c87c04f9a88b0b0ccaeacb8832` |
| BFCL data and reproduction receipts | [Occupying-Mars/issue49-bfcl-repro-artifacts](https://huggingface.co/datasets/Occupying-Mars/issue49-bfcl-repro-artifacts) | `303db0bddcfb04bebaf07ab4a4dc4c089240c545` |
| BFCL k160 adapter | [Occupying-Mars/issue49-k160-r32-len1024-adapter](https://huggingface.co/Occupying-Mars/issue49-k160-r32-len1024-adapter) | `e5104eee6e9dd0fff11f377b743330429970d672` |
| BFCL k240 adapter | [Occupying-Mars/issue49-k240-r16-adapter](https://huggingface.co/Occupying-Mars/issue49-k240-r16-adapter) | `b9018cc3090b856df701240fd73f9f98c627917c` |
| BFCL full-model reproduction | [TokenBender/issue51-r32-more-online-codex-repro-551-full](https://huggingface.co/TokenBender/issue51-r32-more-online-codex-repro-551-full) | `ff4daac9e49a8f927153c9a04daa9faba2fb5a66` |

## License

This repository is licensed under the Apache License, Version 2.0. See
`LICENSE`.
