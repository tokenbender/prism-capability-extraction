# Reproduction Notes

This repository contains code and small receipts for the released Prism
experiments. Large model weights, generated datasets, raw attribution files, and
full-model artifacts are hosted on Hugging Face and referenced from
`docs/ARTIFACT_MANIFEST.json`.

## Environment

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./code
```

Install CUDA-specific `torch` builds as appropriate for your hardware.

## Arithmetic

The arithmetic task uses generated two-digit addition pairs.

Representative commands:

```bash
cd code
python3 train_lora_2digit_kl.py --model Qwen/Qwen2.5-Math-1.5B --out-dir runs/arithmetic_lora
python3 evaluate_full_answer_masks.py --model Qwen/Qwen2.5-Math-1.5B --help
python3 evaluate_full_answer_generation_masks.py --model Qwen/Qwen2.5-Math-1.5B --help
```

Included receipts are under `results/arithmetic/`.

## Translation

Build the held-out NTREX EN-PT evaluation file:

```bash
cd code
python3 build_ntrex_en2pt_jsonl.py
```

Representative commands:

```bash
python3 train_masked_kl_conditioning.py --help
python3 evaluate_translation_adapter_masks.py --help
python3 evaluate_translation_masks.py --help
```

Included receipts are under `results/translation/`.

## BFCL / Function Calling

Download and prepare public BFCL v3 single-call rows:

```bash
cd code
python3 scripts/bfcl_direct_qwen3.py download-bfcl-single-call --help
```

Filter public ToolMind and Argilla/APIGen-style data into strict
BFCL-compatible single-call rows:

```bash
python3 scripts/filter_toolmind_bfcl_strict.py --help
python3 scripts/filter_argilla_apigen_bfcl_strict.py --help
python3 scripts/build_bfcl_strict_10k_mix.py --help
```

Condition and evaluate:

```bash
python3 scripts/train_bfcl_masked_lora.py --help
python3 scripts/train_bfcl_masked_policy_distill.py --help
python3 scripts/train_bfcl_prime_opd_sampled_lora.py --help
python3 scripts/bfcl_direct_qwen3.py eval-mask --help
```


### Latest verified BFCL result

The latest end-to-end result is the Issue #19 physical substrate: lossless
gate/up packing with pad-128 physical widths, SDPA, a hybrid SiLU dispatcher,
batch 64, and compiled StaticCache decode on one NVIDIA B200. It sustained
`989.097` accepted generated tokens/s over the frozen 1,007-example workload,
versus `267.084` for the eager physical baseline (`+270.33%`), while both passed
the recorded full-quality floor. The claim is bounded to the recorded model,
workload, hardware, and software stack.

The exact benchmark, quality, profiling, comparison, and microbenchmark commands
are in
[`results/bfcl/issue19_physical_throughput/environment/commands.md`](results/bfcl/issue19_physical_throughput/environment/commands.md).
The result summary, environment lock, hashes, negative results, and preservation
receipts are in
[`results/bfcl/issue19_physical_throughput/`](results/bfcl/issue19_physical_throughput/).

Small BFCL receipts are under `results/bfcl/`. Complete BFCL data, adapters,
and the full-model reproduction are available from the Hugging Face artifact
repositories listed in `docs/ARTIFACT_MANIFEST.json`.

## Sparse-inference failure mining

The [`sparse-inference-failure-atlas/`](sparse-inference-failure-atlas/) package
contains the source-linked report, frozen corpus, deterministic consolidation
and ranking scripts, publication figures, and exact rebuild checks.

Start with its
[`Reproduce the corpus and figures`](sparse-inference-failure-atlas/README.md#reproduce-the-corpus-and-figures)
section. The live-web discovery step is intentionally not replayed: normalized
source snapshots are frozen inputs, while schema validation, lineage joining,
quality gates, ranking, atlas construction, and aggregate figures rebuild
offline.
