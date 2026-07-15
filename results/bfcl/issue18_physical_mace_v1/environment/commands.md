# Reproduction commands

The evaluation input is the pinned 1,007-row PRISM BFCL single-call slice with
SHA-256 `f5466f7d1c17744bcfc2b6bc1e1647e2095c98762e5afd6fe7641dfcb6c1276e`. It is not bundled here pending a data-terms review.

Strict artifact load:

```bash
python scripts/load_bfcl_physical_bundle.py load-check --bundle . --device cuda:0
```

Full evaluation, once the pinned pairs file is available:

```bash
python scripts/load_bfcl_physical_bundle.py eval \
  --bundle . \
  --pairs /path/to/pairs.jsonl \
  --output physical_predictions.jsonl \
  --device cuda:0 \
  --batch-size 8 \
  --max-new-tokens 512 \
  --bfcl-canonicalization-prompt \
  --normalized
```

Benchmark contract:

```bash
python scripts/benchmark_bfcl_physical_bundle.py \
  --mode bundle --bundle . --pairs /path/to/smoke24.jsonl \
  --output physical.json --device cuda:0 --batch-size 8 \
  --max-new-tokens 128 --decode-steps 32 --warmup 1 --repeats 5 \
  --bfcl-canonicalization-prompt
```

Pinned software: Python 3.12, PyTorch 2.12.0+cu130, Transformers 4.57.6,
PEFT 0.19.1, Accelerate 1.14.0, Safetensors 0.8.0. Eager attention,
`fix_mistral_regex=False`, thinking disabled, and deterministic generation are
part of the experiment contract.
