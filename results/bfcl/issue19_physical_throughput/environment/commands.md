# Issue #19 reproduction commands

These commands assume a clean checkout at implementation commit
`8755cf454de011f5996f9158264ca9c7ad3b3318`, the verified Issue #18 physical
bundle, the frozen pair file, one CUDA-visible B200, and the package versions in
`requirements.txt`.

```bash
python code/scripts/load_bfcl_physical_bundle.py load-check \
  --bundle <physical-bundle> \
  --device cuda:0 \
  --attention-implementation sdpa \
  --mlp-implementation packed_gate_up \
  --activation-implementation hybrid \
  --hybrid-activation-threshold-rows 2048 \
  --width-alignment 128
```

The primary five-repeat benchmark is uninstrumented. Its metric includes the
complete fixed-order `1007`-example generation workload after two warmups.

```bash
python code/scripts/benchmark_bfcl_physical_throughput.py \
  --bundle <physical-bundle> \
  --pairs <frozen-pairs.jsonl> \
  --output winner_hybrid_static_b64_r5.json \
  --device cuda:0 \
  --attention-implementation sdpa \
  --mlp-implementation packed_gate_up \
  --activation-implementation hybrid \
  --hybrid-activation-threshold-rows 2048 \
  --width-alignment 128 \
  --cache-implementation static \
  --batch-size 64 \
  --max-new-tokens 128 \
  --decode-steps 32 \
  --warmup 2 \
  --repeats 5 \
  --phase-repeats 1 \
  --limit 1007 \
  --bfcl-canonicalization-prompt
```

The separate phase receipt uses an actual StaticCache for eager prefill and
CUDA events around the already-compiled Transformers decode callable. It keeps
that device-only timing out of the primary end-to-end claim.

```bash
TORCH_LOGS=recompiles \
python code/scripts/benchmark_bfcl_physical_throughput.py \
  --bundle <physical-bundle> \
  --pairs <frozen-pairs.jsonl> \
  --output winner_phase_r3.json \
  --device cuda:0 \
  --attention-implementation sdpa \
  --mlp-implementation packed_gate_up \
  --activation-implementation hybrid \
  --hybrid-activation-threshold-rows 2048 \
  --width-alignment 128 \
  --cache-implementation static \
  --instrument-compiled-decode \
  --batch-size 64 \
  --max-new-tokens 128 \
  --decode-steps 32 \
  --warmup 2 \
  --repeats 3 \
  --phase-repeats 3 \
  --limit 1007 \
  --bfcl-canonicalization-prompt
```

The full correctness gate uses the Issue #18 generation length rather than the
performance length.

```bash
python code/scripts/load_bfcl_physical_bundle.py eval \
  --bundle <physical-bundle> \
  --pairs <frozen-pairs.jsonl> \
  --output winner_predictions.jsonl \
  --device cuda:0 \
  --batch-size 64 \
  --max-new-tokens 512 \
  --bfcl-canonicalization-prompt \
  --normalized \
  --attention-implementation sdpa \
  --mlp-implementation packed_gate_up \
  --activation-implementation hybrid \
  --hybrid-activation-threshold-rows 2048 \
  --width-alignment 128 \
  --cache-implementation static
```

```bash
python code/scripts/profile_bfcl_physical_runtime.py \
  --bundle <physical-bundle> \
  --pairs <frozen-pairs.jsonl> \
  --output winner_profile.json \
  --device cuda:0 \
  --attention-implementation sdpa \
  --mlp-implementation packed_gate_up \
  --activation-implementation hybrid \
  --hybrid-activation-threshold-rows 2048 \
  --width-alignment 128 \
  --cache-implementation static \
  --batch-size 64 \
  --max-new-tokens 16 \
  --top-k 30
```

```bash
python code/scripts/benchmark_bfcl_silu_mul.py \
  --bundle <physical-bundle> \
  --output silu_mul_large_prefill.json \
  --device cuda:0 \
  --dtype bf16 \
  --rows 8192 32768 \
  --alignment 128 \
  --warmups 10 \
  --repeats 50 \
  --max-elements 3000000000 \
  --seed 1900 \
  --triton required
```

```bash
python code/scripts/compare_bfcl_physical_throughput.py \
  --baseline benchmarks/eager_separate_b16_r5.json \
  --candidate eager_packed128_b64=benchmarks/eager_packed128_b64_r5.json \
  --candidate sdpa_packed128_b64_dynamic=benchmarks/sdpa_packed128_dynamic_b64_r5.json \
  --candidate winner_hybrid_static_b64=benchmarks/winner_combined.json \
  --output benchmarks/comparison.json
```

`winner_combined.json` retains the uninstrumented five-repeat end-to-end fields
and overlays only the three-repeat candidate-specific phase fields. Its
`derived_receipt` object binds both source hashes.
