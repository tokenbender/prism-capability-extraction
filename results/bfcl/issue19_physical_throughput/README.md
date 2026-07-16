# Issue #19 physical BFCL throughput receipt

The fastest tested quality-preserving setup is SDPA attention, lossless
gate/up packing with pad-128 physical widths, a hybrid SiLU-multiply dispatcher,
batch 64, and Transformers StaticCache automatic decode compilation on one
NVIDIA B200.

The dispatcher keeps rows below `2,048` on Torch so compiled decode remains
fast. It uses the custom Triton kernel only for large prefill rows. This is the
serving-runtime pattern that survived end-to-end and correctness gates; it is
not a claim that Triton is universally faster.

## Primary result

| Metric | Eager physical baseline | Selected setup | Change |
|---|---:|---:|---:|
| accepted generated tokens/s, median of 5 | `267.084` | `989.097` | `+270.33%` |
| generated slots/s, median of 5 | `453.726` | `2,196.184` | `+384.03%` |
| examples/s, median of 5 | `6.181` | `22.900` | `+270.49%` |
| full-workload seconds, median of 5 | `162.922` | `43.974` | `-73.01%` |
| per-batch latency p50 | `2,297.577 ms` | `2,694.282 ms` | larger winner batch |
| per-batch latency p95 | `4,332.383 ms` | `4,044.360 ms` | `-6.65%` |
| peak allocated VRAM | `18.128 GB` | `30.715 GB` | larger winner batch |
| full quality, normalized/raw exact | `599 / 586` | `600 / 587` | both pass `598 / 585` floor |

The primary metric is uninstrumented end-to-end generation over all `1007`
inputs in frozen order. Each setup uses its fastest full-quality-valid batch:
16 for eager/separate and 64 for the winner. Compile, load, repack, and first-use
costs are outside the steady-state median.

The winner beats the StaticCache+Torch finalist by `0.68%` (`989.097` versus
`982.416` accepted tokens/s). The custom kernel is therefore a narrow prefill
refinement on top of the much larger packing, SDPA, batching, and compiled-cache
gains.

## Why it is faster

- Gate and up projections are packed into one lossless GEMM, and zero padding
  aligns each jagged physical width to 128 channels.
- SDPA replaces eager attention.
- StaticCache lets Transformers compile decode while keeping prefill eager.
- The profiler records `12,411` kernel launches and `0.998 s` for the selected
  hybrid-activation batch-64 profile, versus `36,732` launches and `3.632 s`
  for the Torch-activation StaticCache control with generation compilation
  disabled.
- The Triton SiLU-multiply kernel is exact on the BF16 large-prefill
  microbenchmark and has `2.367x` median speedup across the tested 8,192- and
  32,768-row shapes. Always-on Triton is slower end to end, so the selected
  dispatcher does not use it for decode.

The three-repeat candidate-specific phase receipt reports `45,921.259` useful
prefill tokens/s. Compiled decode reports `3,290.795` input slots/s in
GPU-active device time. That decode number excludes host scheduling and is not
substituted for the primary end-to-end metric. Slot accounting is exact, no new
Dynamo graphs appear during measurement, and CUDA-event instrumentation adds an
estimated `0.281%` overhead.

## Setup cost

The selected runtime loads in `13.668 s`, including `8.936 s` of runtime
repacking. In the measured benchmark process, the first full generation takes
`178.654 s`, the next takes `44.072 s`, and the first-use compile/capture cost is
therefore `134.582 s`. StaticCache without generation compilation reaches only
`525.440` accepted tokens/s in the single no-compile control run; the cache
alone is not the result.

## Quality and numerical boundary

The winner scores `600/1007` normalized exact and `587/1007` raw exact at
`max_new_tokens=512`. Against the eager/separate batch-16 primary baseline, 41
prediction texts and parsed calls differ, with five normalized correctness
losses and six gains. A clean checkout independently passes the quality floor,
but it also demonstrates that exact greedy predictions can move across runs and
configurations, consistent with near-tied logits. Aggregate floor compliance is
proven; strict prediction identity is not.
That narrower numerical contract remains explicit in
[`#20`](https://github.com/tokenbender/prism-capability-extraction/issues/20).

## Evidence map

- `selection.json`: winner, baseline, exact metric, quality, overhead, hashes,
  and claim boundary.
- `benchmarks/comparison.json`: reusable direction-aware comparison.
- `benchmarks/stage1_matrix.json`: broad batch and execution-family sweep.
- `benchmarks/sdpa_packed128_dynamic_b128_full_r1.json`: full-1,007-example
  control showing why the apparently faster 256-example batch-128 screen did
  not survive workload expansion.
- `benchmarks/winner_hybrid_static_b64_r5.json`: uninstrumented primary result.
- `benchmarks/static_torch_finalist_b64_r5.json`: five-repeat Torch-activation
  finalist used to isolate the hybrid dispatcher's end-to-end contribution.
- `benchmarks/winner_phase_r3.json`: candidate-specific StaticCache prefill and
  compiled-decode instrumentation.
- `evaluation/quality_matrix.json`: full quality-valid batch map.
- `evaluation/winner_vs_primary_baseline_diff_summary.json`: every differing
  eval ID and correctness direction, without committing raw predictions.
- `profiling/`: eager, dynamic optimized, no-compile, and selected profiles.
- `negative_results.json`: failures, incompatibilities, quality drift, OOM, and
  pruned lanes.
- `SERVING_RUNTIME_AUDIT.md`: official-source vLLM/SGLang transplant audit and
  the boundary routed to
  [`#21`](https://github.com/tokenbender/prism-capability-extraction/issues/21).
- `MODEL_INPUT_RECEIPT.json`, `WANDB_RECEIPT.json`,
  `LOCAL_ARCHIVE_RECEIPT.json`, and `LIUM_TEARDOWN_RECEIPT.json`: input,
  remote, private-local, and executor preservation.

The W&B group is
[`ahm-rimer/prism-bfcl/issue19-physical-throughput`](https://wandb.ai/ahm-rimer/prism-bfcl/groups/issue19-physical-throughput).
Its three required runs are finished, and artifact
`issue19-physical-throughput-evidence:v0` has digest
`6277f72b33e9998d41b679369f37f2d6`.

## Claim boundary

This is the fastest setup in the tested candidate set for one frozen PRISM BFCL
workload, one physical jagged-Qwen artifact, one B200, and the recorded software
stack. It is not a universal model-serving speedup. Native continuous batching,
paged/Radix KV, engine CUDA graphs, prefix caching, and speculation require an
out-of-tree physical-Qwen vLLM/SGLang runner and remain in Issue #21 rather than
being credited here.
