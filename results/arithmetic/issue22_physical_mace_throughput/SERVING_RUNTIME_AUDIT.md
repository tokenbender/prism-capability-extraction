# Serving runtime audit

## Accepted runtime

The accepted Issue #22 winner is an independent PRISM/Transformers execution
path for the exact physical Qwen2 substrate:

- one NVIDIA B200, TP1;
- BF16;
- SDPA attention;
- packed gate/up projections;
- per-layer physical widths padded to an execution alignment of 16 without
  reconstructing discarded model parameters;
- dynamic KV cache;
- Triton SiLU-multiply;
- batch 1,500;
- one warmup and five uninstrumented measured repeats.

The model loads from strict sharded safetensors without the dense MLP tensors,
original adapter, runtime mask, donor model, or source checkout. The loader
reconstructs only the proven tied `lm_head.weight` alias from
`model.embed_tokens.weight`; it adds zero independent parameters.

## Benchmark boundary

The primary timing contract is synchronized, pretokenized `model.generate`
time. Decoding text and scoring predictions are excluded. Compilation, model
load, and warmup are reported separately rather than amortized invisibly.

| Runtime | Batch | Correct | Median examples/s | Median generated tokens/s |
|---|---:|---:|---:|---:|
| Canonical eager physical | `512` | `1,386/1,500` | `2,966.286` | — |
| Optimized physical winner | `1,500` | `1,395/1,500` | `10,208.963` | `81,671.707` |
| Equally optimized dense parent | `1,500` | `1,403/1,500` | `10,244.551` | `81,956.406` |

The optimized physical setup is `3.4417x` the canonical physical setup. That is
a full-setup improvement, including batch saturation, SDPA, packing, alignment,
and Triton activation. It is not a kernel-only comparison.

The physical winner reaches `99.6526%` of equally optimized dense throughput
and is `0.3474%` slower. No dense speedup is claimed.

## Quality boundary

Physical compaction and runtime changes alter BF16 GEMM shapes and can change
greedy decisions. Runtime candidates therefore advance only after full scoring;
microbenchmark speed is insufficient.

The cheap 1,024-row gate demonstrates this boundary:

- packed-64 physical Torch reached `8,321.862` examples/s but scored
  `952/1,024`, one below the `953` floor;
- packed-16 physical Triton reached `9,063.467` examples/s in the diversity
  sweep but also scored `952/1,024`;
- separate physical SDPA was slower at `4,290.434` examples/s but passed
  `954/1,024`.

The final batch-1,500 Triton winner passed the frozen `1,386/1,500` floor with
`1,395/1,500`, and all five repeats had the same prediction hash.

## vLLM and SGLang boundary

vLLM and SGLang optimization ideas informed the candidate race: attention
backend selection, cache policy, graph/compile controls, packing, and batch
saturation were considered. No stock engine-native result is credited.

The accepted artifact has non-uniform per-layer Qwen MLP widths. The stock
engine paths audited for this run did not expose a release-preserving contract
for those jagged widths. A path that pads or reconstructs full dense MLP
parameters would violate the physical-artifact contract. A valid engine-native
follow-up must therefore:

1. implement a custom jagged-Qwen model and weight loader;
2. load the physical shards without allocating dense MLP parameters;
3. preserve tied-weight reconstruction and tokenizer behavior;
4. rerun the complete frozen quality comparison;
5. benchmark against an equally optimized dense engine path.

Until that exists, this result is a high-throughput custom Transformers runtime,
not native vLLM or SGLang serving.

Relevant upstream optimization surfaces:

- vLLM CUDA graph design:
  <https://docs.vllm.ai/en/latest/design/cuda_graphs/>
- SGLang server arguments:
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md>

## Profiling boundary

The winner profile estimates:

- `64,457,592,470,932` supported-op FLOPs;
- `362.502` estimated TFLOP/s;
- `16.111%` estimated executed MFU;
- `21.320%` active-time MFU diagnostic;
- `75.568%` summed-kernel-active-to-wall diagnostic;
- `8,914` kernel launches.

The denominator is 2,250 TFLOP/s BF16 dense per B200. FLOPs come from
`torch.profiler(with_flops=True)` and cover supported operators only. Summed
kernel durations may double-count overlap. None of these values are hardware
counters.

Sampled utilization peaked at 99% in the finalist queue. Sampler means include
loads, compilation, and idle queue boundaries; they must not be reported as
active-generation utilization.

## Not demonstrated

- continuous multi-tenant request batching;
- streaming latency under concurrent arrival;
- native vLLM or SGLang integration;
- multi-GPU tensor or pipeline parallelism;
- CUDA-graph capture of the accepted final winner;
- lower-precision quality parity;
- throughput beyond the frozen short-prompt, eight-token-cap arithmetic
  workload;
- a speedup over equally optimized dense execution.

"Fastest" means fastest tested quality-valid configuration for this frozen
artifact, workload, B200, and recorded software stack.
