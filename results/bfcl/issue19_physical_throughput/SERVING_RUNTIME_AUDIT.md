# vLLM and SGLang optimization audit for Issue #19

## Scope

This audit asks which vLLM and SGLang ideas can improve the frozen Issue #19
workload without reconstructing the removed dense MLP channels, changing BF16
weights, reordering the 1007 inputs, or weakening the `598/585` quality floor.
It distinguishes small lossless changes that can be tested in the current
Transformers runtime from changes that require a native serving model runner.

The current physical model has 36 different MLP widths. Stock Qwen model
constructors in both engines receive one global intermediate size, so neither
engine can load this artifact as an ordinary Qwen checkpoint. A native engine
test first needs a small out-of-tree physical-Qwen model that constructs each
layer from its recorded width. That work is tracked in
[`#21`](https://github.com/tokenbender/prism-capability-extraction/issues/21).

## Changes tested in Issue #19

| Idea from serving runtimes | Mechanism | Current-runtime test | Claim boundary |
|---|---|---|---|
| Packed gate and up projection | replace two GEMMs with one projection whose output halves are gate and up | `packed_gate_up`, with exact and aligned widths | runtime-only repack; active weights remain exact and padding is zero |
| Fused SiLU and multiply | consume the packed tensor in one pointwise kernel | optional Triton selector, benchmarked only after profile and parity gates | mathematically equivalent but not assumed bitwise identical |
| Decode graphs through a static KV cache | keep decode shapes compileable while leaving prefill eager | Transformers `StaticCache` auto-compile, plus explicit no-compile and dynamic-shape controls | compile/capture overhead and recompiles are separate from steady state |
| Optimized attention | replace eager attention with exact SDPA | `sdpa` selector | full BFCL replay decides whether BF16 numerical changes are acceptable |

[vLLM's Qwen2 MLP](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py)
and [SGLang's Qwen2 MLP](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2.py)
use the same packed gate/up topology followed by a fused `SiluAndMul` and a
down projection. The local Triton experiment tests that pointwise fusion
without installing either engine. vLLM 0.25.1 pins PyTorch 2.11, while the
task pod uses PyTorch 2.12.0+cu130; the
[vLLM installation guide](https://github.com/vllm-project/vllm/blob/v0.25.1/docs/getting_started/installation/gpu.cuda.inc.md)
warns that its compiled kernels are tied to the CUDA and PyTorch build. An
in-place wheel install would therefore invalidate the controlled environment.

Transformers 4.57.6 documents that
[`StaticCache` supports compilation and enables automatic decode compilation](https://huggingface.co/docs/transformers/kv_cache).
Its [generation implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py)
keeps prefill eager and uses the compiled call for subsequent decode. This is
the closest low-cost transplant of the decode CUDA-graph pattern used by both
serving engines. It is tested without the failed outer whole-model compile.

## Engine-level ideas deferred to Issue #21

| Lane | Expected resource or behavior change | Why it is not a drop-in #19 kernel |
|---|---|---|
| Flattened or ragged token execution | removes prompt padding through the whole model | requires a model runner, token metadata, and paged attention rather than a module swap |
| Continuous and overlap scheduling | refills decode batches and hides host scheduling gaps | changes request batching over time and needs a real queue/KV manager |
| Full and piecewise CUDA graphs | reduces launch overhead across stable decode and mixed shapes | needs engine-owned static buffers, capture sizes, and graph-padding accounting |
| Paged or Radix KV and prefix caching | reuses exact common prompt blocks and increases concurrency | needs block allocation, cache reset, hit-rate receipts, and cold/warm separation |
| Blackwell backend matrix | selects FlashInfer/TRTLLM-gen or FlashAttention-4 by phase | needs the engine's attention and KV-page integration |
| N-gram or suffix speculation | reduces target-model forward calls for predictable JSON continuations | needs proposal, verification, acceptance, and scheduler accounting |
| Packed QKV and fused residual RMSNorm | removes projection and pointwise launches | requires attention/decoder rewrites and a new full-quality replay |

vLLM documents
[flattened model inputs](https://docs.vllm.ai/en/latest/contributing/model/basic/),
[continuous batching and chunked prefill controls](https://docs.vllm.ai/en/latest/configuration/optimization/),
[CUDA graph modes](https://docs.vllm.ai/en/latest/design/cuda_graphs/), and
[automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/).
SGLang exposes the corresponding
[server and CUDA-graph controls](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md),
[attention backend selection](https://docs.sglang.io/docs/advanced_features/attention_backend), and
[RadixAttention prefix reuse](https://www.lmsys.org/blog/2024-01-17-sglang/).

These lanes may produce a larger gain than a local kernel because the current
fixed batches keep generating slots after some examples have reached EOS and
pad each prompt batch to its longest member. They are not credited to Issue
#19 until a native engine implementation reproduces the physical shape and
passes the same full quality and accepted-token accounting.

## Exclusions

- Length or prefix sorting is excluded from the primary comparison because it
  violates the frozen input-order contract.
- Weight quantization and quantized KV change the numerical contract and are
  not candidates for the primary BF16 result.
- Tensor, pipeline, data, or prefill/decode parallelism does not fit the
  one-B200 contract.
- A single-expert MoE reinterpretation adds routing and permutation work; it
  is not assumed to improve a dense sequential MLP.
- Warm prefix-cache throughput cannot replace the cold primary result.
- Batch-invariant modes may stabilize an engine internally, but they do not
  prove equality to the Transformers reference.

## Promotion rule

Every local or engine candidate must report accepted generated tokens/s,
generated slots/s, examples/s, prefill and decode throughput, p50/p95 latency,
VRAM, launch/profile data, setup overhead, and the complete `1007`-row score.
Prediction differences remain explicit by eval ID. A faster microkernel or a
warm-cache serving number is not a winner by itself.
