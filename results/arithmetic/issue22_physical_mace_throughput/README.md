# Issue #22 arithmetic physical MACE and B200 throughput

This directory is the compact, Git-sized evidence index for GitHub issue #22.
It binds one frozen 1,500-example two-digit-addition workload to source
provenance, bounded logical extraction, physicalization, quality-gated runtime
selection, profiling, publication receipts, and explicit negative results.

The core 1,500-row experiment is complete. Issue #22 remains open because the
historical donor-patch replay, independent data-role confirmation, exhaustive
8,100-pair audit, Hugging Face publication, and camera-ready integration are
not all complete.

## Result in one paragraph

The historical 12,661-channel paper mask scores `0/1,500` when evaluated as a
standalone zero-isolated model without counterfactual donor activations or
gold-answer-length input. A bounded 20-round search found a logical MACE-90 at
`242,724/250,880` channels, but BF16 shape-dependent numerical changes prevented
the smaller candidates from retaining 99% of logical correctness after physical
compaction. The first tested physical candidate satisfying both MACE-90 and
parity-99 retains `250,237/250,880 = 99.7437%` of MLP channels and scores
`1,386/1,500 = 92.4%`. Its fastest tested quality-valid one-B200 setup reaches
`10,208.963` median examples/s, `3.4417x` the canonical eager physical setup,
but remains `0.3474%` slower than the equally optimized dense parent. This is a
positive standalone physical MACE and runtime result, not a strong arithmetic
compression result or a dense speedup.

## Frozen scientific contract

- Base: `Qwen/Qwen2.5-Math-1.5B` at
  `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.
- Collimator: rank-32 RS-LoRA, selected merge scale `0.55`.
- Evaluated merged checkpoint: exact reconstruction match across all `339`
  tensors and `1,777,088,000` elements.
- Dtype: BF16.
- Workload: 1,500 historical-spaced two-digit-addition records.
- Generation: unguided greedy generation with a global eight-token cap.
- Scorer: strict first-line integer equality.
- Isolation: zero complement, with no donor model/pass and no gold-length input.
- Hardware for final runtime selection: one NVIDIA B200, TP1.

The historical `1,370/1,500 = 91.3333%` paper result used a different
counterfactual-patch and gold-answer-length intervention. It remains historical
causal evidence; it was not reproduced as part of this standalone run and must
not be described as a standalone sparse model.

## Quality and physicalization

| State | Correct | Matched-dense recovery | MLP channels |
|---|---:|---:|---:|
| Dense standalone parent | `1,406/1,500` | — | `250,880/250,880` |
| Historical paper seed under zero isolation | `0/1,500` | `0/1,406` | `12,661/250,880` |
| Bounded logical winner, round 19 | `1,272/1,500` | `1,270/1,406 = 90.3272%` | `242,724/250,880 = 96.7490%` |
| Selected physical parent, round 3 | `1,386/1,500` | `1,378/1,406 = 98.0085%` | `250,237/250,880 = 99.7437%` |

The physical model retains `1,376/1,388 = 99.1354%` of the logical parent's
correct rows. The canonical receipt is `evaluation/logical_vs_physical.json`.
The provisional wording `1,377/1,389` is unsupported and must not be used.

The physical artifact contains `1,540,751,360` canonical parameters versus
`1,543,714,304` in the dense parent. It removes `2,962,944` parameters, or
`0.1919%` of the total. All unselected MLP rows/columns are absent, but this is
not a material total-model compression win.

The evaluated checkpoint included a proven duplicate tied
`lm_head.weight`/`model.embed_tokens.weight` serialization and was
`3,548,288,864` bytes with SHA-256
`7d8315ae8da2734e322eb95d692d4975e13d58730bff30ffbe26926d2de41e02`.
The publication form omits the duplicate alias, reconstructs it before strict
state assignment, and deterministically shards the canonical `338` tensors:

- canonical unsharded bytes: `3,081,541,256`;
- canonical unsharded SHA-256:
  `682f4b6812f2f7452e3b646cabf4bea0452cf752375339b544def400f10c90cb`;
- publication shards: `12`;
- shard payload plus index: `3,081,568,517` bytes;
- shard-manifest SHA-256:
  `571a1e50bdddb5dc289367663f24090c4225725cbac4006c0c009bc08cbc5521`.

The tied-weight deduplication and sharding are serialization-only operations.
The clean CPU load proves zero added runtime parameters and exact alias
reconstruction.

## Runtime winner

The diversity-first race exercised 17 configurations across eager/SDPA
attention, separate/packed projections, five width alignments,
Torch/Triton/hybrid activation, dynamic/static cache controls, and compile
controls. The full sweep and batch saturation receipts are in `benchmarks/`.

| Metric | Canonical eager physical | Optimized physical | Equally optimized dense |
|---|---:|---:|---:|
| Batch | `512` | `1,500` | `1,500` |
| Correct | `1,386/1,500` | `1,395/1,500` | `1,403/1,500` |
| Median examples/s | `2,966.286` | `10,208.963` | `10,244.551` |
| Median generated tokens/s | — | `81,671.707` | `81,956.406` |
| Estimated supported-op MFU | `4.987%` | `16.111%` | `16.365%` |

The winner is SDPA attention, packed gate/up projections aligned to 16,
dynamic cache, and the Triton SiLU-multiply path. Its five repeats have
identical predictions. The `3.4417x` improvement is a full-setup result that
includes batching and runtime changes. The physical-to-dense throughput ratio
is `99.6526%`; no dense speedup is claimed.

MFU values come from `torch.profiler` FLOP estimates for supported operators
and a 2,250-TFLOP/s BF16 dense B200 denominator. They are not hardware-counter
measurements. Sampled GPU utilization peaked at 99% during the finalist queue;
queue-wide means include model loads and idle boundaries and are not presented
as active-generation utilization.

## Evidence map

- `selection.json`: final quality-valid runtime winner and comparison.
- `physical_selection.json`: complete tested physical frontier.
- `search/`: frozen search manifest plus logical and physical-parent receipts.
- `evaluation/`: dense/physical predictions, parity, recovery, and all diffs.
- `model/`: evaluated and publication-form substrate metadata.
- `benchmarks/`: saturation, diversity, finalist, and maximum-batch results.
- `profiling/`: canonical, optimized physical, and optimized dense profiles.
- `utilization/`: raw sampler CSVs and summaries.
- `provenance/`: exact merge, tokenizer, base revision, sharding, clean-load,
  software, and hardware receipts.
- `publication/`: W&B, local preservation, and explicit publication/teardown
  status receipts.
- `CLAIM_LEDGER.md`: authorized paper language and prohibited extensions.
- `SERVING_RUNTIME_AUDIT.md`: exact serving boundary.
- `negative_results.json`: failed candidates and unexecuted gates.

## Publication status

- W&B: verified. Group:
  <https://wandb.ai/ahm-rimer/prism-arithmetic/groups/issue22-arithmetic-physical-throughput>.
  Artifact `issue22-arithmetic-physical-throughput-evidence:v0`, digest
  `1ea5edf20728d1435529ea735e49105d`.
- ModelScope: private repository exists at
  <https://modelscope.ai/models/tokenbender/prism-arithmetic-mace-physical-v1>.
  Immutable tag `issue22-physical-v1.2` is verified: its fresh download exactly
  matches all 44 uploaded payload files; 43 `SHA256SUMS` entries passed, the
  checksum manifest is separately hash-bound, and the strict donor-free CPU
  clean load passed. See `publication/MODELSCOPE_RECEIPT.json`.
- Hugging Face: blocked by absent local write authentication; see
  `publication/HF_BLOCKER.json`.
- Local preservation: the original evidence and evaluated selected bundle are
  checksummed, and the exact 44-file ModelScope `v1.2` payload is archived, in
  `publication/LOCAL_ARCHIVE_RECEIPT.json`.
- Lium: the task-owned one-B200 pod is removed and the post-removal inventory
  reports no active pods; see `publication/LIUM_TEARDOWN_RECEIPT.json`.

## Residual work

1. Reproduce the historical `91.33%` counterfactual-patch protocol.
2. Separate search/calibration/final-confirmation roles and run the declared
   exhaustive 8,100-pair domain audit.
3. Complete or durably block Hugging Face publication.
4. Update the camera-ready manuscript and issue #18 with
   this intervention boundary.

Global MACE minimality, general arithmetic capability, a stock serving-engine
integration, and a universal speedup are not claimed.
