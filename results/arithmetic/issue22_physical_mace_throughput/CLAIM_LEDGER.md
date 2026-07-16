# Issue #22 claim ledger

This ledger separates what the preserved receipts support from language that
would extend beyond the experiment. Every claim is scoped to the frozen
1,500-row two-digit-addition workload unless stated otherwise.

## Authorized claims

| ID | Authorized language | Primary receipt | Prohibited extension |
|---|---|---|---|
| A1 | The evaluated checkpoint is exactly reconstructible from `Qwen/Qwen2.5-Math-1.5B@4a83ca6e...`, the audited rank-32 adapter, and merge scale `0.55`; all 339 tensors and 1,777,088,000 elements match exactly. | `provenance/merge_provenance.json`, `provenance/base_revision_receipt.json` | Do not describe an unpinned model alias as the source. |
| A2 | The paper's `1,370/1,500 = 91.33%` arithmetic result is historical causal evidence under counterfactual donor patching and gold-answer-length generation. | Historical receipts indexed by `docs/ARTIFACT_MANIFEST.json`; intervention audit in issue #22 | Do not call the 12,661-channel paper mask a standalone or physical model. The historical protocol was not freshly reproduced here. |
| A3 | The dense merged parent scores `1,406/1,500 = 93.7333%` under the frozen standalone scorer. | `evaluation/dense_manifest.json`, `evaluation/dense_predictions.jsonl` | Do not generalize this score to the exhaustive 8,100-pair domain or unseen arithmetic. |
| A4 | The historical 12,661-channel mask scores `0/1,500` under true zero isolation. | `search/manifest.json` | Do not say the historical causal result was falsified; the intervention contract changed. |
| A5 | The smallest bounded logical MACE-90 found in 20 rounds retains `242,724/250,880 = 96.7490%` of MLP channels and recovers `1,270/1,406 = 90.3272%` of dense-correct rows. | `search/logical_winner_round19_receipt.json` | Do not claim global minimality. |
| A6 | The first tested physical candidate satisfying MACE-90 and parity-99 retains `250,237/250,880 = 99.7437%` of MLP channels, scores `1,386/1,500`, and recovers `1,378/1,406 = 98.0085%` of dense-correct rows. | `physical_selection.json`, `evaluation/physical_dense_recovery.json` | Do not describe it as a highly compressed model. |
| A7 | Physicalization retains `1,376/1,388 = 99.1354%` of logical-correct rows and preserves all 209 prediction differences. | `evaluation/logical_vs_physical.json`, `evaluation/logical_vs_physical_differences.jsonl` | Do not use the unsupported provisional count `1,377/1,389`. |
| A8 | Unselected MLP rows and columns are absent. The strict loader allocates no dense MLP, donor model, adapter, or runtime mask. | `model/evaluated_substrate_metadata.json`, `provenance/clean_load_receipt.json` | Do not equate runtime zeroing with physical removal. |
| A9 | The canonical physical model has `1,540,751,360` parameters versus `1,543,714,304` dense, a `0.1919%` total-parameter reduction. | `selection.json` | Do not infer a large total-model reduction from the `99.7437%` MLP-channel fraction. |
| A10 | A proven tied-weight alias was omitted from publication serialization and reconstructed before strict assignment; deterministic 12-way sharding is semantically unchanged. | `provenance/tied_weight_dedup_receipt.json`, `provenance/sharding_receipt.json`, `provenance/clean_load_receipt.json` | Do not compare the alias-duplicated evaluated checkpoint byte size directly to the canonical transport as if deduplication were model pruning. |
| A11 | The fastest tested quality-valid physical setup reaches `10,208.963` median examples/s and `81,671.707` generated tokens/s on one B200 at batch 1,500, with identical predictions across five repeats. | `selection.json`, `benchmarks/max_batch_benchmark.json` | Do not call it universally fastest or a general serving result. |
| A12 | The optimized setup is `3.4417x` faster than the canonical eager physical setup for this workload. | `selection.json` | Keep batch sizes visible: canonical is batch 512 and the winner is batch 1,500. Do not reduce this to a kernel-only speedup. |
| A13 | The physical winner is `0.3474%` slower than the equally optimized dense parent and reaches `99.6526%` of dense throughput. | `selection.json` | No dense speedup is claimed. |
| A14 | Estimated supported-op MFU rises from `4.987%` for canonical eager physical to `16.111%` for the winner; equally optimized dense is `16.365%`. | `profiling/*.json` | MFU is a `torch.profiler` estimate, not a hardware-counter measurement. |
| A15 | W&B contains six finished issue-scoped runs and verified evidence artifact `v0`, digest `1ea5edf20728d1435529ea735e49105d`. | `publication/WANDB_RECEIPT.json` | Do not claim Hugging Face, ModelScope payload, or teardown completion from the W&B receipt alone. |
| A16 | Private ModelScope tag `issue22-physical-v1.2` exactly matches all 44 uploaded payload files. A fresh download passed all 43 checksums named by `SHA256SUMS`, the checksum manifest is separately hash-bound, and the strict donor-free CPU load passed. | `publication/MODELSCOPE_RECEIPT.json` | Do not call the private repository a public release or substitute the mutable `master` branch for the immutable tag. |
| A17 | The task-owned one-B200 Lium pod was removed after preservation; the post-removal inventory reported no active pods and no unrelated pod was modified. | `publication/LIUM_TEARDOWN_RECEIPT.json` | The exact removal timestamp was not captured; do not invent one or infer task completion from teardown alone. |
| A18 | The exact 44-file ModelScope `v1.2` payload is preserved in a local tar archive bound by SHA-256. | `publication/LOCAL_ARCHIVE_RECEIPT.json` | The local archive does not turn the private artifact into a public release. |

## Camera-ready routing

The defensible paper story is:

1. **Arithmetic:** collimation can concentrate causal sufficiency under a named
   counterfactual intervention. The historical 5.05%-channel result must carry
   its donor-patch and gold-length contract.
2. **Issue #22 standalone audit:** causal extraction is not automatically
   standalone physical deployment. This is useful contract-validation evidence,
   but the passing physical candidate is near-dense and should not be promoted
   as a second compression win.
3. **Function calling:** issue #18 remains the genuine physical parameter-removal
   demonstration; issue #19 remains its bounded runtime result.
4. **Translation:** historical non-primary provenance only.

## Explicitly unauthorized headlines

- "Five percent of the arithmetic MLP runs standalone."
- "The arithmetic model is compressed by roughly 99%."
- "The physical arithmetic substrate beats the dense model."
- "The arithmetic MACE is globally minimal."
- "The result establishes general arithmetic reasoning."
- "vLLM or SGLang serves the physical substrate natively."
- "GPU utilization is 99% throughout the run."
- "MFU is a hardware-counter measurement."
- "Issue #22 is closed" while the residual gates in `README.md` remain open.
