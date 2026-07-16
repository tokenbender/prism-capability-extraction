# BFCL / Function-Calling Receipts

This directory contains small BFCL/function-calling receipts included directly
in the git release. Public legacy generated artifacts, adapters, datasets, and
full-model outputs are released on Hugging Face and pinned in
`docs/ARTIFACT_MANIFEST.json`. The Issue #18 physical substrate is privately
preserved on ModelScope pending adapter provenance and public-release approval;
its receipt and model SHA-256, rather than mutable `master`, are the proof surface.

The current camera-ready evidence is the Issue #18 physical MACE substrate in
`issue18_physical_mace_v1/` plus the quality-gated B200 runtime result in
`issue19_physical_throughput/`. Issue #19 changes the runtime representation and
execution setup; it does not change the stored physical-substrate weights.
Older k160/k240 results below are retained as historical runtime-mask lineage
and are not the primary physical-model claim.

## Included Local Receipts

- `eval_masked_summary.json`: k160 masked eval summary.
- `eval_unmasked_summary.json`: unmasked merged-model eval summary.
- `gkd_aopd_hybrid_k160_v0__eval_k160_hybrid_v0_masked.summary.json`: k160
  hybrid masked eval summary.
- `gkd_aopd_hybrid_k160_v0__run_config.json`: k160 hybrid run config.
- `gkd_aopd_hybrid_k160_v0__train_summary.json`: k160 hybrid train summary.
- `r32_more_online_repro_summary.json`: reproduction summary for the uploaded
  full-model artifact.
- `issue18_physical_mace_v1/`: physical jagged-MLP bundle receipts, strict
  loader provenance, full parity audit, and same-harness benchmark.
- `issue19_physical_throughput/`: diversity-first kernel/runtime sweep,
  five-repeat B200 winner, full quality replay, profiler and phase receipts,
  vLLM/SGLang audit, W&B artifact, and executor cleanup proof.

## Public Artifact Pointers

| artifact | repository | revision |
|---|---|---|
| BFCL data and reproduction artifacts | `Occupying-Mars/issue49-bfcl-repro-artifacts` | `303db0bddcfb04bebaf07ab4a4dc4c089240c545` |
| k160 rank-32 adapter | `Occupying-Mars/issue49-k160-r32-len1024-adapter` | `e5104eee6e9dd0fff11f377b743330429970d672` |
| k240 rank-16 adapter | `Occupying-Mars/issue49-k240-r16-adapter` | `b9018cc3090b856df701240fd73f9f98c627917c` |
| r32 more-online full model | `TokenBender/issue51-r32-more-online-codex-repro-551-full` | `ff4daac9e49a8f927153c9a04daa9faba2fb5a66` |
