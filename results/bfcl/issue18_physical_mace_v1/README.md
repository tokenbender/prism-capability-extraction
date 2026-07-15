# Issue #18 physical BFCL MACE receipt

This directory is the git-sized evidence index for the physical
`category_repair_java_r500_protect_tail_b140875_p10000` substrate. The model
weights, prediction-level diffs, and full checksum tree live in the private
verification artifact; they are not committed to git.

## Result boundary

| comparison | score | ratio | status |
|---|---:|---:|---|
| physical vs same-environment dense merged parent | `598/671` | `89.1207%` | primary |
| physical vs frozen logical mask | `598/600` | `99.6667%` | physicalization parity |
| physical vs preserved live-adapter dense anchor | `598/672` | `88.9881%` | cross-run provenance |
| physical vs historical unadapted base anchor | `598/664` | `90.0602%` | cross-state provenance |

All counts use the frozen PRISM internal normalized structured matcher, not
the official BFCL leaderboard scorer. The physical artifact scores `585/1007`
under raw exact matching.

The artifact retains `140,875 / 442,368` MLP intermediate channels
(`31.846%`) but `4,485,989,376 / 8,190,735,360` total parameters (`54.769%`).
Its safetensors payload is `8,972,024,720` bytes, also `54.769%` of the dense
merged parent tensor bytes. Non-selected dense MLP dimensions are absent.

## Performance boundary

Five same-harness B200 repeats show mixed performance:

- after-load allocated VRAM: `-45.06%`;
- peak allocated VRAM: `-40.70%`;
- prefill throughput: `-29.73%`;
- cached-decode throughput: `+7.84%`;
- end-to-end generated-token throughput: `+4.68%`.

This supports a measured memory reduction and workload-specific decode gain,
not a universal speedup claim.

## Included receipts

- `evaluation/bundle_inspection.json`: tensor shapes, parameter accounting,
  serialized bytes, and mask/metadata verification.
- `evaluation/logical_vs_physical_summary.json`: full `1007`-row physical vs
  frozen logical-mask parity and category/split audit.
- `evaluation/dense_vs_physical_summary.json`: same-environment physical vs
  dense merged-parent comparison.
- `evaluation/merge_vs_live_adapter_summary.json`: the one-point merge drift
  boundary between the replayed merged parent and preserved live-adapter run.
- `evaluation/smoke_summary.json`: strict cold-load prediction parity.
- `benchmarks/`: repeated same-GPU load, memory, prefill, cached-decode, and
  end-to-end generation results.
- `REMOTE_MANIFEST.json`: the complete private artifact file ledger and hashes.
- `MODELSCOPE_RECEIPT.json`: private v2 inventory, checksum, independent cold-load,
  and `17/24` smoke-parity verification.
- `WANDB_RECEIPT.json`: finished run and verified historical v0 evidence lineage,
  including the exact clean-payload comparison and ModelScope receipt binding.
- `WANDB_STORAGE_RESIDUAL.json`: the provider-side signed-storage failure that
  blocked creation of a clean W&B v1 artifact.
- `LOCAL_ARCHIVE_RECEIPT.json`: clean local tar identity, size, checksum tree,
  and secret/cache scan result.
- `LIUM_TEARDOWN_RECEIPT.json`: task-owned pod removal and transient-credential
  cleanup verification.
- `MODEL_CARD.md`, `LICENSE_STATUS.md`, and `ATTRIBUTION.md`: intended remote
  documentation and the current public-release restriction.

The physical model SHA-256 is
`d0e78a244b01e0d0cce027f9bc65cea304315f522dc4359de2fc9fc4b5f2fb64`.
The PRISM loader is `code/scripts/load_bfcl_physical_bundle.py`.

## Release state

The W&B evaluation run is
[`ahm-rimer/prism-bfcl/bfcl_physical_mace_k140875_v1`](https://wandb.ai/ahm-rimer/prism-bfcl/runs/bfcl_physical_mace_k140875_v1).
ModelScope repository `tokenbender/prism-bfcl-mace-140875-physical-restricted-v2`
is the verified private weight mirror. Its remote inventory, all 46 checksum
entries, independent cold load, and `17/24` smoke prediction parity passed. The
repository revision is mutable `master`, so `MODELSCOPE_RECEIPT.json` plus the
model SHA-256 above is the preservation proof. Its license state is
`unspecified_restricted`; verification does not authorize redistribution.

The clean local archive is `8,990,531,584` bytes with SHA-256
`746eb891668cd543a890f1838032f1c54659d986d448d4c0809cfeb1100be5f1`.
The ModelScope remote has 48 entries because it adds managed `.gitattributes`;
the release tree has 47 files including `SHA256SUMS`, which covers 46 payloads.

W&B remains a qualified lineage surface: v0 is verified and binds the complete
ModelScope receipt in metadata, but it contains one generated Python cache file.
That file is excluded from both clean canonical copies. Four independent clean
v1 upload attempts failed at W&B's signed storage with `SignatureDoesNotMatch`,
and no v1 is claimed.

The task-owned 8xB200 Lium pod was removed after preservation, with transient
W&B and ModelScope credentials deleted locally and remotely. A public Hugging
Face upload remains
blocked until write authentication and adapter redistribution provenance are
resolved; this issue must remain open until those receipts and the camera-ready
manuscript changes are complete.
