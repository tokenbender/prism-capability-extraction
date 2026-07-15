# Attribution and modification notice

This physical model derives from `Qwen/Qwen3-8B` at revision
`b968826d9c46dd6066d109eabc6255188de91218` and from the PRISM `b007` adapter snapshot at revision
`cdebd3e01fab1d886f169ed87470a4e83d10a5b5`.

PRISM modifications:

1. Merge the rank-32 BFCL adapter into the pinned BF16 base.
2. Retain the exact 140,875-channel Issue #12 selection.
3. Slice `gate_proj` and `up_proj` rows and matching `down_proj` columns in
   every MLP layer.
4. Serialize the resulting jagged per-layer MLPs with the full transformer
   scaffold unchanged.
5. Add an independently implemented PRISM loader and verification receipts.

No private `circuit-shotting` source code is included in this package.
