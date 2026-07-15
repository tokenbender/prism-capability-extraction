# PRISM BFCL MACE physical substrate

This is the **private, verification-stage** physical artifact for
[`tokenbender/prism-capability-extraction` issue #18](https://github.com/tokenbender/prism-capability-extraction/issues/18).
It is not yet authorized for public redistribution; see `LICENSE_STATUS.md`.

## What is physically smaller

The artifact keeps full attention, embeddings, normalization, residual, rotary,
and LM-head components. It physically removes unselected Qwen3 MLP intermediate
channels from `gate_proj`, `up_proj`, and `down_proj` in every layer.

- selected MLP channels: **140,875 / 442,368 (31.846%)**
- physical parameters: **4,485,989,376 / 8,190,735,360 (54.769%)**
- physical safetensors bytes: **8,972,024,720 / 16,381,516,824 (54.769%)**
- physical weight SHA-256: `d0e78a244b01e0d0cce027f9bc65cea304315f522dc4359de2fc9fc4b5f2fb64`

`config.json` intentionally retains the dense parent's global
`intermediate_size=12288`; the actual non-uniform widths live in
`substrate_metadata.json`. Generic `AutoModel.from_pretrained` and config-only
FLOP estimators are therefore invalid. Use the included strict loader.

## Measured BFCL result

All counts use the frozen PRISM internal normalized structured matcher, not the
official BFCL leaderboard scorer.

| comparison | numerator | denominator | ratio |
|---|---:|---:|---:|
| same-environment dense merged parent | 598 | 671 | **89.1207%** |
| frozen logical mask | 598 | 600 | **99.6667%** |
| preserved live-adapter anchor | 598 | 672 | 88.9881% |
| historical base-only anchor | 598 | 664 | 90.0602% |

The first row is the primary physical-compression comparison. The latter two
are cross-run or cross-state provenance and must not be relabelled as same-state.

Physical category scores are deliberately preserved: `exec_simple 87/100`,
`java 58/100`, `javascript 31/50`, `live_simple 96/258`, `simple 284/400`,
and `sql 42/99`.

## Same-harness B200 benchmark

Five measured repeats followed one warmup on the same B200, BF16, eager
attention, batch size 8, and the same 24-example workload.

| metric | dense | physical | change |
|---|---:|---:|---:|
| load seconds | 4.634 | 4.322 | -6.72% |
| after-load allocated VRAM | 16.384 GB | 9.002 GB | -45.06% |
| peak allocated VRAM | 18.135 GB | 10.754 GB | -40.70% |
| prefill tokens/s | 32,787.5 | 23,040.6 | **-29.73%** |
| cached decode tokens/s | 257.28 | 277.45 | **+7.84%** |
| end-to-end generated tokens/s | 239.85 | 251.07 | **+4.68%** |

Performance is mixed: the irregular skinny MLPs reduce memory and help decode
on this workload, but hurt prefill. This is not a blanket speedup claim.

## Strict load check

Create a Python 3.12 environment with the versions in
`environment/requirements.txt`, then run:

```bash
python scripts/load_bfcl_physical_bundle.py load-check \
  --bundle . \
  --device cuda:0
```

The loader fails closed on format, Qwen model type, per-layer coverage, BF16
metadata/config/checkpoint/parameters, tensor keys and shapes, multi-EOS
generation configuration, and the frozen legacy-tokenizer contract.

The full scorer inputs are not redistributed in this private model package.
`environment/commands.md` records how to obtain the pinned BFCL source and run
evaluation after its terms have been reviewed.

## Provenance

- base: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
- adapter snapshot: `TokenBender/circuit-discovery@cdebd3e01fab1d886f169ed87470a4e83d10a5b5`
- candidate: `category_repair_java_r500_protect_tail_b140875_p10000`
- mask SHA-256: `f83816319ec7d39cd4d95e8c8f44cecc139f23657233ccdd061bda1f71350567`
- PRISM source commit: `07b46cdd41648dd83fbe8180752085bd59adafb5`
- frozen scorer SHA-256: `17ce35b8ca1633e6cf8a5da5de032d5671b33b184f52daf5b8f9a63f9b76bc12`

See `MANIFEST.json`, `SHA256SUMS`, `ATTRIBUTION.md`, and the evaluation and
benchmark directories for the complete evidence ledger.
