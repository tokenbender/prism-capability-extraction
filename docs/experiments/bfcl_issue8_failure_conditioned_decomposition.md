# BFCL Issue #8: Failure-Conditioned Capability Decomposition

Issue: https://github.com/tokenbender/prism-capability-extraction/issues/8

This corpus build turns the #4/#5/#6 BFCL failure surface into a reusable
failure-conditioned decomposition dataset. It is not a training result. It is
the first artifact for testing whether error buckets can act as an automatic
task/capability decomposition signal for sparse selected-MLP-channel recovery.

## Artifact

Published path:

```text
TokenBender/circuit-discovery/bfcl/issue8_failure_conditioned_decomposition_v1
```

Direct link:

```text
https://huggingface.co/datasets/TokenBender/circuit-discovery/tree/main/bfcl/issue8_failure_conditioned_decomposition_v1
```

The HF tree contains:

- `failure_matrix.jsonl.gz`: one row per failure observation
- `eval_id_catalog.jsonl.gz`: one row per stable BFCL eval ID with prompt,
  tools, reference calls, and split role
- `bucket_report.md` and `bucket_report.json`
- `split_manifest.json`
- overlap and cross-tab CSVs
- `launch_recommendations.json`
- `checksums.sha256`

## Source Surface

Inputs are the published failure-bucket artifacts from:

| Source | HF root |
|---|---|
| #4 | `bfcl/issue4_edge_collimation_v1` |
| #5 | `bfcl/issue5_nearmiss_loop_v1` |
| #6 | `bfcl/issue6_tree_search_v1` |

The builder reads every `all_failures.jsonl` file under those roots.

## Verified Corpus Counts

| Metric | Count |
|---|---:|
| Source failure files | 250 |
| Failure observations | 150,582 |
| Unique BFCL eval IDs failed at least once | 1,007 |
| Decision-eligible failure observations | 134,948 |
| Heldout failure observations | 15,634 |

Observation counts are repeated across issues, rounds, branches, and selected
MLP-channel budgets. Unique ID counts collapse repeated observations of the
same BFCL row.

## Split Policy

The split is deterministic by stable BFCL eval ID, not by observation row.

| Split | Unique eval IDs | Failure observations |
|---|---:|---:|
| train | 609 | 90,344 |
| calibration | 152 | 22,629 |
| validation | 146 | 21,975 |
| heldout | 100 | 15,634 |

Split seed:

```text
issue8_failure_conditioned_decomposition_v1
```

Launch decisions should use train + calibration + validation summaries only.
Heldout rows are preserved for final auditing and should not be used to choose
attribution buckets, train synthetic rows, or select branches.

## Main Buckets

Primary failure types:

| Failure type | Observations | Unique eval IDs |
|---|---:|---:|
| wrong_arg_value | 69,449 | 922 |
| wrong_function | 42,947 | 1,007 |
| missing_arg | 18,116 | 474 |
| arg_key_mismatch | 7,846 | 591 |
| multi_call_or_extra_call | 7,461 | 1,007 |
| extra_arg | 4,763 | 237 |

Repair buckets:

| Repair bucket | Observations | Unique eval IDs |
|---|---:|---:|
| arg_value_exactness | 69,449 | 922 |
| live_slot_values | 54,356 | 411 |
| function_name_disambiguation | 42,947 | 1,007 |
| schema_completion | 30,725 | 791 |
| sql_schema_discipline | 17,875 | 101 |
| unit_default_normalization | 4,974 | 89 |
| misc_failure | 4,826 | 631 |
| time_normalization | 3,278 | 20 |
| formula_normalization | 1,586 | 19 |

## First Attribution Launch Recommendation

Run the following before any teacher-guided repair training:

| Priority | Run | Mode |
|---:|---|---|
| 0 | `r0_global_decision_eligible` | global attribution baseline |
| 1 | `r0_value_recovery` | `arg_value_exactness` + `live_slot_values` |
| 2 | `r0_function_selection` | `wrong_function` / `function_name_disambiguation` |
| 3 | `r0_schema_completion` | missing/extra/key mismatch / `schema_completion` |
| 4 | `r0_sql_domain_control` | `sql_schema_discipline` plus SQL category control |
| 5 | `r0_category_controls` | category-conditioned controls |
| 6 | `r0_small_normalization_pilots` | time/unit/formula nested pilots |

The comparison should be global top-k versus bucket top-k, bucket-union,
weighted bucket-union, category-conditioned controls, and any routed variant
only if its routing signal is available before final tool-call emission.

## Attribution Slice Preparation

Prepare runnable pair files from the published catalog and failure matrix:

```bash
python3 code/scripts/prepare_bfcl_issue8_attribution_slices.py \
  --catalog results/bfcl/issue8_failure_conditioned_decomposition/eval_id_catalog.jsonl.gz \
  --failure-matrix results/bfcl/issue8_failure_conditioned_decomposition/failure_matrix.jsonl.gz \
  --out-dir runs/issue8_failure_conditioned_decomposition \
  --smoke-limit 4
```

The script writes:

- `pairs/decision_eligible_all.jsonl`
- `pairs/heldout_audit_only.jsonl`
- one full pair file per first-round attribution slice
- one `.smoke.jsonl` file per slice
- `attribution_slice_manifest.json`

Heldout eval IDs are excluded from every attribution-selection file. They are
written only to `heldout_audit_only.jsonl` for final auditing.

## Attribution And Substrate Results

Compute used for the completed attribution/evaluation phase:

- pod: `zesty-shark-55`
- config: `8xB200`
- runtime fix: isolated CUDA 13 venv with `torch 2.12.0+cu130`
- code commits:
  - `d38b3ec` - attribution slice preparation
  - `9579eb0` - candidate mask construction

The current full-model anchor under this exact harness is `661/1007`
normalized exact. This is close to, but not identical with, the earlier
`664/1007` anchor.

All scores below are normalized exact on the 1007-row BFCL catalog. Behavior
recovery uses the current `661/1007` full-model anchor.

| Strategy | k80 | k120 | k160 | k200 | k240 |
|---|---:|---:|---:|---:|---:|
| global attribution | 0 | 9 | 91 | 237 | 378 |
| value bucket | 0 | 9 | 87 | 236 | 378 |
| failure-max bucket blend | 0 | 8 | 119 | 266 | 409 |
| even bucket union | 1 | 16 | 99 | 247 | 377 |
| weighted bucket union | 0 | 18 | 94 | 244 | 367 |
| category weighted control | 0 | 2 | 85 | 219 | 337 |

Best #8 candidate by budget:

| Budget | Selected MLP channels | MLP % | Best #8 strategy | Score | Recovery vs 661 |
|---:|---:|---:|---|---:|---:|
| k80 | 80,000 | 18.08% | even bucket union | 1/1007 | 0.15% |
| k120 | 120,000 | 27.13% | weighted bucket union | 18/1007 | 2.72% |
| k160 | 160,000 | 36.17% | failure-max bucket blend | 119/1007 | 18.00% |
| k200 | 200,000 | 45.21% | failure-max bucket blend | 266/1007 | 40.24% |
| k240 | 240,000 | 54.25% | failure-max bucket blend | 409/1007 | 61.88% |

Heldout-only k240 audit:

| Strategy | Heldout score | Heldout recovery vs full heldout 59/100 |
|---|---:|---:|
| full unmasked Qwen3-8B | 59/100 | 100.00% |
| failure-max bucket blend | 39/100 | 66.10% |
| global attribution | 37/100 | 62.71% |
| value bucket | 37/100 | 62.71% |
| even bucket union | 37/100 | 62.71% |
| weighted bucket union | 36/100 | 61.02% |
| category weighted control | 34/100 | 57.63% |

Interpretation:

- Failure-conditioned attribution produced a real but modest signal: the
  failure-max blend beats global attribution by `+28` at k160, `+29` at k200,
  and `+31` at k240 on the full 1007-row catalog.
- The same ordering mostly survives the 100-row heldout audit, but the heldout
  gain over global is only `+2` at k240.
- The absolute scores remain far below the trained BFCL frontiers from prior
  issues, especially the #5/#6 k160/k200/k240 region.
- Teacher-guided repair training was therefore not launched inside #8. The
  attribution signal is useful as a decomposition diagnostic, but not strong
  enough to justify an expensive #8 training phase without a redesigned
  training objective or a follow-up issue.

Conclusion: #8 is a clean negative/diagnostic result. Failure-conditioned
decomposition does improve sparse attribution over global attribution at matched
larger budgets, but it does not recover enough behavior to compete with the
existing trained sparse-substrate frontier.

## Reproduction

Build from source artifacts:

```bash
python3 code/scripts/build_bfcl_issue8_failure_matrix.py \
  --out-dir results/bfcl/issue8_failure_conditioned_decomposition
```

Then verify local checksums:

```bash
cd results/bfcl/issue8_failure_conditioned_decomposition
shasum -a 256 -c checksums.sha256
```
