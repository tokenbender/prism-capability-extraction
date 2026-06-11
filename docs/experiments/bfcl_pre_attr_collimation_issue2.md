# BFCL Pre-Attribution Collimation Setup

GitHub issue: https://github.com/tokenbender/prism-capability-extraction/issues/2

This setup locks the launch choices for the publishable BFCL experiment:

- run grade: publishable
- compute: H100-class
- tracking: W&B online, project `prism-bfcl`, group `issue-2`
- artifact destination: `TokenBender/circuit-discovery` with path prefix
  `bfcl/issue2_pre_attr_collimation_v0`
- starting model: `Qwen/Qwen3-8B`
- eval boundary: BFCL 1007 single-call rows stay held out

## Launch Contract

The experiment tests the ordering:

```text
collimate Qwen3-8B on strict BFCL-style rows
        -> run attribution on the collimated local model
        -> select top-k MLP-channel masks
        -> evaluate those masks on held-out BFCL 1007
```

This differs from the previous BFCL recovery line, which selected the mask first
and then conditioned behavior through that fixed mask.

## Required Files

| file | role |
|---|---|
| `code/configs/bfcl_pre_attr_collimation_issue2.json` | machine-readable run contract |
| `code/scripts/train_bfcl_unmasked_lora.py` | pre-attribution unmasked collimation runner |
| `code/scripts/audit_bfcl_train_eval_overlap.py` | train/eval overlap audit |
| `data/bfcl_single_call/pairs.jsonl` | held-out BFCL eval pairs |
| `data/bfcl_strict_10k_mix_len1024/train.jsonl` | strict public training mix |
| `runs/issue2_bfcl_pre_attr_collimation/leak_audit.json` | required overlap audit before training |

## Leak-Audit Checklist

The expensive run should not start until the audit confirms:

- no exact duplicate normalized prompt/tool/target rows between train and eval
- no exact duplicate target tool calls paired with matching or near-matching
  prompt/tool schema
- no suspicious near-duplicate request/schema rows above the chosen threshold
- BFCL eval ids are absent from training manifests
- audit output is saved and linked from issue #2

If any overlap appears, stop and either remove the rows or create a new issue
for the data-boundary decision.

## Artifact Policy

Upload:

- run config and command log
- leak-audit report
- train summary and W&B run link
- LoRA adapter
- attribution scores and attribution summary
- selected masks
- eval summaries and failure buckets

Do not upload:

- API keys or tokens
- local package/model caches
- private prompts or private datasets
- merged/full `Qwen/Qwen3-8B` weights

The local merged model produced by `train_bfcl_unmasked_lora.py` exists only so
the attribution script can load one model directory. Public release should keep
the adapter plus recipe, not the full merged base weights.

## Closeout Bar

Issue #2 can close only after the final comment links:

- W&B run group
- immutable Hugging Face revision or artifact path
- exact config used
- leak-audit output
- k80k, k120k, k160k, k200k, and k240k normalized-exact results
- failure buckets
- comparison to raw ReLP and post-attribution BFCL anchors
- follow-up issues for any missing control or residual risk
