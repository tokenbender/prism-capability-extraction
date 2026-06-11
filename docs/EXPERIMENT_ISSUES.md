# Experiment Issue Guide

Use this guide for future PRISM experiment issues across arithmetic,
translation, and BFCL/function calling. The goal is not bureaucracy. The goal is
to make every issue resumable, comparable, and auditable after the run context
has faded.

## Issue Contract

Every experiment issue should state the following before expensive work starts:

| field | what to record |
|---|---|
| Research question | The one claim or uncertainty the experiment tests. |
| Task | Arithmetic, EN-PT translation, BFCL/function calling, or another explicit task. |
| Dataset and split | Training, attribution, calibration, validation, and held-out evaluation sources. |
| Prompt or target format | Exact answer surface being optimized or scored, such as full answer, sentence translation, or full tool call. |
| Starting model | Base checkpoint or already post-trained checkpoint, plus any adapters already applied. |
| Controlled variables | Dataset, model, metric, mask size, prompt format, scorer, or other variables that must stay fixed. |
| Experimental variable | The one method, ordering, region, loss, or training change being tested. |
| Baselines | Prior scores, raw masks, unmasked model scores, and relevant controls. |
| Success metric | Primary metric and secondary guardrails. |
| Artifact plan | Which configs, summaries, masks, adapters, tables, and logs will be preserved. |
| Residual risk | Known leak risk, scorer artifact risk, runtime mismatch, or follow-up work. |

If any of these are unknown, the issue should say so directly and either choose
a conservative default or create a follow-up issue.

## Tracking Surfaces

Use each surface for the job it is good at:

| surface | role |
|---|---|
| GitHub issue | Research intent, decisions, links, closeout, and human-readable ledger. |
| W&B run | Exact run config, metrics, charts, and per-example debugging tables. |
| W&B artifact | Versioned dataset manifests, masks, result bundles, and adapter pointers. |
| Hugging Face | Public immutable release surface for large artifacts. |
| Git commit | Code or documentation change that made the run possible. |

The issue remains the source of truth for why the run exists. W&B is the source
of truth for what happened during the run.

## W&B Minimum

When W&B is used, each run should log enough structure to compare across
experiments without reading raw logs.

| key | recommended value |
|---|---|
| project | `prism-arithmetic`, `prism-translation`, or `prism-bfcl` |
| group | GitHub issue id, for example `issue-2` |
| job_type | `dataset_build`, `attribution`, `train`, `eval`, `leak_audit`, or `runtime` |
| tags | task, model, method, mask size, ordering, and run grade |
| config | model id, dataset ids, split hashes, prompt format, scorer, mask source, k, rank, loss weights, seed, and git sha |
| metrics | primary score, guardrail scores, runtime/cost summaries, and failure counts |
| table | per-example predictions, labels, parsed outputs, correctness, and failure bucket |
| artifacts | input manifest, selected mask, adapter pointer, eval summary, failure buckets, and leak-audit summary |

Do not upload secrets, tokens, local caches, private prompts, or full base-model
weights unless that release boundary has been explicitly approved.

## Task-Specific Fields

Arithmetic issues should record:

- number range and prompt template
- answer span being attributed
- mask size or channel budget
- exact-recovery metric
- carry and digit-position strata when relevant

Translation issues should record:

- source and target language
- dataset source and split
- sentence-level or token-level attribution target
- XCOMET, COMET, chrF++, and degeneracy guardrails when relevant
- destructive or keep-only intervention definition

BFCL/function-calling issues should record:

- BFCL category mix and row count
- tool-schema normalization rules
- target function-call format
- parser and canonicalizer version
- normalized exact, raw exact, parser-validity rate, and failure buckets
- leak-audit or canary checks when the experiment touches public benchmark-like data

## Closeout Checklist

Before closing an experiment issue, post a comment with:

- final result and whether the research question was answered
- exact config or W&B run links
- dataset, mask, adapter, and result artifact links
- comparison against the declared baselines
- notable failure buckets or negative results
- leak, scorer, and reproducibility checks completed
- follow-up issues for any remaining cleanup, missing control, or residual risk

An issue is not done just because a run finished. It is done when another person
can understand what was tested, compare it to prior runs, and resume the next
experiment without reconstructing the apparatus from chat history.
