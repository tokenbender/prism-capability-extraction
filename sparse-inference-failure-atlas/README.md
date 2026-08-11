# What Breaks for VRAMlets

> **Ten failures that happened on the compressed path, not merely near it.**

**[Read the report in Markdown](report.md) · [Download the PDF](report.pdf)**

The goal of aggressive compression is simple: fit more model, context, or users into fixed hardware. Here **super-sparse** is an umbrella for low-bit weights or cache state, ternary or pruned representations, structured sparsity, and the converters and kernels needed to run them.

The report admits a case only when a matched control, an enable/disable comparison, or a repaired compressed path ties the optimization to measurable harm. Ordinary prompt, routing, loading, and serving bugs are excluded.

## What survived the filter

1. **[BitsAndBytes serving produced garbage](https://github.com/vllm-project/vllm/issues/5569).**
2. **[A quantized KV-cache bug produced gibberish](https://github.com/ggml-org/llama.cpp/pull/25202).**
3. **[Four-bit weights multiplied tool-name errors](https://arxiv.org/abs/2607.27275).**
4. **[INT4 and INT3 inflated reasoning traces](https://arxiv.org/abs/2606.25519).**
5. **[Dynamic FP8 erased its own memory saving](https://github.com/vllm-project/vllm/issues/19855).**
6. **[Quantized CPU offloading ignored the GPU memory budget](https://github.com/huggingface/transformers/issues/43873).**
7. **[A 2:4-plus-W4A16 kernel corrupted output](https://github.com/vllm-project/vllm/issues/10819).**
8. **[One 2:4 sparse operation took 2.7× as long as dense](https://github.com/pytorch/pytorch/issues/153825).**
9. **[At two bits, the answer signal collapsed inside tested models](https://arxiv.org/abs/2604.19884).**
10. **[A bad Q4 conversion quantized the wrong attention tensor](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/5).**

## Repository map

| Path | Contents |
|---|---|
| [`report.md`](report.md) | Complete, source-linked report |
| [`report.pdf`](report.pdf) | Portable PDF edition of the same report |
| [`data/publication_summary.json`](data/publication_summary.json) | Frozen aggregate summary for the 1,640-record corpus |
| [`assets/`](assets/) | Publication figures generated from the corpus |
| [`scripts/build_readme_assets.py`](scripts/build_readme_assets.py) | Deterministic summary and figure builder |
| [`../results/sparse_inference_failure_mining/`](../results/sparse_inference_failure_mining/) | Raw candidates, normalized incidents, census, atlas, and research report |
| [`../code/scripts/`](../code/scripts/) | Corpus construction and summarization scripts |

## Corpus boundary

The aggregate corpus is frozen at **2026-08-08**. The Transformers offloading failure was found during a later recall audit and is included as a report addendum, but not in the 1,640-record aggregate totals.

## Reproduce the corpus and figures

The live-web discovery and manual normalization steps are frozen inputs. The
commands below deterministically rerun schema validation, lineage joining,
quality gates, ranking, curated-atlas construction, and publication figures
without depending on current search results or mutable source pages.

From the repository root, create an isolated output directory:

```bash
OUT=\"${TMPDIR:-/tmp}/prism-sparse-repro\"
mkdir -p \"$OUT/assets\"
```

Rebuild the ranked 25-report summary:

```bash
python3 code/scripts/summarize_sparse_inference_failures.py \
  --reports results/sparse_inference_failure_mining/reports.json \
  --scope results/sparse_inference_failure_mining/scope.json \
  --output \"$OUT/summary.json\"
cmp \"$OUT/summary.json\" \
  results/sparse_inference_failure_mining/summary.json
```

Rebuild the curated 25-record incident atlas and its coverage receipt:

```bash
python3 code/scripts/build_sparse_inference_incident_atlas.py \
  --reports results/sparse_inference_failure_mining/reports.json \
  --annotations results/sparse_inference_failure_mining/atlas_annotations.json \
  --csv-output \"$OUT/incident_atlas.csv\" \
  --summary-output \"$OUT/incident_atlas_coverage.json\"
cmp \"$OUT/incident_atlas.csv\" \
  results/sparse_inference_failure_mining/incident_atlas.csv
cmp \"$OUT/incident_atlas_coverage.json\" \
  results/sparse_inference_failure_mining/incident_atlas_coverage.json
```

Rebuild the 1,640-row complaint census from the 25 curated reports and 1,615
normalized source records:

```bash
python3 code/scripts/build_sparse_inference_complaint_census.py \
  --reports results/sparse_inference_failure_mining/reports.json \
  --annotations results/sparse_inference_failure_mining/atlas_annotations.json \
  --legacy-atlas results/sparse_inference_failure_mining/incident_atlas.csv \
  --normalized results/sparse_inference_failure_mining/manual_normalized_incidents.json \
  --normalized results/sparse_inference_failure_mining/github_normalized_incidents.json \
  --normalized results/sparse_inference_failure_mining/reddit_normalized_incidents.json \
  --normalized results/sparse_inference_failure_mining/hf_normalized_incidents.json \
  --normalized results/sparse_inference_failure_mining/hackernews_normalized_incidents.json \
  --normalized results/sparse_inference_failure_mining/additional_paper_incidents.json \
  --window-start 2022-08-15 \
  --as-of 2026-08-08 \
  --minimum-lineages 250 \
  --json-output \"$OUT/complaint_census.json\" \
  --csv-output \"$OUT/complaint_census.csv\" \
  --coverage-output \"$OUT/complaint_census_coverage.json\" \
  --unresolved-output \"$OUT/unresolved_concerns.csv\"
```

The census command must report:

```text
validated 1640 source-bound rows across 1588 incident lineages
```

Verify every rebuilt corpus artifact byte for byte:

```bash
cmp \"$OUT/complaint_census.json\" \
  results/sparse_inference_failure_mining/complaint_census.json
cmp \"$OUT/complaint_census.csv\" \
  results/sparse_inference_failure_mining/complaint_census.csv
cmp \"$OUT/complaint_census_coverage.json\" \
  results/sparse_inference_failure_mining/complaint_census_coverage.json
cmp \"$OUT/unresolved_concerns.csv\" \
  results/sparse_inference_failure_mining/unresolved_concerns.csv
```

Rebuild and verify the public aggregate summary and SVG figures:

```bash
python3 sparse-inference-failure-atlas/scripts/build_readme_assets.py \
  --input \"$OUT/complaint_census.json\" \
  --data-output \"$OUT/publication_summary.json\" \
  --assets-dir \"$OUT/assets\"
cmp \"$OUT/publication_summary.json\" \
  sparse-inference-failure-atlas/data/publication_summary.json
cmp \"$OUT/assets/evidence-bottleneck.svg\" \
  sparse-inference-failure-atlas/assets/evidence-bottleneck.svg
cmp \"$OUT/assets/nonsense-is-not-a-diagnosis.svg\" \
  sparse-inference-failure-atlas/assets/nonsense-is-not-a-diagnosis.svg
```

The Markdown report is the editable source. `report.pdf` is the checked,
six-page publication rendering; it embeds only portable HTTPS source
annotations.
