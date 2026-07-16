#!/usr/bin/env bash
set -euo pipefail

# Issue #22 paid-run queue. Model downloads and environment installation must
# finish before this script starts so every metered minute is compute-bearing.

REPO_ROOT="${REPO_ROOT:-/root/prism-capability-extraction}"
INPUT_ROOT="${INPUT_ROOT:-/root/issue22_inputs}"
RUN_ROOT="${RUN_ROOT:-/root/issue22_run}"
BASE_MODEL="${BASE_MODEL:-${INPUT_ROOT}/qwen25_math_1p5b_base}"
MERGED_MODEL="${MERGED_MODEL:-${INPUT_ROOT}/circuit_discovery_model/checkpoints/checkpoint-b}"
EVAL_BATCH="${EVAL_BATCH:-256}"
SWEEP_LIMIT="${SWEEP_LIMIT:-1024}"
PEAK_BF16_TFLOPS="${PEAK_BF16_TFLOPS:-2250}"

DATA_ROOT="${INPUT_ROOT}/hf_dataset/circuit-shotting/artifacts/pod_logs/steed-medium/full"
MAX_ROOT="${DATA_ROOT}/qwen25_math_1p5b_2digit_max_recovery_v1"
VALIDATE_ROOT="${DATA_ROOT}/qwen25_math_1p5b_2digit_rank32_validate_v1"
ADAPTER="${MAX_ROOT}/lora_r32_beta005/adapter"
COMPOSED="${MAX_ROOT}/position_adacs_lora_r32_beta005/composed_union.full.npz"
TOPK="${VALIDATE_ROOT}/r32_topk500.full.npz"
RANKING="${REPO_ROOT}/results/arithmetic/r32_direct_group_rank_merged.json"
HUNDREDS="${VALIDATE_ROOT}/fresh_pairs_seed123/hundreds_pairs.json"
TENS="${VALIDATE_ROOT}/fresh_pairs_seed123/tens_pairs.json"
ONES="${VALIDATE_ROOT}/fresh_pairs_seed123/ones_pairs.json"
SCRIPTS="${REPO_ROOT}/code/scripts"
CONFIG="${REPO_ROOT}/code/configs/issue22_arithmetic_runtime_candidates.json"

mkdir -p "${RUN_ROOT}/receipts" "${RUN_ROOT}/utilization"
cd "${REPO_ROOT}"

git rev-parse HEAD > "${RUN_ROOT}/receipts/git_commit.txt"
nvidia-smi -q > "${RUN_ROOT}/receipts/nvidia_smi_q.txt"
nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,driver_version,power.limit,compute_cap \
  --format=csv,noheader \
  > "${RUN_ROOT}/receipts/gpu_identity.csv"
python -VV > "${RUN_ROOT}/receipts/python_version.txt" 2>&1
python -m pip freeze > "${RUN_ROOT}/receipts/pip_freeze.txt"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv \
  --loop=1 \
  > "${RUN_ROOT}/utilization/compute_queue.csv" &
SAMPLER_PID=$!
cleanup_sampler() {
  kill "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}" 2>/dev/null || true
}
trap cleanup_sampler EXIT

python "${SCRIPTS}/verify_arithmetic_merge_provenance.py" \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --candidate "${MERGED_MODEL}" \
  --scale 0.55 \
  --dtype bfloat16 \
  --device cuda:0 \
  --output "${RUN_ROOT}/receipts/merge_provenance.json"

python "${SCRIPTS}/evaluate_arithmetic_standalone.py" \
  --mode dense \
  --model "${MERGED_MODEL}" \
  --historical-pair "hundreds=${HUNDREDS}" \
  --historical-pair "tens=${TENS}" \
  --historical-pair "ones=${ONES}" \
  --historical-final-n 500 \
  --batch-size "${EVAL_BATCH}" \
  --max-new-tokens 8 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attention-implementation eager \
  --output-dir "${RUN_ROOT}/dense"

python "${SCRIPTS}/search_arithmetic_standalone_mace.py" \
  --model "${MERGED_MODEL}" \
  --records-jsonl "${RUN_ROOT}/dense/records.jsonl" \
  --seed-mask "topk12661=${TOPK}:mlp_final" \
  --ceiling-mask "rel001=${COMPOSED}:mlp_rel_0.001" \
  --safety-mask "positive=${COMPOSED}:mlp_positive" \
  --ranking-json "${RANKING}" \
  --batch-size "${EVAL_BATCH}" \
  --max-new-tokens 8 \
  --recovery-floor 0.90 \
  --max-rounds 20 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attention-implementation eager \
  --output-dir "${RUN_ROOT}/search"

PHYSICAL_ROOT="${RUN_ROOT}/physical_candidates"
PHYSICAL_QUEUE="${PHYSICAL_ROOT}/logical_mace_queue.tsv"
PHYSICAL_ATTEMPTS="${PHYSICAL_ROOT}/attempts.jsonl"
PHYSICAL_SELECTION="${RUN_ROOT}/physical_selection.json"
PHYSICAL_GATE_FAILURE_EXIT_CODE=3
mkdir -p "${PHYSICAL_ROOT}"
: > "${PHYSICAL_ATTEMPTS}"

# Preserve search/winner as the bounded logical winner. Physicalization is a
# separate frontier because BF16 shape changes need not preserve that winner's
# standalone predictions.
jq -er '
  [.rounds[] | select(.passes_mace == true)]
  | sort_by(.kept, -(.comparison.matched_dense_recovery), .round)
  | if length == 0 then
      error("logical search produced no MACE-passing physical candidates")
    else
      .[]
      | [.round, .label, .mask_path, .predictions_path, .kept]
      | @tsv
    end
' "${RUN_ROOT}/search/manifest.json" > "${PHYSICAL_QUEUE}"

SELECTED_ROUND=""
SELECTED_DIR=""
while IFS=$'\t' read -r ROUND LABEL MASK LOGICAL_PREDICTIONS KEPT; do
  CANDIDATE_DIR="${PHYSICAL_ROOT}/round${ROUND}"
  CANDIDATE_BUNDLE="${CANDIDATE_DIR}/bundle"
  CANDIDATE_EVAL="${CANDIDATE_DIR}/eval"
  CANDIDATE_PARITY="${CANDIDATE_DIR}/parity.json"
  CANDIDATE_DIFFERENCES="${CANDIDATE_DIR}/parity_differences.jsonl"
  mkdir -p "${CANDIDATE_DIR}"

  python "${SCRIPTS}/build_arithmetic_physical_bundle.py" \
    --model "${MERGED_MODEL}" \
    --mask "${MASK}" \
    --candidate-id "issue22-physical-round${ROUND}-${LABEL}" \
    --dtype bfloat16 \
    --device cuda:0 \
    --output "${CANDIDATE_BUNDLE}"

  python "${SCRIPTS}/evaluate_arithmetic_standalone.py" \
    --mode physical \
    --bundle "${CANDIDATE_BUNDLE}" \
    --historical-pair "hundreds=${HUNDREDS}" \
    --historical-pair "tens=${TENS}" \
    --historical-pair "ones=${ONES}" \
    --historical-final-n 500 \
    --batch-size "${EVAL_BATCH}" \
    --max-new-tokens 8 \
    --device cuda:0 \
    --output-dir "${CANDIDATE_EVAL}"

  COMPARE_STATUS=0
  python "${SCRIPTS}/compare_arithmetic_physical_parity.py" \
    --logical "${LOGICAL_PREDICTIONS}" \
    --physical "${CANDIDATE_EVAL}/predictions.jsonl" \
    --dense "${RUN_ROOT}/dense/predictions.jsonl" \
    --retention-floor 0.99 \
    --recovery-floor 0.90 \
    --output "${CANDIDATE_PARITY}" \
    --differences "${CANDIDATE_DIFFERENCES}" \
    || COMPARE_STATUS=$?

  case "${COMPARE_STATUS}" in
    0)
      jq -e '.status == "pass"' "${CANDIDATE_PARITY}" > /dev/null
      ;;
    "${PHYSICAL_GATE_FAILURE_EXIT_CODE}")
      jq -e '.status == "fail"' "${CANDIDATE_PARITY}" > /dev/null
      ;;
    *)
      echo \
        "physical comparator failed unexpectedly for round ${ROUND}: " \
        "exit ${COMPARE_STATUS}" >&2
      exit "${COMPARE_STATUS}"
      ;;
  esac

  jq -cn \
    --argjson round "${ROUND}" \
    --arg bundle "${CANDIDATE_BUNDLE}" \
    --arg evaluation "${CANDIDATE_EVAL}" \
    --arg parity "${CANDIDATE_PARITY}" \
    --arg differences "${CANDIDATE_DIFFERENCES}" \
    --slurpfile search "${RUN_ROOT}/search/manifest.json" \
    --slurpfile acceptance "${CANDIDATE_PARITY}" \
    --slurpfile metadata \
      "${CANDIDATE_BUNDLE}/substrate_metadata.json" \
    '
      ($search[0].rounds | map(select(.round == $round)) | first) as $logical
      | $acceptance[0] as $physical
      | $metadata[0] as $bundle_metadata
      | if $logical == null then
          error("physical candidate round is absent from search manifest")
        else
          {
            round: $logical.round,
            search_label: $logical.label,
            kept: $logical.kept,
            available: $logical.available,
            kept_fraction: $logical.kept_fraction,
            search_selection_sha256: $logical.selection_sha256,
            search_mask_sha256: $logical.mask_sha256,
            logical_candidate_correct:
              $logical.comparison.candidate_correct,
            logical_matched_dense_correct:
              $logical.comparison.matched_dense_correct,
            logical_matched_dense_recovery:
              $logical.comparison.matched_dense_recovery,
            logical_passes_mace: $logical.passes_mace,
            physical_correct: $physical.physical_correct,
            matched_dense_correct:
              $physical.dense_recovery.matched_dense_correct,
            matched_dense_recovery:
              $physical.dense_recovery.matched_dense_recovery,
            passes_mace90:
              ($physical.dense_recovery.status == "pass"),
            logical_correctness_retention:
              $physical.logical_correctness_retention,
            passes_parity99: ($physical.parity_status == "pass"),
            accepted: ($physical.status == "pass"),
            gates: $physical.gates,
            physical_checkpoint_sha256:
              $bundle_metadata.physicalization.checkpoint_sha256,
            physical_parameters:
              $bundle_metadata.physicalization.physical_parameters,
            artifacts: {
              search_mask: $logical.mask_path,
              logical_predictions: $logical.predictions_path,
              bundle: $bundle,
              evaluation: $evaluation,
              parity: $parity,
              parity_differences: $differences
            }
          }
        end
    ' >> "${PHYSICAL_ATTEMPTS}"

  if [[ "${COMPARE_STATUS}" -eq 0 ]]; then
    SELECTED_ROUND="${ROUND}"
    SELECTED_DIR="${CANDIDATE_DIR}"
    break
  fi
done < "${PHYSICAL_QUEUE}"

jq -s \
  --arg selected_round "${SELECTED_ROUND}" \
  --arg search_manifest "${RUN_ROOT}/search/manifest.json" \
  --arg candidate_queue "${PHYSICAL_QUEUE}" \
  --arg logical_winner_mask "${RUN_ROOT}/search/winner/mask.npz" \
  --arg logical_winner_predictions \
    "${RUN_ROOT}/search/winner/predictions.jsonl" \
  --slurpfile search "${RUN_ROOT}/search/manifest.json" \
  '
    . as $attempts
    | $search[0].winner as $logical_winner
    | {
        schema_version: "prism_arithmetic_physical_selection_v1",
        status: (
          if $selected_round == "" then "fail" else "pass" end
        ),
        selection_rule: (
          "fewest channels among bounded logical MACE-pass candidates whose "
          + "physical form passes MACE-90 and retains at least 99% of "
          + "logical-zero correct rows"
        ),
        candidate_order: (
          "kept ascending, logical matched-dense recovery descending, "
          + "search round ascending"
        ),
        stop_rule: "first candidate passing both frozen physical gates",
        floors: {
          physical_matched_dense_recovery: 0.90,
          logical_correctness_retention: 0.99
        },
        dense_recovery_denominator:
          "correct rows in the frozen standalone dense evaluation",
        eligible_logical_candidate_count: (
          $search[0].rounds
          | map(select(.passes_mace == true))
          | length
        ),
        candidate_queue_artifact: $candidate_queue,
        logical_winner_preserved: true,
        logical_winner: {
          semantics: "bounded standalone logical-zero MACE winner",
          round: $logical_winner.round,
          label: $logical_winner.label,
          kept: $logical_winner.kept,
          matched_dense_recovery:
            $logical_winner.comparison.matched_dense_recovery,
          artifacts: {
            mask: $logical_winner_mask,
            predictions: $logical_winner_predictions,
            search_manifest: $search_manifest
          }
        },
        bounded_minimality: ($selected_round != ""),
        global_minimality_claimed: false,
        tested_candidate_count: ($attempts | length),
        untested_larger_candidate_count: (
          (
            $search[0].rounds
            | map(select(.passes_mace == true))
            | length
          ) - ($attempts | length)
        ),
        tested_candidates: $attempts,
        selected: (
          if $selected_round == "" then
            null
          else
            (
              $attempts
              | map(select((.round | tostring) == $selected_round))
              | first
            )
          end
        )
      }
  ' "${PHYSICAL_ATTEMPTS}" > "${PHYSICAL_SELECTION}.partial"
mv "${PHYSICAL_SELECTION}.partial" "${PHYSICAL_SELECTION}"

if [[ -z "${SELECTED_DIR}" ]]; then
  echo \
    "no bounded logical MACE candidate passed the frozen physical gates; " \
    "see ${PHYSICAL_SELECTION}" >&2
  exit 1
fi

SELECTED_BUNDLE="$(
  jq -er '.selected.artifacts.bundle' "${PHYSICAL_SELECTION}"
)"
SELECTED_EVAL="$(
  jq -er '.selected.artifacts.evaluation' "${PHYSICAL_SELECTION}"
)"

QUALITY_CORRECT="$(
  jq -s --argjson limit "${SWEEP_LIMIT}" \
    '.[0:$limit] | map(select(.exact_numeric_correct == true)) | length' \
    "${SELECTED_EVAL}/predictions.jsonl"
)"

python "${SCRIPTS}/benchmark_arithmetic_physical_throughput.py" \
  --bundle "${SELECTED_BUNDLE}" \
  --dense-model "${MERGED_MODEL}" \
  --records "${RUN_ROOT}/dense/records.jsonl" \
  --candidate-config "${CONFIG}" \
  --batch-sizes 16 32 64 128 256 512 1024 \
  --max-new-tokens 8 \
  --limit "${SWEEP_LIMIT}" \
  --warmup 1 \
  --repeats 1 \
  --latency-repeats 1 \
  --quality-floor-correct "${QUALITY_CORRECT}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --output "${RUN_ROOT}/cheap_sweep.json" \
  --attempts-jsonl "${RUN_ROOT}/cheap_sweep_attempts.jsonl"

cleanup_sampler
trap - EXIT

python - "${RUN_ROOT}/utilization/compute_queue.csv" \
  "${RUN_ROOT}/utilization/compute_queue_summary.json" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
rows = list(csv.DictReader(source.open()))

def number(value):
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return float(match.group()) if match else None

gpu = [number(row.get(" utilization.gpu [%]")) for row in rows]
power = [number(row.get(" power.draw [W]")) for row in rows]
gpu = [value for value in gpu if value is not None]
power = [value for value in power if value is not None]
summary = {
    "status": "pass" if gpu else "no_samples",
    "source": str(source),
    "samples": len(gpu),
    "gpu_utilization_percent": {
        "mean": sum(gpu) / len(gpu) if gpu else None,
        "max": max(gpu) if gpu else None,
        "samples_at_or_above_90_percent": (
            sum(value >= 90 for value in gpu) if gpu else None
        ),
    },
    "power_draw_watts": {
        "mean": sum(power) / len(power) if power else None,
        "max": max(power) if power else None,
    },
    "boundary": (
        "one-second nvidia-smi samples across merge verification, dense and "
        "logical evaluation, physicalization, parity, and cheap runtime sweep"
    ),
}
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Issue #22 opening queue complete: ${RUN_ROOT}"
