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
FINE="${VALIDATE_ROOT}/compress_fine_seed123_min995.final_mask.full.npz"
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
  --seed-mask "fine58619=${FINE}:mlp_final" \
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

python "${SCRIPTS}/build_arithmetic_physical_bundle.py" \
  --model "${MERGED_MODEL}" \
  --mask "${RUN_ROOT}/search/winner/mask.npz" \
  --candidate-id issue22-mace90-winner \
  --dtype bfloat16 \
  --device cuda:0 \
  --output "${RUN_ROOT}/physical_bundle"

python "${SCRIPTS}/evaluate_arithmetic_standalone.py" \
  --mode physical \
  --bundle "${RUN_ROOT}/physical_bundle" \
  --historical-pair "hundreds=${HUNDREDS}" \
  --historical-pair "tens=${TENS}" \
  --historical-pair "ones=${ONES}" \
  --historical-final-n 500 \
  --batch-size "${EVAL_BATCH}" \
  --max-new-tokens 8 \
  --device cuda:0 \
  --output-dir "${RUN_ROOT}/physical_eval"

python "${SCRIPTS}/compare_arithmetic_physical_parity.py" \
  --logical "${RUN_ROOT}/search/winner/predictions.jsonl" \
  --physical "${RUN_ROOT}/physical_eval/predictions.jsonl" \
  --retention-floor 0.99 \
  --output "${RUN_ROOT}/physical_parity.json" \
  --differences "${RUN_ROOT}/physical_parity_differences.jsonl"

QUALITY_CORRECT="$(
  jq -s --argjson limit "${SWEEP_LIMIT}" \
    '.[0:$limit] | map(select(.exact_numeric_correct == true)) | length' \
    "${RUN_ROOT}/physical_eval/predictions.jsonl"
)"

python "${SCRIPTS}/benchmark_arithmetic_physical_throughput.py" \
  --bundle "${RUN_ROOT}/physical_bundle" \
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
