#!/usr/bin/env bash
set -Eeuo pipefail

TARGET="${LIUM_TARGET:-tokenbender/prism-bfcl-issue5-h200}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/tokenbender-prism}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/tokenbender-prism-runs}"
REMOTE_LAUNCH="${REMOTE_RUNS}/issue5_launch_full.sh"
SESSION="${TMUX_SESSION:-issue5_bfcl_nearmiss}"
WANDB_ENTITY_DEFAULT="${WANDB_ENTITY:-ahm-rimer}"
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT:-prism-bfcl}"
MAX_AUG_ROUNDS="${MAX_AUG_ROUNDS:-3}"
AUG_RATIO="${AUG_RATIO:-0.20}"
PLATEAU_GAIN_EXAMPLES="${PLATEAU_GAIN_EXAMPLES:-15}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

need lium
need python3

echo "== Lium pod =="
lium ps
echo

if ! lium exec "$TARGET" "true" >/dev/null; then
  echo "Pod target not found in lium ps: $TARGET" >&2
  echo "Set LIUM_TARGET=<pod-name> and rerun if the pod has a different name." >&2
  exit 3
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

WANDB_MODE_INPUT="${WANDB_MODE:-online}"
HF_TOKEN_INPUT="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "$HF_TOKEN_INPUT" && -s "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN_INPUT="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
fi
if [[ -z "$HF_TOKEN_INPUT" ]]; then
  echo "No Hugging Face token found in env or local cache; aborting before touching the pod." >&2
  exit 5
fi

WANDB_API_KEY_INPUT="${WANDB_API_KEY:-}"
if [[ -z "$WANDB_API_KEY_INPUT" && -s "$HOME/.netrc" ]]; then
  WANDB_API_KEY_INPUT="$(
    python3 - <<'PY'
import netrc
from pathlib import Path

path = Path.home() / ".netrc"
try:
    parsed = netrc.netrc(str(path))
except Exception:
    raise SystemExit(0)
for host in ("api.wandb.ai", "wandb.ai"):
    auth = parsed.authenticators(host)
    if auth and auth[2]:
        print(auth[2])
        break
PY
  )"
fi
if [[ -z "$WANDB_API_KEY_INPUT" && "$WANDB_MODE_INPUT" == "online" ]]; then
  echo "No W&B key found; downgrading WANDB_MODE to disabled for this launch." >&2
  WANDB_MODE_INPUT="disabled"
fi

WANDB_API_KEY_INPUT="$WANDB_API_KEY_INPUT" \
WANDB_ENTITY_INPUT="$WANDB_ENTITY_DEFAULT" \
WANDB_PROJECT_INPUT="$WANDB_PROJECT_DEFAULT" \
WANDB_MODE_INPUT="$WANDB_MODE_INPUT" \
HF_TOKEN_INPUT="$HF_TOKEN_INPUT" \
python3 - "$tmpdir/issue5_env" <<'PY'
import os
import shlex
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines = {
    "WANDB_ENTITY": os.environ["WANDB_ENTITY_INPUT"],
    "WANDB_PROJECT": os.environ["WANDB_PROJECT_INPUT"],
    "WANDB_MODE": os.environ["WANDB_MODE_INPUT"],
    "HF_TOKEN": os.environ["HF_TOKEN_INPUT"],
    "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN_INPUT"],
}
if os.environ.get("WANDB_API_KEY_INPUT"):
    lines["WANDB_API_KEY"] = os.environ["WANDB_API_KEY_INPUT"]
out.write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in lines.items()))
out.chmod(0o600)
PY

cat >"$tmpdir/issue5_launch_full.sh" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_REPO="${REMOTE_REPO:-/workspace/tokenbender-prism}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/tokenbender-prism-runs}"
MAX_AUG_ROUNDS="${MAX_AUG_ROUNDS:-3}"
AUG_RATIO="${AUG_RATIO:-0.20}"
PLATEAU_GAIN_EXAMPLES="${PLATEAU_GAIN_EXAMPLES:-15}"
RUN_DIR="${REMOTE_REPO}/runs/issue5_bfcl_nearmiss_loop"
DATA_DIR="${REMOTE_REPO}/data/bfcl_issue5_nearmiss_loop"
LOG_DIR="${REMOTE_RUNS}/issue5_logs"
HF_STAGE="${REMOTE_RUNS}/hf_stage/issue5_nearmiss_loop_v1"
FULL_LOG="${LOG_DIR}/issue5_full_pipeline.log"
CONFIG="${REMOTE_REPO}/code/configs/bfcl_issue5_nearmiss_loop.json"

source /root/issue5_env
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-prism-bfcl}"
export WANDB_ENTITY="${WANDB_ENTITY:-ahm-rimer}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REMOTE_REPO}/code:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$DATA_DIR"

round_dir() {
  echo "${RUN_DIR}/$1"
}

round_train_dir() {
  echo "$(round_dir "$1")/unmasked_r32"
}

round_attr() {
  echo "$(round_dir "$1")/relp_full_collimated.npz"
}

round_train_jsonl() {
  local round_id="$1"
  if [[ "$round_id" == "r0" ]]; then
    echo "${REMOTE_REPO}/data/bfcl_strict_10k_mix_len1024/train.jsonl"
  else
    echo "${DATA_DIR}/${round_id}/train_mixed.jsonl"
  fi
}

setup_env() {
  cd "$REMOTE_REPO"
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv --system-site-packages .venv
  fi
  .venv/bin/python -m pip install -U pip setuptools wheel
  .venv/bin/python -m pip install -U \
    "transformers>=4.40" \
    "datasets>=2.18" \
    "huggingface-hub>=0.23" \
    "wandb>=0.15" \
    "accelerate>=0.30" \
    "peft>=0.19.1" \
    "safetensors" \
    "numpy"
  .venv/bin/python -m pip install --no-deps -e code
}

check_auth_and_env() {
  cd "$REMOTE_REPO"
  .venv/bin/python - <<'PY'
import os
import torch
from huggingface_hub import HfApi

mods = ["torch", "transformers", "peft", "datasets", "huggingface_hub", "wandb", "accelerate", "numpy"]
for name in mods:
    mod = __import__(name)
    print(f"{name} {getattr(mod, '__version__', 'ok')}")
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
hf = HfApi().whoami()
print("hf_user", hf.get("name") or "ok")
print("wandb_mode", os.environ.get("WANDB_MODE"))
print("wandb_key", "present" if os.environ.get("WANDB_API_KEY") else "missing")
PY
}

restore_data() {
  cd "$REMOTE_REPO"
  if [[ ! -s data/bfcl_single_call/pairs.jsonl ]]; then
    .venv/bin/python code/scripts/bfcl_direct_qwen3.py download-bfcl-single-call \
      --output-dir data/bfcl_single_call \
      --output data/bfcl_single_call/pairs.jsonl \
      --manifest data/bfcl_single_call/manifest.json
  fi
  if [[ ! -s data/bfcl_strict_10k_mix_len1024/train.jsonl ]]; then
    .venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

items = [
    ("data/bfcl_strict_10k_mix_len1024/train.jsonl", "data/bfcl_strict_10k_mix_len1024/train.jsonl"),
    ("data/bfcl_strict_10k_mix/manifest.json", "data/bfcl_strict_10k_mix/manifest.json"),
]
for filename, out in items:
    src = hf_hub_download("Occupying-Mars/issue49-bfcl-repro-artifacts", filename, repo_type="dataset")
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(Path(src).read_bytes())
    print(out, outp.stat().st_size)
PY
  fi
}

run_leak_audit() {
  local train_jsonl="$1"
  local out="$2"
  cd "$REMOTE_REPO"
  .venv/bin/python code/scripts/audit_bfcl_train_eval_overlap.py \
    --train-jsonl "$train_jsonl" \
    --eval-jsonl data/bfcl_single_call/pairs.jsonl \
    --output "$out" \
    --near-threshold 0.85 \
    --fail-on-overlap
}

train_round() {
  local round_id="$1"
  local train_jsonl="$2"
  local out_dir
  out_dir="$(round_train_dir "$round_id")"
  cd "$REMOTE_REPO"
  if [[ -s "${out_dir}/train_summary.json" && -d "${out_dir}/adapter" && -d "${out_dir}/merged" ]]; then
    echo "[skip] ${round_id} training exists"
    return
  fi
  .venv/bin/python code/scripts/train_bfcl_unmasked_lora.py \
    --model Qwen/Qwen3-8B \
    --train-jsonl "$train_jsonl" \
    --out-dir "$out_dir" \
    --experiment-id "bfcl_issue5_${round_id}_nearmiss_loop" \
    --github-issue 5 \
    --dtype bfloat16 \
    --device-map auto \
    --max-seq-length 1024 \
    --epochs 1.0 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.0 \
    --use-rslora \
    --policy-kl-beta 1.0 \
    --ce-beta 0.2 \
    --eval-every 25 \
    --save-every 0 \
    --save-merged \
    --wandb-mode "$WANDB_MODE" \
    --wandb-entity "$WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group issue-5 \
    --wandb-job-type train \
    --wandb-tags "bfcl,issue-5,nearmiss,${round_id},qwen3-8b"
}

attribute_round() {
  local round_id="$1"
  local attr
  attr="$(round_attr "$round_id")"
  cd "$REMOTE_REPO"
  if [[ -s "$attr" ]]; then
    echo "[skip] ${round_id} attribution exists"
    return
  fi
  .venv/bin/python code/scripts/bfcl_direct_qwen3.py relp-attribute \
    --pairs data/bfcl_single_call/pairs.jsonl \
    --output "$attr" \
    --model "$(round_train_dir "$round_id")/merged" \
    --dtype bfloat16 \
    --device-map auto \
    --log-every 10 \
    --report-topk 50
}

eval_round() {
  local round_id="$1"
  local dir attr
  dir="$(round_dir "$round_id")"
  attr="$(round_attr "$round_id")"
  cd "$REMOTE_REPO"
  for topk in 80000 120000 160000 200000 240000; do
    local out="${dir}/eval_k${topk}_masked.jsonl"
    if [[ ! -s "${out%.jsonl}.summary.json" ]]; then
      .venv/bin/python code/scripts/bfcl_direct_qwen3.py eval-mask \
        --pairs data/bfcl_single_call/pairs.jsonl \
        --output "$out" \
        --attribution "$attr" \
        --topk "$topk" \
        --model "$(round_train_dir "$round_id")/merged" \
        --dtype bfloat16 \
        --device-map auto \
        --batch-size 8 \
        --bfcl-canonicalization-prompt \
        --normalized
    else
      echo "[skip] ${round_id} eval exists for k=${topk}"
    fi
    local bucket_dir="${dir}/failure_buckets_k${topk}"
    if [[ ! -d "$bucket_dir" ]]; then
      .venv/bin/python code/scripts/build_bfcl_failure_buckets.py \
        --eval-jsonl "$out" \
        --pairs-jsonl data/bfcl_single_call/pairs.jsonl \
        --out-dir "$bucket_dir" \
        --run-name "issue5_${round_id}_k${topk}"
    fi
  done
}

write_round_summary() {
  local round_id="$1"
  local train_jsonl="$2"
  local aug_manifest="${3:-}"
  local leak_audit="${4:-}"
  cd "$REMOTE_REPO"
  ROUND_ID="$round_id" TRAIN_JSONL="$train_jsonl" AUG_MANIFEST="$aug_manifest" LEAK_AUDIT="$leak_audit" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

round_id = os.environ["ROUND_ID"]
run_dir = Path("runs/issue5_bfcl_nearmiss_loop") / round_id
anchor = 664
main_topks = {"160000", "200000", "240000"}
summary = {
    "round_id": round_id,
    "train_jsonl": os.environ["TRAIN_JSONL"],
    "train_rows": sum(1 for line in Path(os.environ["TRAIN_JSONL"]).read_text().splitlines() if line.strip()),
    "augmentation_manifest": os.environ.get("AUG_MANIFEST") or None,
    "leak_audit": os.environ.get("LEAK_AUDIT") or None,
    "train_summary": str(run_dir / "unmasked_r32" / "train_summary.json"),
    "attribution": str(run_dir / "relp_full_collimated.npz"),
    "evals": {},
}
for p in sorted(run_dir.glob("eval_k*_masked.summary.json")):
    topk = p.name.split("_masked.summary.json")[0].removeprefix("eval_k")
    row = json.loads(p.read_text())
    correct = row.get("normalized_exact_correct", row.get("exact_correct"))
    row["behavior_recovery_vs_full_anchor"] = correct / anchor if isinstance(correct, int) else None
    row["full_anchor_normalized_correct"] = anchor
    summary["evals"][topk] = row
main_scores = [
    row.get("normalized_exact_correct", row.get("exact_correct"))
    for topk, row in summary["evals"].items()
    if topk in main_topks
]
summary["main_frontier_best_correct"] = max(main_scores) if main_scores else None
if main_scores:
    best_topk = max(
        (topk for topk in summary["evals"] if topk in main_topks),
        key=lambda k: summary["evals"][k].get("normalized_exact_correct", summary["evals"][k].get("exact_correct", -1)),
    )
    summary["main_frontier_best_topk"] = int(best_topk)
(run_dir / "round_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

build_augmented_round() {
  local round_id="$1"
  local prev_round="$2"
  local base_train_jsonl="$3"
  local round_data="${DATA_DIR}/${round_id}"
  local round_run
  round_run="$(round_dir "$round_id")"
  mkdir -p "$round_data" "$round_run"
  cd "$REMOTE_REPO"
  .venv/bin/python code/scripts/build_bfcl_nearmiss_curriculum.py \
    --base-train-jsonl "$base_train_jsonl" \
    --eval-jsonl data/bfcl_single_call/pairs.jsonl \
    --failure-bucket-dirs \
      "$(round_dir "$prev_round")/failure_buckets_k160000" \
      "$(round_dir "$prev_round")/failure_buckets_k200000" \
      "$(round_dir "$prev_round")/failure_buckets_k240000" \
    --edge-output "${round_data}/edge.jsonl" \
    --mixed-output "${round_data}/train_mixed.jsonl" \
    --manifest "${round_data}/manifest.json" \
    --round-id "$round_id" \
    --augmentation-ratio "$AUG_RATIO" \
    --max-augmentation-ratio "$AUG_RATIO" \
    --seed "$((55 + ${round_id#r}))" \
    --fail-on-leak
  run_leak_audit "${round_data}/train_mixed.jsonl" "${round_run}/mixed_overlap_audit.json"
}

run_round() {
  local round_id="$1"
  local train_jsonl="$2"
  local aug_manifest="${3:-}"
  local leak_audit="${4:-}"
  mkdir -p "$(round_dir "$round_id")"
  train_round "$round_id" "$train_jsonl"
  attribute_round "$round_id"
  eval_round "$round_id"
  write_round_summary "$round_id" "$train_jsonl" "$aug_manifest" "$leak_audit"
}

write_decision() {
  cd "$REMOTE_REPO"
  MAX_AUG_ROUNDS="$MAX_AUG_ROUNDS" PLATEAU_GAIN_EXAMPLES="$PLATEAU_GAIN_EXAMPLES" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path("runs/issue5_bfcl_nearmiss_loop")
rounds = []
for p in sorted(run_dir.glob("r*/round_summary.json"), key=lambda p: int(p.parent.name[1:])):
    rounds.append(json.loads(p.read_text()))
issue2 = {"160000": 488, "200000": 579, "240000": 604}
max_aug = int(os.environ["MAX_AUG_ROUNDS"])
plateau_gain = int(os.environ["PLATEAU_GAIN_EXAMPLES"])
aug_rounds = [row for row in rounds if row["round_id"] != "r0"]
latest = rounds[-1] if rounds else None
promoted = False
promotion_hits = []
if latest:
    for topk, threshold in issue2.items():
        row = latest.get("evals", {}).get(topk)
        correct = row.get("normalized_exact_correct", row.get("exact_correct")) if row else None
        if isinstance(correct, int) and correct > threshold:
            promoted = True
            promotion_hits.append({"topk": int(topk), "correct": correct, "issue2": threshold})
gains = []
for prev, cur in zip(rounds, rounds[1:]):
    if cur["round_id"] == "r0":
        continue
    if prev.get("main_frontier_best_correct") is not None and cur.get("main_frontier_best_correct") is not None:
        gains.append({
            "from": prev["round_id"],
            "to": cur["round_id"],
            "gain": cur["main_frontier_best_correct"] - prev["main_frontier_best_correct"],
        })
last_two_low = len(gains) >= 2 and all(item["gain"] < plateau_gain for item in gains[-2:])
plateau = len(aug_rounds) >= 2 and last_two_low
max_rounds_reached = len(aug_rounds) >= max_aug
completed = promoted or plateau
decision = {
    "rounds_completed": [row["round_id"] for row in rounds],
    "latest_round": latest["round_id"] if latest else None,
    "promoted": promoted,
    "promotion_hits": promotion_hits,
    "plateau": plateau,
    "plateau_gain_examples": plateau_gain,
    "gains": gains,
    "max_augmentation_rounds": max_aug,
    "max_rounds_reached": max_rounds_reached,
    "completed": completed,
    "should_continue": (not completed) and (not max_rounds_reached),
    "stop_reason": "promoted" if promoted else ("plateau" if plateau else ("max_rounds_no_plateau" if max_rounds_reached else "continue")),
}
(run_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision, indent=2, sort_keys=True))
PY
}

write_final_summary() {
  cd "$REMOTE_REPO"
  .venv/bin/python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/issue5_bfcl_nearmiss_loop")
summary = {
    "experiment_id": "bfcl_issue5_nearmiss_loop_v1",
    "config": "code/configs/bfcl_issue5_nearmiss_loop.json",
    "baselines": {
        "full_unmasked_qwen3_8b_normalized": "664/1007",
        "issue2_k160": "488/1007",
        "issue2_k200": "579/1007",
        "issue2_k240": "604/1007",
        "issue4_k160": "487/1007",
        "issue4_k200": "520/1007",
        "issue4_k240": "540/1007",
    },
    "rounds": {},
    "decision": None,
}
for p in sorted(run_dir.glob("r*/round_summary.json"), key=lambda p: int(p.parent.name[1:])):
    row = json.loads(p.read_text())
    summary["rounds"][row["round_id"]] = row
decision = run_dir / "decision.json"
if decision.exists():
    summary["decision"] = json.loads(decision.read_text())
(run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

stage_artifacts() {
  cd "$REMOTE_REPO"
  rm -rf "$HF_STAGE"
  mkdir -p "$HF_STAGE/run" "$HF_STAGE/data" "$HF_STAGE/logs" "$HF_STAGE/configs"
  cp -f "$CONFIG" "$HF_STAGE/configs/bfcl_issue5_nearmiss_loop.json"
  cp -f docs/experiments/bfcl_issue5_nearmiss_loop.md "$HF_STAGE/configs/bfcl_issue5_nearmiss_loop.md"
  cp -f "$FULL_LOG" "$HF_STAGE/logs/issue5_full_pipeline.log" || true
  rsync -a \
    --exclude "*/merged" \
    --exclude "merged" \
    --exclude "checkpoints" \
    --exclude "checkpoints/*/merged" \
    "$RUN_DIR/" "$HF_STAGE/run/"
  rsync -a "$DATA_DIR/" "$HF_STAGE/data/"
  tar -C "$HF_STAGE" -czf "${REMOTE_RUNS}/issue5_nearmiss_loop_artifacts.tgz" .
  sha256sum "${REMOTE_RUNS}/issue5_nearmiss_loop_artifacts.tgz" \
    | tee "${REMOTE_RUNS}/issue5_nearmiss_loop_artifacts.tgz.sha256"
}

upload_hf() {
  cd "$REMOTE_REPO"
  .venv/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="TokenBender/circuit-discovery",
    repo_type="dataset",
    folder_path="/workspace/tokenbender-prism-runs/hf_stage/issue5_nearmiss_loop_v1",
    path_in_repo="bfcl/issue5_nearmiss_loop_v1",
    commit_message="Add BFCL issue 5 near-miss loop artifacts",
)
print("uploaded_to_hf", "TokenBender/circuit-discovery", "bfcl/issue5_nearmiss_loop_v1")
PY
}

run_pipeline() {
  cd "$REMOTE_REPO"
  echo "== issue5 BFCL near-miss loop pipeline =="
  date
  echo "max_aug_rounds=${MAX_AUG_ROUNDS}"
  echo "aug_ratio=${AUG_RATIO}"
  setup_env
  check_auth_and_env
  restore_data

  run_leak_audit data/bfcl_strict_10k_mix_len1024/train.jsonl "${RUN_DIR}/r0/leak_audit.json"
  run_round "r0" "$(round_train_jsonl r0)" "" "${RUN_DIR}/r0/leak_audit.json"

  prev_round="r0"
  prev_train="$(round_train_jsonl r0)"
  for n in $(seq 1 "$MAX_AUG_ROUNDS"); do
    round_id="r${n}"
    build_augmented_round "$round_id" "$prev_round" "$prev_train"
    train_jsonl="$(round_train_jsonl "$round_id")"
    run_round "$round_id" "$train_jsonl" "${DATA_DIR}/${round_id}/manifest.json" "$(round_dir "$round_id")/mixed_overlap_audit.json"
    write_decision
    if .venv/bin/python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("runs/issue5_bfcl_nearmiss_loop/decision.json").read_text())
raise SystemExit(0 if not d["should_continue"] else 1)
PY
    then
      break
    fi
    prev_round="$round_id"
    prev_train="$train_jsonl"
  done

  write_final_summary
  stage_artifacts
  upload_hf
  echo "== done =="
  date
}

mode="${1:-run}"
case "$mode" in
  check)
    setup_env
    check_auth_and_env
    ;;
  run)
    exec > >(tee -a "$FULL_LOG") 2>&1
    run_pipeline
    ;;
  *)
    echo "usage: $0 [check|run]" >&2
    exit 2
    ;;
esac
REMOTE

echo "== Preparing remote directories =="
lium exec "$TARGET" "mkdir -p '$REMOTE_REPO' '$REMOTE_RUNS' /root"

echo "== Syncing working tree =="
lium rsync "$TARGET" "$(pwd)/" "$REMOTE_REPO/"

echo "== Uploading credentials and launcher =="
lium scp "$TARGET" "$tmpdir/issue5_env" /root/issue5_env
lium scp "$TARGET" "$tmpdir/issue5_launch_full.sh" "$REMOTE_LAUNCH"
lium exec "$TARGET" "chmod 600 /root/issue5_env && chmod +x '$REMOTE_LAUNCH'"

echo "== Verifying pod environment =="
lium exec "$TARGET" "RUN_MODE=full REMOTE_REPO='$REMOTE_REPO' REMOTE_RUNS='$REMOTE_RUNS' MAX_AUG_ROUNDS='$MAX_AUG_ROUNDS' AUG_RATIO='$AUG_RATIO' PLATEAU_GAIN_EXAMPLES='$PLATEAU_GAIN_EXAMPLES' bash '$REMOTE_LAUNCH' check"

echo "== Launching remote tmux session: $SESSION =="
lium exec "$TARGET" "bash -lc 'tmux has-session -t \"$SESSION\" 2>/dev/null && { echo \"tmux session already exists: $SESSION\"; tmux ls; exit 0; }; tmux new-session -d -s \"$SESSION\" \"REMOTE_REPO=$REMOTE_REPO REMOTE_RUNS=$REMOTE_RUNS MAX_AUG_ROUNDS=$MAX_AUG_ROUNDS AUG_RATIO=$AUG_RATIO PLATEAU_GAIN_EXAMPLES=$PLATEAU_GAIN_EXAMPLES bash $REMOTE_LAUNCH run\"; tmux ls'"

cat <<EOF

Launched issue #5 near-miss loop.

Monitor:
  lium exec $TARGET "tail -f ${REMOTE_RUNS}/issue5_logs/issue5_full_pipeline.log"

Check session:
  lium exec $TARGET "tmux ls"
EOF
