#!/usr/bin/env bash
set -Eeuo pipefail

TARGET="${LIUM_TARGET:-tokenbender/prism-bfcl-issue6-editionx8}"
EXECUTOR="${LIUM_EXECUTOR:-noble-raven-91}"
POD_TTL="${POD_TTL:-48h}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/tokenbender-prism}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/tokenbender-prism-runs}"
REMOTE_LAUNCH="${REMOTE_RUNS}/issue6_launch_full.sh"
REMOTE_HEARTBEAT="${REMOTE_RUNS}/issue6_heartbeat.sh"
SESSION="${TMUX_SESSION:-issue6_bfcl_tree}"
HEARTBEAT_SESSION="${HEARTBEAT_SESSION:-issue6_heartbeat}"
WANDB_ENTITY_DEFAULT="${WANDB_ENTITY:-ahm-rimer}"
WANDB_PROJECT_DEFAULT="${WANDB_PROJECT:-prism-bfcl}"
MAX_BRANCH_ROUNDS="${MAX_BRANCH_ROUNDS:-20}"
PARALLEL_BRANCHES="${PARALLEL_BRANCHES:-8}"
BEAM_WIDTH="${BEAM_WIDTH:-3}"
TOPKS="${TOPKS:-40000 60000 80000 100000 120000 140000 160000 180000 200000 220000 240000}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch}"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

need git
need lium
need python3

echo "== Lium active pods =="
lium ps
echo

pod_present() {
  lium exec "$TARGET" "true" 2>&1 | grep -q "Executing on"
}

if ! pod_present; then
  active="$(lium ps)"
  if [[ "$active" != "No active pods" ]]; then
    echo "Unrelated active pod(s) exist; refusing to launch issue #6 until the surface is explicit." >&2
    echo "$active" >&2
    exit 3
  fi
  echo "== Creating selected 8xEdition pod: $EXECUTOR =="
  lium up "$EXECUTOR" --name "$TARGET" --ttl "$POD_TTL" --yes
fi

if ! pod_present; then
  echo "Pod target not reachable after create/check: $TARGET" >&2
  lium ps >&2
  exit 4
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
python3 - "$tmpdir/issue6_env" <<'PY'
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

cat >"$tmpdir/issue6_launch_full.sh" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

REMOTE_REPO="${REMOTE_REPO:-/workspace/tokenbender-prism}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/tokenbender-prism-runs}"
MAX_BRANCH_ROUNDS="${MAX_BRANCH_ROUNDS:-20}"
PARALLEL_BRANCHES="${PARALLEL_BRANCHES:-8}"
BEAM_WIDTH="${BEAM_WIDTH:-3}"
TOPKS="${TOPKS:-40000 60000 80000 100000 120000 140000 160000 180000 200000 220000 240000}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch}"
RUN_DIR="${REMOTE_REPO}/runs/issue6_bfcl_tree_search"
DATA_DIR="${REMOTE_REPO}/data/bfcl_issue6_tree_search"
LOG_DIR="${REMOTE_RUNS}/issue6_logs"
HF_STAGE="${REMOTE_RUNS}/hf_stage/issue6_tree_search_v1"
FULL_LOG="${LOG_DIR}/issue6_full_pipeline.log"
CONFIG="${REMOTE_REPO}/code/configs/bfcl_issue6_tree_search.json"

source /root/issue6_env
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-prism-bfcl}"
export WANDB_ENTITY="${WANDB_ENTITY:-ahm-rimer}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REMOTE_REPO}/code:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$DATA_DIR" "$RUN_DIR/plans" "$RUN_DIR/branches"

branch_dir() {
  echo "${RUN_DIR}/branches/$1"
}

branch_train_dir() {
  echo "$(branch_dir "$1")/unmasked_r32"
}

branch_data_dir() {
  echo "${DATA_DIR}/branches/$1"
}

branch_attr() {
  echo "$(branch_dir "$1")/relp_full_collimated.npz"
}

json_get() {
  local path="$1"
  local key="$2"
  .venv/bin/python - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
value = data
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
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
  .venv/bin/python -m pip install -U --force-reinstall "$TORCH_PACKAGE" --index-url "$TORCH_INDEX_URL"
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
print("cuda_device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print("gpu", idx, torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))
if torch.cuda.is_available():
    x = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    print("cuda_tensor_smoke", float(x.item()))
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

restore_r0_root() {
  cd "$REMOTE_REPO"
  if [[ -s "${RUN_DIR}/r0/round_summary.json" ]]; then
    echo "[skip] imported r0 root exists"
    return
  fi
  .venv/bin/python - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

repo_root = Path("/workspace/tokenbender-prism")
run_dir = repo_root / "runs/issue6_bfcl_tree_search"
snapshot = Path(snapshot_download(
    "TokenBender/circuit-discovery",
    repo_type="dataset",
    allow_patterns=["bfcl/issue5_nearmiss_loop_v1/run/r0/**"],
))
src = snapshot / "bfcl/issue5_nearmiss_loop_v1/run/r0"
dst = run_dir / "r0"
if not src.exists():
    raise FileNotFoundError(src)
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("imported_r0_root", dst)
PY
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

parent_train_jsonl() {
  local parent="$1"
  if [[ "$parent" == "r0" ]]; then
    echo "${REMOTE_REPO}/data/bfcl_strict_10k_mix_len1024/train.jsonl"
  else
    echo "${DATA_DIR}/branches/${parent}/train_mixed.jsonl"
  fi
}

parent_dir() {
  local parent="$1"
  if [[ "$parent" == "r0" ]]; then
    echo "${RUN_DIR}/r0"
  else
    echo "$(branch_dir "$parent")"
  fi
}

failure_dirs_for_parent() {
  local parent="$1"
  local dir
  dir="$(parent_dir "$parent")"
  local dirs=()
  for topk in 80000 120000 160000 200000 240000; do
    if [[ -d "${dir}/failure_buckets_k${topk}" ]]; then
      dirs+=("${dir}/failure_buckets_k${topk}")
    fi
  done
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    echo "no failure buckets found for parent ${parent} in ${dir}" >&2
    exit 20
  fi
  printf '%s\n' "${dirs[@]}"
}

build_branch_data() {
  local spec="$1"
  local branch_id parent_id profile seed
  branch_id="$(json_get "$spec" branch_id)"
  parent_id="$(json_get "$spec" parent_id)"
  profile="$(json_get "$spec" branch_profile)"
  seed="$(json_get "$spec" seed)"
  local bdata brun base_train
  bdata="$(branch_data_dir "$branch_id")"
  brun="$(branch_dir "$branch_id")"
  base_train="$(parent_train_jsonl "$parent_id")"
  mkdir -p "$bdata" "$brun"
  mapfile -t failure_dirs < <(failure_dirs_for_parent "$parent_id")
  cd "$REMOTE_REPO"
  .venv/bin/python code/scripts/build_bfcl_tree_branch_curriculum.py \
    --base-train-jsonl "$base_train" \
    --eval-jsonl data/bfcl_single_call/pairs.jsonl \
    --failure-bucket-dirs "${failure_dirs[@]}" \
    --edge-output "${bdata}/edge.jsonl" \
    --mixed-output "${bdata}/train_mixed.jsonl" \
    --manifest "${bdata}/manifest.json" \
    --branch-id "$branch_id" \
    --parent-id "$parent_id" \
    --branch-profile "$profile" \
    --seed "$seed" \
    --fail-on-leak
  run_leak_audit "${bdata}/train_mixed.jsonl" "${brun}/mixed_overlap_audit.json"
}

train_branch() {
  local spec="$1"
  local gpu="$2"
  local branch_id profile
  branch_id="$(json_get "$spec" branch_id)"
  profile="$(json_get "$spec" branch_profile)"
  local out_dir train_jsonl
  out_dir="$(branch_train_dir "$branch_id")"
  train_jsonl="$(branch_data_dir "$branch_id")/train_mixed.jsonl"
  cd "$REMOTE_REPO"
  if [[ -s "${out_dir}/train_summary.json" && -d "${out_dir}/adapter" && -d "${out_dir}/merged" ]]; then
    echo "[skip] ${branch_id} training exists"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python code/scripts/train_bfcl_unmasked_lora.py \
    --model Qwen/Qwen3-8B \
    --train-jsonl "$train_jsonl" \
    --out-dir "$out_dir" \
    --experiment-id "bfcl_issue6_${branch_id}_${profile}" \
    --github-issue 6 \
    --dtype bfloat16 \
    --device cuda \
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
    --wandb-group issue-6 \
    --wandb-job-type train \
    --wandb-tags "bfcl,issue-6,tree,${branch_id},${profile},qwen3-8b"
}

attribute_branch() {
  local spec="$1"
  local gpu="$2"
  local branch_id
  branch_id="$(json_get "$spec" branch_id)"
  local attr
  attr="$(branch_attr "$branch_id")"
  cd "$REMOTE_REPO"
  if [[ -s "$attr" ]]; then
    echo "[skip] ${branch_id} attribution exists"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python code/scripts/bfcl_direct_qwen3.py relp-attribute \
    --pairs data/bfcl_single_call/pairs.jsonl \
    --output "$attr" \
    --model "$(branch_train_dir "$branch_id")/merged" \
    --dtype bfloat16 \
    --device-map auto \
    --log-every 10 \
    --report-topk 50
}

eval_branch() {
  local spec="$1"
  local gpu="$2"
  local branch_id dir attr
  branch_id="$(json_get "$spec" branch_id)"
  dir="$(branch_dir "$branch_id")"
  attr="$(branch_attr "$branch_id")"
  cd "$REMOTE_REPO"
  for topk in $TOPKS; do
    local out="${dir}/eval_k${topk}_masked.jsonl"
    if [[ ! -s "${out%.jsonl}.summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python code/scripts/bfcl_direct_qwen3.py eval-mask \
        --pairs data/bfcl_single_call/pairs.jsonl \
        --output "$out" \
        --attribution "$attr" \
        --topk "$topk" \
        --model "$(branch_train_dir "$branch_id")/merged" \
        --dtype bfloat16 \
        --device-map auto \
        --batch-size 8 \
        --bfcl-canonicalization-prompt \
        --normalized
    else
      echo "[skip] ${branch_id} eval exists for k=${topk}"
    fi
    local bucket_dir="${dir}/failure_buckets_k${topk}"
    if [[ ! -d "$bucket_dir" ]]; then
      .venv/bin/python code/scripts/build_bfcl_failure_buckets.py \
        --eval-jsonl "$out" \
        --pairs-jsonl data/bfcl_single_call/pairs.jsonl \
        --out-dir "$bucket_dir" \
        --run-name "issue6_${branch_id}_k${topk}"
    fi
  done
}

write_branch_summary() {
  local spec="$1"
  local branch_id
  branch_id="$(json_get "$spec" branch_id)"
  cd "$REMOTE_REPO"
  SPEC="$spec" TOPKS="$TOPKS" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

spec = json.loads(Path(os.environ["SPEC"]).read_text())
branch_id = spec["branch_id"]
run_dir = Path("runs/issue6_bfcl_tree_search") / "branches" / branch_id
data_dir = Path("data/bfcl_issue6_tree_search") / "branches" / branch_id
anchor = 664
thresholds = {"80": 532, "85": 565, "90": 598}
epsilon_floor = {"160000": 478, "200000": 569, "240000": 594}
issue5_best = {"80000": 319, "120000": 497, "160000": 567, "200000": 601, "240000": 619}
summary = {
    **spec,
    "train_jsonl": str(data_dir / "train_mixed.jsonl"),
    "train_rows": sum(1 for line in (data_dir / "train_mixed.jsonl").read_text().splitlines() if line.strip()),
    "manifest": str(data_dir / "manifest.json"),
    "leak_audit": str(run_dir / "mixed_overlap_audit.json"),
    "train_summary": str(run_dir / "unmasked_r32" / "train_summary.json"),
    "attribution": str(run_dir / "relp_full_collimated.npz"),
    "evals": {},
    "thresholds": {},
}
for p in sorted(run_dir.glob("eval_k*_masked.summary.json")):
    topk = p.name.split("_masked.summary.json")[0].removeprefix("eval_k")
    row = json.loads(p.read_text())
    correct = row.get("normalized_exact_correct", row.get("exact_correct"))
    row["behavior_recovery_vs_full_anchor"] = correct / anchor if isinstance(correct, int) else None
    row["full_anchor_normalized_correct"] = anchor
    summary["evals"][topk] = row

for label, threshold in thresholds.items():
    hits = []
    for topk, row in summary["evals"].items():
        correct = row.get("normalized_exact_correct", row.get("exact_correct"))
        if isinstance(correct, int) and correct >= threshold:
            hits.append(int(topk))
    summary["thresholds"][f"smallest_k_ge_{label}_percent"] = min(hits) if hits else None

main_scores = {}
for topk in ("160000", "200000", "240000"):
    row = summary["evals"].get(topk)
    if row:
        main_scores[topk] = row.get("normalized_exact_correct", row.get("exact_correct"))
summary["main_frontier_scores"] = main_scores
summary["main_frontier_best_correct"] = max(main_scores.values()) if main_scores else None
summary["epsilon_safe"] = all(
    isinstance(main_scores.get(topk), int) and main_scores[topk] >= floor
    for topk, floor in epsilon_floor.items()
)
summary["issue5_promotion_hits"] = [
    {"topk": int(topk), "correct": correct, "issue5_best": issue5_best[topk]}
    for topk, correct in main_scores.items()
    if topk in issue5_best and isinstance(correct, int) and correct > issue5_best[topk]
]
summary["compression_promising"] = any(
    summary["thresholds"].get(key) is not None
    for key in ("smallest_k_ge_80_percent", "smallest_k_ge_85_percent", "smallest_k_ge_90_percent")
)
summary["survivor"] = bool(summary["epsilon_safe"] or summary["compression_promising"] or summary["issue5_promotion_hits"])
(run_dir / "branch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_branch() {
  local spec="$1"
  local gpu="$2"
  local branch_id
  branch_id="$(json_get "$spec" branch_id)"
  echo "== branch ${branch_id} gpu=${gpu} =="
  date
  build_branch_data "$spec"
  train_branch "$spec" "$gpu"
  attribute_branch "$spec" "$gpu"
  eval_branch "$spec" "$gpu"
  write_branch_summary "$spec"
  rm -rf "$(branch_train_dir "$branch_id")/merged"
  echo "== branch ${branch_id} done =="
  date
}

write_tree_state() {
  cd "$REMOTE_REPO"
  TOPKS="$TOPKS" MAX_BRANCH_ROUNDS="$MAX_BRANCH_ROUNDS" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path("runs/issue6_bfcl_tree_search")
summaries = []
for p in sorted((run_dir / "branches").glob("b*/branch_summary.json")):
    summaries.append(json.loads(p.read_text()))
topks = [str(x) for x in os.environ["TOPKS"].split()]
thresholds = {"80": 532, "85": 565, "90": 598}
best_by_topk = {}
for topk in topks:
    rows = []
    for row in summaries:
        ev = row.get("evals", {}).get(topk)
        if ev:
            correct = ev.get("normalized_exact_correct", ev.get("exact_correct"))
            if isinstance(correct, int):
                rows.append((correct, row["branch_id"], row["branch_profile"]))
    if rows:
        correct, branch_id, profile = max(rows)
        best_by_topk[topk] = {"correct": correct, "branch_id": branch_id, "branch_profile": profile}
smallest = {}
for label, threshold in thresholds.items():
    hits = []
    for topk, item in best_by_topk.items():
        if item["correct"] >= threshold:
            hits.append(int(topk))
    smallest[f"smallest_k_ge_{label}_percent"] = min(hits) if hits else None
survivors = [row for row in summaries if row.get("survivor")]
state = {
    "experiment_id": "bfcl_issue6_tree_search_v1",
    "branches_completed": len(summaries),
    "max_branch_rounds": int(os.environ["MAX_BRANCH_ROUNDS"]),
    "best_by_topk": best_by_topk,
    "smallest_recovery_k": smallest,
    "survivor_count": len(survivors),
    "survivors": [
        {
            "branch_id": row["branch_id"],
            "parent_id": row["parent_id"],
            "depth": row["depth"],
            "branch_profile": row["branch_profile"],
            "main_frontier_best_correct": row.get("main_frontier_best_correct"),
            "thresholds": row.get("thresholds", {}),
            "epsilon_safe": row.get("epsilon_safe"),
        }
        for row in survivors
    ],
    "should_continue": len(summaries) < int(os.environ["MAX_BRANCH_ROUNDS"]),
    "stop_reason": "branch_budget_exhausted" if len(summaries) >= int(os.environ["MAX_BRANCH_ROUNDS"]) else "continue",
}
(run_dir / "tree_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
print(json.dumps(state, indent=2, sort_keys=True))
PY
}

plan_next_wave() {
  local wave_dir="$1"
  mkdir -p "$wave_dir"
  cd "$REMOTE_REPO"
  WAVE_DIR="$wave_dir" TOPKS="$TOPKS" MAX_BRANCH_ROUNDS="$MAX_BRANCH_ROUNDS" PARALLEL_BRANCHES="$PARALLEL_BRANCHES" BEAM_WIDTH="$BEAM_WIDTH" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path("runs/issue6_bfcl_tree_search")
wave_dir = Path(os.environ["WAVE_DIR"])
max_rounds = int(os.environ["MAX_BRANCH_ROUNDS"])
parallel = int(os.environ["PARALLEL_BRANCHES"])
beam_width = int(os.environ["BEAM_WIDTH"])
profiles_first = [
    "conservative_nearmiss",
    "bucket_balanced",
    "teacher_ranked",
    "schema_stratified",
    "compression_biased",
    "hardcase_replay",
]
profiles_next = [
    "epsilon_repair",
    "pareto_trim",
    "compression_biased",
    "teacher_ranked",
    "bucket_balanced",
    "schema_stratified",
    "conservative_nearmiss",
    "hardcase_replay",
]
summaries = []
for p in sorted((run_dir / "branches").glob("b*/branch_summary.json")):
    summaries.append(json.loads(p.read_text()))
completed = {row["branch_id"] for row in summaries}
remaining = max_rounds - len(completed)
if remaining <= 0:
    print("planned 0")
    raise SystemExit(0)

def parent_score(row):
    thresholds = row.get("thresholds", {})
    score = 0
    for label, weight in (("90", 1_000_000), ("85", 100_000), ("80", 10_000)):
        key = f"smallest_k_ge_{label}_percent"
        value = thresholds.get(key)
        if value is not None:
            score += weight - int(value)
    score += int(row.get("main_frontier_best_correct") or 0)
    if row.get("epsilon_safe"):
        score += 500
    if row.get("issue5_promotion_hits"):
        score += 250
    return score

specs = []
if not summaries:
    for profile in profiles_first[: min(parallel, remaining)]:
        idx = len(specs) + 1
        specs.append({
            "branch_id": f"b{idx:03d}",
            "parent_id": "r0",
            "depth": 1,
            "branch_profile": profile,
            "seed": 606 + idx * 17,
        })
else:
    parents = [row for row in summaries if row.get("survivor")]
    if not parents:
        parents = summaries
    parents = sorted(parents, key=parent_score, reverse=True)[:beam_width]
    next_idx = len(summaries) + 1
    used_pairs = {
        (row.get("parent_id"), row.get("branch_profile"), row.get("depth"))
        for row in summaries
    }
    for parent in parents:
        for profile in profiles_next:
            if len(specs) >= min(parallel, remaining):
                break
            depth = int(parent.get("depth", 1)) + 1
            pair = (parent["branch_id"], profile, depth)
            if pair in used_pairs:
                continue
            specs.append({
                "branch_id": f"b{next_idx:03d}",
                "parent_id": parent["branch_id"],
                "depth": depth,
                "branch_profile": profile,
                "seed": 606 + next_idx * 17,
            })
            next_idx += 1
        if len(specs) >= min(parallel, remaining):
            break

for spec in specs:
    (wave_dir / f"{spec['branch_id']}.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
print("planned", len(specs))
PY
}

run_wave() {
  local wave="$1"
  local wave_dir="${RUN_DIR}/plans/wave_${wave}"
  rm -rf "$wave_dir"
  mkdir -p "$wave_dir"
  plan_next_wave "$wave_dir"
  mapfile -t specs < <(find "$wave_dir" -name 'b*.json' -maxdepth 1 -type f | sort)
  if [[ "${#specs[@]}" -eq 0 ]]; then
    echo "No specs planned for wave ${wave}"
    return 1
  fi
  local pids=()
  local i=0
  for spec in "${specs[@]}"; do
    local branch_id gpu log
    branch_id="$(json_get "$spec" branch_id)"
    gpu="$((i % PARALLEL_BRANCHES))"
    log="${LOG_DIR}/${branch_id}.log"
    echo "[wave ${wave}] launching ${branch_id} on gpu ${gpu}"
    (run_branch "$spec" "$gpu") >"$log" 2>&1 &
    pids+=("$!")
    i="$((i + 1))"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one branch failed in wave ${wave}; preserving state and stopping." >&2
    exit 30
  fi
  write_tree_state
}

write_final_summary() {
  cd "$REMOTE_REPO"
  .venv/bin/python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/issue6_bfcl_tree_search")
summary = {
    "experiment_id": "bfcl_issue6_tree_search_v1",
    "config": "code/configs/bfcl_issue6_tree_search.json",
    "baselines": {
        "full_unmasked_qwen3_8b_normalized": "664/1007",
        "issue2_r0_k160": "488/1007",
        "issue2_r0_k200": "579/1007",
        "issue2_r0_k240": "604/1007",
        "issue5_r3_k160": "567/1007",
        "issue5_r3_k200": "601/1007",
        "issue5_r3_k240": "619/1007",
    },
    "r0": None,
    "branches": {},
    "tree_state": None,
}
r0 = run_dir / "r0" / "round_summary.json"
if r0.exists():
    summary["r0"] = json.loads(r0.read_text())
for p in sorted((run_dir / "branches").glob("b*/branch_summary.json")):
    row = json.loads(p.read_text())
    summary["branches"][row["branch_id"]] = row
state = run_dir / "tree_state.json"
if state.exists():
    summary["tree_state"] = json.loads(state.read_text())
(run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["tree_state"], indent=2, sort_keys=True))
PY
}

stage_artifacts() {
  cd "$REMOTE_REPO"
  rm -rf "$HF_STAGE"
  mkdir -p "$HF_STAGE/run" "$HF_STAGE/data" "$HF_STAGE/logs" "$HF_STAGE/configs"
  cp -f "$CONFIG" "$HF_STAGE/configs/bfcl_issue6_tree_search.json"
  cp -f docs/experiments/bfcl_issue6_tree_search.md "$HF_STAGE/configs/bfcl_issue6_tree_search.md"
  cp -f "$FULL_LOG" "$HF_STAGE/logs/issue6_full_pipeline.log" || true
  cp -f "${LOG_DIR}/heartbeat.jsonl" "$HF_STAGE/logs/heartbeat.jsonl" || true
  rsync -a \
    --exclude "*/merged" \
    --exclude "merged" \
    --exclude "checkpoints" \
    --exclude "checkpoints/*/merged" \
    "$RUN_DIR/" "$HF_STAGE/run/"
  rsync -a "$DATA_DIR/" "$HF_STAGE/data/"
  tar -C "$HF_STAGE" -czf "${REMOTE_RUNS}/issue6_tree_search_artifacts.tgz" .
  sha256sum "${REMOTE_RUNS}/issue6_tree_search_artifacts.tgz" \
    | tee "${REMOTE_RUNS}/issue6_tree_search_artifacts.tgz.sha256"
}

upload_hf() {
  cd "$REMOTE_REPO"
  .venv/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="TokenBender/circuit-discovery",
    repo_type="dataset",
    folder_path="/workspace/tokenbender-prism-runs/hf_stage/issue6_tree_search_v1",
    path_in_repo="bfcl/issue6_tree_search_v1",
    commit_message="Add BFCL issue 6 tree-search artifacts",
)
print("uploaded_to_hf", "TokenBender/circuit-discovery", "bfcl/issue6_tree_search_v1")
PY
}

run_pipeline() {
  cd "$REMOTE_REPO"
  echo "== issue6 BFCL tree-search pipeline =="
  date
  echo "max_branch_rounds=${MAX_BRANCH_ROUNDS}"
  echo "parallel_branches=${PARALLEL_BRANCHES}"
  echo "beam_width=${BEAM_WIDTH}"
  echo "topks=${TOPKS}"
  setup_env
  check_auth_and_env
  restore_data
  restore_r0_root
  write_tree_state

  wave=1
  while true; do
    completed="$(
      .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("runs/issue6_bfcl_tree_search/tree_state.json")
print((json.loads(p.read_text()) if p.exists() else {}).get("branches_completed", 0))
PY
    )"
    if [[ "$completed" -ge "$MAX_BRANCH_ROUNDS" ]]; then
      break
    fi
    run_wave "$wave" || break
    wave="$((wave + 1))"
  done

  write_tree_state
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
  branch)
    run_branch "$2" "$3"
    ;;
  run)
    exec > >(tee -a "$FULL_LOG") 2>&1
    run_pipeline
    ;;
  *)
    echo "usage: $0 [check|run|branch SPEC GPU]" >&2
    exit 2
    ;;
esac
REMOTE

cat >"$tmpdir/issue6_heartbeat.sh" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail
REMOTE_REPO="${REMOTE_REPO:-/workspace/tokenbender-prism}"
REMOTE_RUNS="${REMOTE_RUNS:-/workspace/tokenbender-prism-runs}"
LOG_DIR="${REMOTE_RUNS}/issue6_logs"
mkdir -p "$LOG_DIR"
while true; do
  cd "$REMOTE_REPO"
  python3 - <<'PY' >>"/workspace/tokenbender-prism-runs/issue6_logs/heartbeat.jsonl" 2>/dev/null || true
import json
import subprocess
import time
from pathlib import Path

run_dir = Path("runs/issue6_bfcl_tree_search")
state_path = run_dir / "tree_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
try:
    tmux = subprocess.check_output(["tmux", "ls"], text=True)
except Exception as exc:
    tmux = f"tmux_error:{type(exc).__name__}"
try:
    gpu = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
except Exception as exc:
    gpu = f"nvidia_smi_error:{type(exc).__name__}"
try:
    disk = subprocess.check_output(["df", "-h", "/workspace"], text=True)
except Exception as exc:
    disk = f"df_error:{type(exc).__name__}"
row = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "branches_completed": state.get("branches_completed", 0),
    "smallest_recovery_k": state.get("smallest_recovery_k", {}),
    "survivor_count": state.get("survivor_count"),
    "tmux": tmux.strip().splitlines(),
    "gpu": gpu.strip().splitlines(),
    "disk": disk.strip().splitlines(),
}
print(json.dumps(row, sort_keys=True))
PY
  sleep 300
done
REMOTE

echo "== Preparing clean tracked snapshot =="
mkdir -p "$tmpdir/repo"
git archive --format=tar HEAD | tar -C "$tmpdir/repo" -xf -

echo "== Preparing remote directories =="
lium exec "$TARGET" "mkdir -p '$REMOTE_REPO' '$REMOTE_RUNS' /root"

echo "== Syncing tracked repository snapshot =="
lium rsync "$TARGET" "$tmpdir/repo/" "$REMOTE_REPO/"

echo "== Uploading credentials and launchers =="
lium scp "$TARGET" "$tmpdir/issue6_env" /root/issue6_env
lium scp "$TARGET" "$tmpdir/issue6_launch_full.sh" "$REMOTE_LAUNCH"
lium scp "$TARGET" "$tmpdir/issue6_heartbeat.sh" "$REMOTE_HEARTBEAT"
lium exec "$TARGET" "chmod 600 /root/issue6_env && chmod +x '$REMOTE_LAUNCH' '$REMOTE_HEARTBEAT'"

echo "== Verifying pod environment =="
lium exec "$TARGET" "REMOTE_REPO='$REMOTE_REPO' REMOTE_RUNS='$REMOTE_RUNS' MAX_BRANCH_ROUNDS='$MAX_BRANCH_ROUNDS' PARALLEL_BRANCHES='$PARALLEL_BRANCHES' BEAM_WIDTH='$BEAM_WIDTH' TOPKS='$TOPKS' TORCH_INDEX_URL='$TORCH_INDEX_URL' TORCH_PACKAGE='$TORCH_PACKAGE' bash '$REMOTE_LAUNCH' check"

echo "== Launching remote tmux session: $SESSION =="
lium exec "$TARGET" "bash -lc 'tmux has-session -t \"$SESSION\" 2>/dev/null && { echo \"tmux session already exists: $SESSION\"; tmux ls; exit 0; }; tmux new-session -d -s \"$SESSION\" \"REMOTE_REPO=$REMOTE_REPO REMOTE_RUNS=$REMOTE_RUNS MAX_BRANCH_ROUNDS=$MAX_BRANCH_ROUNDS PARALLEL_BRANCHES=$PARALLEL_BRANCHES BEAM_WIDTH=$BEAM_WIDTH TOPKS=\\\"$TOPKS\\\" TORCH_INDEX_URL=$TORCH_INDEX_URL TORCH_PACKAGE=$TORCH_PACKAGE bash $REMOTE_LAUNCH run\"; tmux ls'"

echo "== Launching heartbeat session: $HEARTBEAT_SESSION =="
lium exec "$TARGET" "bash -lc 'tmux has-session -t \"$HEARTBEAT_SESSION\" 2>/dev/null && { echo \"heartbeat tmux session already exists: $HEARTBEAT_SESSION\"; tmux ls; exit 0; }; tmux new-session -d -s \"$HEARTBEAT_SESSION\" \"REMOTE_REPO=$REMOTE_REPO REMOTE_RUNS=$REMOTE_RUNS bash $REMOTE_HEARTBEAT\"; tmux ls'"

cat <<EOF

Launched issue #6 BFCL tree search.

Monitor:
  lium exec $TARGET "tail -f ${REMOTE_RUNS}/issue6_logs/issue6_full_pipeline.log"

Heartbeat:
  lium exec $TARGET "tail -n 5 ${REMOTE_RUNS}/issue6_logs/heartbeat.jsonl"

Check session:
  lium exec $TARGET "tmux ls"
EOF
