#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/prism-capability-extraction}"
INPUT_ROOT="${INPUT_ROOT:-/root/issue23_best_tree_inputs}"
RUN_ROOT="${RUN_ROOT:-/root/issue23_best_checkpoint_tree}"
MODEL_REVISION="${MODEL_REVISION:-f75ca2f123ce6aaca0e8096918df1ddb34b5d546}"
DATA_REVISION="${DATA_REVISION:-4b9fb53fef92550042d8576fe011e99270fdca8b}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2}"
ROOT_ARTIFACT_REVISION="${ROOT_ARTIFACT_REVISION:-7a44ce26f513e68b3cb92eea0b5f22cca100dd67}"
MODEL_REPO="${MODEL_REPO:-TokenBender/circuit-discovery}"
DATA_REPO="${DATA_REPO:-TokenBender/circuit-discovery}"
TOKENIZER_REPO="${TOKENIZER_REPO:-Qwen/Qwen2.5-Math-1.5B}"
ROOT_ARTIFACT_REPO="${ROOT_ARTIFACT_REPO:-TokenBender/circuit-discovery}"
BASELINE_MASK_RELATIVE="circuit-shotting/artifacts/pod_logs/steed-medium/full/qwen25_math_1p5b_2digit_max_recovery_v1/position_adacs_lora_r32_beta005/composed_union.full.npz"
ROOT_ARTIFACT_RELATIVE="circuit-shotting/artifacts/issue23/eight_round_repair_v1/issue23_eight_round_run_corrected.tar.gz"
ROOT_ARTIFACT_SHA256="5cdc4731c793e0e45b9c5fe0939fe0ef037ea57347f68d9e18f548f6d1f4ce1a"

mkdir -p "${INPUT_ROOT}" "${RUN_ROOT}/setup"
STARTED_EPOCH="$(date +%s)"
printf '%s\n' "${STARTED_EPOCH}" > "${RUN_ROOT}/setup/started_epoch.txt"
nvidia-smi -L | tee "${RUN_ROOT}/setup/nvidia_smi_L.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee "${RUN_ROOT}/setup/gpu_identity.csv"
git -C "${REPO_ROOT}" rev-parse HEAD | tee "${RUN_ROOT}/setup/git_commit.txt"

python -m pip install --break-system-packages --quiet --upgrade \
  'transformers==4.57.6' \
  'peft==0.19.1' \
  'huggingface-hub>=0.34,<1.0' \
  'safetensors>=0.4'

export INPUT_ROOT MODEL_REVISION DATA_REVISION TOKENIZER_REVISION ROOT_ARTIFACT_REVISION
export MODEL_REPO DATA_REPO TOKENIZER_REPO ROOT_ARTIFACT_REPO
export BASELINE_MASK_RELATIVE ROOT_ARTIFACT_RELATIVE ROOT_ARTIFACT_SHA256
python - <<'PY'
import hashlib
import json
import os
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(os.environ["INPUT_ROOT"])
model_root = Path(snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    repo_type="model",
    revision=os.environ["MODEL_REVISION"],
    allow_patterns=["checkpoints/checkpoint-b/**"],
    local_dir=root / "model_repo",
))
data_root = Path(snapshot_download(
    repo_id=os.environ["DATA_REPO"],
    repo_type="dataset",
    revision=os.environ["DATA_REVISION"],
    allow_patterns=[os.environ["BASELINE_MASK_RELATIVE"]],
    local_dir=root / "data_repo",
))
tokenizer_root = Path(snapshot_download(
    repo_id=os.environ["TOKENIZER_REPO"],
    repo_type="model",
    revision=os.environ["TOKENIZER_REVISION"],
    allow_patterns=[
        "tokenizer*",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "generation_config.json",
    ],
    local_dir=root / "tokenizer_repo",
))
artifact = Path(hf_hub_download(
    repo_id=os.environ["ROOT_ARTIFACT_REPO"],
    repo_type="dataset",
    revision=os.environ["ROOT_ARTIFACT_REVISION"],
    filename=os.environ["ROOT_ARTIFACT_RELATIVE"],
))
hash_value = hashlib.sha256(artifact.read_bytes()).hexdigest()
if hash_value != os.environ["ROOT_ARTIFACT_SHA256"]:
    raise RuntimeError(f"root artifact SHA-256 mismatch: {hash_value}")
member_root = "issue23_eight_round_run_corrected/run/round_01"
wanted = [
    f"{member_root}/selected_mask.npz",
    f"{member_root}/adapter/adapter_config.json",
    f"{member_root}/adapter/adapter_model.safetensors",
    f"{member_root}/adapter/README.md",
]
with tarfile.open(artifact, "r:gz") as archive:
    for name in wanted:
        member = archive.getmember(name)
        archive.extract(member, path=root / "accepted_root", filter="data")

model = model_root / "checkpoints/checkpoint-b"
baseline_mask = data_root / os.environ["BASELINE_MASK_RELATIVE"]
accepted_root = root / "accepted_root" / member_root
root_mask = accepted_root / "selected_mask.npz"
root_adapter = accepted_root / "adapter"
required = [
    model / "config.json",
    model / "model.safetensors",
    baseline_mask,
    root_mask,
    root_adapter / "adapter_config.json",
    root_adapter / "adapter_model.safetensors",
    tokenizer_root / "tokenizer.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"missing staged inputs: {missing}")
receipt = {
    "model": str(model),
    "model_revision": os.environ["MODEL_REVISION"],
    "data_revision": os.environ["DATA_REVISION"],
    "tokenizer": str(tokenizer_root),
    "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
    "baseline_mask": str(baseline_mask),
    "root_artifact": str(artifact),
    "root_artifact_revision": os.environ["ROOT_ARTIFACT_REVISION"],
    "root_artifact_sha256": hash_value,
    "root_adapter": str(root_adapter),
    "root_mask": str(root_mask),
}
(root / "staging_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(receipt, sort_keys=True))
PY

python -m pip freeze > "${RUN_ROOT}/setup/pip_freeze.txt"
cp "${INPUT_ROOT}/staging_receipt.json" "${RUN_ROOT}/setup/staging_receipt.json"

MODEL="${INPUT_ROOT}/model_repo/checkpoints/checkpoint-b"
TOKENIZER="${INPUT_ROOT}/tokenizer_repo"
ACCEPTED_ROOT="${INPUT_ROOT}/accepted_root/issue23_eight_round_run_corrected/run/round_01"
ROOT_ADAPTER="${ACCEPTED_ROOT}/adapter"
ROOT_MASK="${ACCEPTED_ROOT}/selected_mask.npz"
BASELINE_MASK="${INPUT_ROOT}/data_repo/${BASELINE_MASK_RELATIVE}"
OUTPUT="${RUN_ROOT}/run"
CONFIG="${REPO_ROOT}/code/configs/issue23_best_checkpoint_tree.json"

set +e
timeout --signal=TERM --kill-after=30s 2250s \
  python "${REPO_ROOT}/code/scripts/run_issue23_best_checkpoint_tree.py" \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    --root-adapter "${ROOT_ADAPTER}" \
    --root-mask "${ROOT_MASK}" \
    --baseline-mask "${BASELINE_MASK}" \
    --baseline-mask-key mlp_rel_0.001 \
    --output-dir "${OUTPUT}" \
    --device cuda \
    --dtype bfloat16 \
    --eval-batch-size 128 \
    --seed 42 \
    --deadline-seconds 2150 \
  2>&1 | tee "${RUN_ROOT}/run.log"
STATUS="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${STATUS}" > "${RUN_ROOT}/setup/run_exit_code.txt"
printf '%s\n' "$(( $(date +%s) - STARTED_EPOCH ))" > "${RUN_ROOT}/setup/elapsed_wall_seconds.txt"
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RUN_ROOT}.tar.gz" "$(basename "${RUN_ROOT}")"
printf 'run_exit_code=%s artifact=%s\n' "${STATUS}" "${RUN_ROOT}.tar.gz"
exit "${STATUS}"
