#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/prism-capability-extraction}"
INPUT_ROOT="${INPUT_ROOT:-/root/issue24_bfcl_inputs}"
RUN_ROOT="${RUN_ROOT:-/root/issue24_bfcl_iterative_tree}"
MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
DATA_REVISION="${DATA_REVISION:-1aacefc5825a48400497f4d638090da967fcba74}"
TRAIN_DATA_REVISION="${TRAIN_DATA_REVISION:-303db0bddcfb04bebaf07ab4a4dc4c089240c545}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
ARTIFACT_REPO="${ARTIFACT_REPO:-TokenBender/circuit-discovery}"
ARTIFACT_REMOTE_PATH="${ARTIFACT_REMOTE_PATH:-bfcl/issue24_iterative_sft_tree_v1/issue24_bfcl_iterative_sft_tree.tar.gz}"
DEVICE_COUNT="${DEVICE_COUNT:-8}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG="${REPO_ROOT}/code/configs/issue24_bfcl_iterative_tree.json"
RUNNER="${REPO_ROOT}/code/scripts/run_issue24_bfcl_iterative_tree.py"
STARTED_EPOCH="$(date +%s)"

mkdir -p "${RUN_ROOT}/setup" "${INPUT_ROOT}"
printf '%s\n' "${STARTED_EPOCH}" > "${RUN_ROOT}/setup/started_epoch.txt"

cd "${REPO_ROOT}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: ${ACTUAL_COMMIT} != ${EXPECTED_COMMIT}" >&2
  exit 10
fi
printf '%s\n' "${ACTUAL_COMMIT}" > "${RUN_ROOT}/setup/git_commit.txt"
git status --short > "${RUN_ROOT}/setup/git_status.txt"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader > "${RUN_ROOT}/setup/gpu_identity.csv"
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -ne "${DEVICE_COUNT}" ]]; then
  echo "GPU count mismatch: ${GPU_COUNT} != ${DEVICE_COUNT}" >&2
  exit 11
fi
if [[ "$(cut -d, -f2 "${RUN_ROOT}/setup/gpu_identity.csv" | sort -u | wc -l | tr -d ' ')" -ne 1 ]]; then
  echo "heterogeneous GPU node is outside contract" >&2
  exit 12
fi

if [[ ! -x "${PYTHON}" ]]; then
  python3 -m venv --system-site-packages "${REPO_ROOT}/.venv"
fi
"${PYTHON}" -m pip install -U pip setuptools wheel
"${PYTHON}" -m pip install \
  'transformers==4.57.6' \
  'peft>=0.19.1' \
  'accelerate>=1.0' \
  'datasets>=2.18' \
  'huggingface-hub>=0.34.4' \
  safetensors numpy
"${PYTHON}" -m pip freeze > "${RUN_ROOT}/setup/pip_freeze.txt"

"${PYTHON}" - "${INPUT_ROOT}" "${MODEL_REVISION}" "${DATA_REVISION}" "${TRAIN_DATA_REVISION}" "${TOKENIZER_REVISION}" <<'PY'
import json
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

input_root = Path(sys.argv[1])
model_revision, data_revision, train_data_revision, tokenizer_revision = sys.argv[2:6]
input_root.mkdir(parents=True, exist_ok=True)

adapter_snapshot = Path(snapshot_download(
    "TokenBender/circuit-discovery",
    repo_type="dataset",
    revision=data_revision,
    allow_patterns=["bfcl/issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter/**"],
))
adapter_source = adapter_snapshot / "bfcl/issue6_tree_search_v1/run/branches/b007/unmasked_r32/adapter"
adapter_dest = input_root / "root_adapter"
if adapter_dest.exists():
    shutil.rmtree(adapter_dest)
shutil.copytree(adapter_source, adapter_dest)

archive = Path(hf_hub_download(
    "TokenBender/circuit-discovery",
    "bfcl/issue12_recursive_coactivation_mace_v1_bundle/issue12_recursive_coactivation_mace_v1_artifacts.tgz",
    repo_type="dataset",
    revision=data_revision,
))
mask_dest = input_root / "root_mask.npz"
pairs_dest = input_root / "pairs.jsonl"
targets = {
    "candidate_masks/category_repair_java_r500_protect_tail_b140875_p10000.npz": mask_dest,
    "data/bfcl_single_call/pairs.jsonl": pairs_dest,
}
with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
    for suffix, destination in targets.items():
        matches = [member for member in members if member.name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one archive member ending {suffix}, found {[member.name for member in matches]}")
        source = handle.extractfile(matches[0])
        if source is None:
            raise RuntimeError(f"archive member is not a file: {matches[0].name}")
        with destination.open("wb") as output:
            shutil.copyfileobj(source, output)

train_source = Path(hf_hub_download(
    "Occupying-Mars/issue49-bfcl-repro-artifacts",
    "data/bfcl_strict_10k_mix_len1024/train.jsonl",
    repo_type="dataset",
    revision=train_data_revision,
))
train_dest = input_root / "train.jsonl"
shutil.copy2(train_source, train_dest)

receipt = {
    "model": "Qwen/Qwen3-8B",
    "model_revision": model_revision,
    "tokenizer_revision": tokenizer_revision,
    "artifact_data_revision": data_revision,
    "training_data_revision": train_data_revision,
    "root_adapter": str(adapter_dest),
    "root_mask": str(mask_dest),
    "pairs": str(pairs_dest),
    "base_train_jsonl": str(train_dest),
}
(input_root / "staging_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY


if [[ ! -s "${INPUT_ROOT}/root_model/config.json" ]]; then
  "${PYTHON}" - "${INPUT_ROOT}" "${MODEL_REVISION}" "${TOKENIZER_REVISION}" <<'PY'
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(sys.argv[1])
model_revision = sys.argv[2]
tokenizer_revision = sys.argv[3]
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    revision=model_revision,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(base, root / "root_adapter")
merged = model.merge_and_unload()
merged.save_pretrained(root / "root_model", safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=tokenizer_revision)
tokenizer.save_pretrained(root / "root_model")
PY
fi
"${PYTHON}" - "${INPUT_ROOT}" "${RUN_ROOT}/setup" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

input_root = Path(sys.argv[1])
setup_root = Path(sys.argv[2])
setup_root.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

paths = {
    "root_mask": input_root / "root_mask.npz",
    "root_adapter": input_root / "root_adapter" / "adapter_model.safetensors",
    "pairs": input_root / "pairs.jsonl",
    "training_data": input_root / "train.jsonl",
}
receipt = {
    name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    for name, path in paths.items()
}
(setup_root / "input_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
shutil.copy2(input_root / "staging_receipt.json", setup_root / "staging_receipt.json")
if (input_root / "pairs_manifest.json").exists():
    shutil.copy2(input_root / "pairs_manifest.json", setup_root / "pairs_manifest.json")
PY


"${PYTHON}" "${RUNNER}" \
  --config "${CONFIG}" \
  --pairs "${INPUT_ROOT}/pairs.jsonl" \
  --base-train-jsonl "${INPUT_ROOT}/train.jsonl" \
  --root-model "${INPUT_ROOT}/root_model" \
  --root-mask "${INPUT_ROOT}/root_mask.npz" \
  --output-dir "${RUN_ROOT}/run" \
  --device-count "${DEVICE_COUNT}" \
  2>&1 | tee "${RUN_ROOT}/run.log"

FINISHED_EPOCH="$(date +%s)"
printf '%s\n' "$((FINISHED_EPOCH - STARTED_EPOCH))" > "${RUN_ROOT}/setup/elapsed_wall_seconds.txt"
printf '%s\n' 0 > "${RUN_ROOT}/setup/run_exit_code.txt"

ARCHIVE="${RUN_ROOT}.tar.gz"
tar \
  --exclude='*/merged' \
  --exclude='__pycache__' \
  --exclude='.cache' \
  -czf "${ARCHIVE}" \
  -C "$(dirname "${RUN_ROOT}")" "$(basename "${RUN_ROOT}")"
sha256sum "${ARCHIVE}" > "${RUN_ROOT}/setup/archive_sha256.txt"
ARCHIVE_SHA256="$(cut -d' ' -f1 "${RUN_ROOT}/setup/archive_sha256.txt")"
ARCHIVE_SIZE="$(stat -c '%s' "${ARCHIVE}")"

"${PYTHON}" - "${ARCHIVE}" "${ARTIFACT_REPO}" "${ARTIFACT_REMOTE_PATH}" "${ARCHIVE_SHA256}" "${ARCHIVE_SIZE}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

archive, repository, remote_path, sha256, size, commit = sys.argv[1:7]
api = HfApi()
result = api.upload_file(
    path_or_fileobj=archive,
    path_in_repo=remote_path,
    repo_id=repository,
    repo_type="dataset",
    commit_message="Add issue 24 BFCL iterative SFT tree artifact",
)
receipt = {
    "schema_version": "prism_bfcl_issue24_artifact_receipt_v1",
    "issue": 24,
    "source_commit": commit,
    "artifact_repository": repository,
    "artifact_path": remote_path,
    "artifact_sha256": sha256,
    "artifact_size_bytes": int(size),
    "upload_result": str(result),
}
receipt_path = Path(archive).with_suffix(".receipt.json")
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
api.upload_file(
    path_or_fileobj=str(receipt_path),
    path_in_repo=str(Path(remote_path).with_name("artifact_receipt.json")),
    repo_id=repository,
    repo_type="dataset",
    commit_message="Add issue 24 BFCL artifact receipt",
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
