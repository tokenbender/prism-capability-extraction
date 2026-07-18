#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/prism-capability-extraction}"
INPUT_ROOT="${INPUT_ROOT:-/root/issue23_round20_inputs}"
RUN_ROOT="${RUN_ROOT:-/root/issue23_arithmetic_round20}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
MODEL_REVISION="${MODEL_REVISION:-f75ca2f123ce6aaca0e8096918df1ddb34b5d546}"
DATA_REVISION="${DATA_REVISION:-4b9fb53fef92550042d8576fe011e99270fdca8b}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2}"
R1_ARTIFACT_REVISION="${R1_ARTIFACT_REVISION:-7a44ce26f513e68b3cb92eea0b5f22cca100dd67}"
MODEL_REPO="TokenBender/circuit-discovery"
DATA_REPO="TokenBender/circuit-discovery"
TOKENIZER_REPO="Qwen/Qwen2.5-Math-1.5B"
BASELINE_MASK_RELATIVE="circuit-shotting/artifacts/pod_logs/steed-medium/full/qwen25_math_1p5b_2digit_max_recovery_v1/position_adacs_lora_r32_beta005/composed_union.full.npz"
R1_ARTIFACT_RELATIVE="circuit-shotting/artifacts/issue23/eight_round_repair_v1/issue23_eight_round_run_corrected.tar.gz"
R1_ARTIFACT_SHA256="5cdc4731c793e0e45b9c5fe0939fe0ef037ea57347f68d9e18f548f6d1f4ce1a"
R8_ARTIFACT_RELATIVE="circuit-shotting/artifacts/issue23/best_checkpoint_tree_full_r8_v2/issue23_best_checkpoint_tree_full_r8.tar.gz"
R8_ARTIFACT_SHA256="cf7ae4c98eb053bd3edc0d12b5a97611b59998296f72ae2d8c437db672644dfe"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG="${REPO_ROOT}/code/configs/issue23_round20_continuation.json"
RUNNER="${REPO_ROOT}/code/scripts/run_issue23_round20_continuation.py"
STARTED_EPOCH="$(date +%s)"

mkdir -p "${INPUT_ROOT}" "${RUN_ROOT}/setup"
printf '%s\n' "${STARTED_EPOCH}" > "${RUN_ROOT}/setup/started_epoch.txt"
cd "${REPO_ROOT}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit mismatch: ${ACTUAL_COMMIT} != ${EXPECTED_COMMIT}" >&2
  exit 10
fi
printf '%s\n' "${ACTUAL_COMMIT}" > "${RUN_ROOT}/setup/git_commit.txt"

GPU_COUNT="$(nvidia-smi -L | tee "${RUN_ROOT}/setup/nvidia_smi_L.txt" | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "expected 8 GPUs, found ${GPU_COUNT}" >&2
  exit 11
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee "${RUN_ROOT}/setup/gpu_identity.csv"

if [[ ! -x "${PYTHON}" ]]; then
  python3 -m venv --system-site-packages "${REPO_ROOT}/.venv"
fi
"${PYTHON}" -m pip install --quiet --upgrade \
  'transformers==4.57.6' \
  'peft==0.19.1' \
  'huggingface-hub>=0.34,<1.0' \
  'safetensors>=0.4'

export INPUT_ROOT MODEL_REVISION DATA_REVISION TOKENIZER_REVISION R1_ARTIFACT_REVISION
export MODEL_REPO DATA_REPO TOKENIZER_REPO BASELINE_MASK_RELATIVE
export R1_ARTIFACT_RELATIVE R1_ARTIFACT_SHA256 R8_ARTIFACT_RELATIVE R8_ARTIFACT_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
import shutil
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
        "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json",
        "special_tokens_map.json", "generation_config.json",
    ],
    local_dir=root / "tokenizer_repo",
))
r1_artifact = Path(hf_hub_download(
    repo_id=os.environ["DATA_REPO"],
    repo_type="dataset",
    revision=os.environ["R1_ARTIFACT_REVISION"],
    filename=os.environ["R1_ARTIFACT_RELATIVE"],
))
r8_artifact = Path(hf_hub_download(
    repo_id=os.environ["DATA_REPO"],
    repo_type="dataset",
    revision="main",
    filename=os.environ["R8_ARTIFACT_RELATIVE"],
))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

r1_hash = sha256(r1_artifact)
r8_hash = sha256(r8_artifact)
if r1_hash != os.environ["R1_ARTIFACT_SHA256"]:
    raise RuntimeError(f"R1 artifact SHA-256 mismatch: {r1_hash}")
if r8_hash != os.environ["R8_ARTIFACT_SHA256"]:
    raise RuntimeError(f"R8 artifact SHA-256 mismatch: {r8_hash}")

r1_member_root = "issue23_eight_round_run_corrected/run/round_01"
r1_wanted = [
    f"{r1_member_root}/adapter/adapter_config.json",
    f"{r1_member_root}/adapter/adapter_model.safetensors",
    f"{r1_member_root}/adapter/README.md",
]
r1_dest = root / "r1_root"
with tarfile.open(r1_artifact, "r:gz") as archive:
    for name in r1_wanted:
        archive.extract(archive.getmember(name), path=r1_dest, filter="data")
r1_adapter = r1_dest / r1_member_root / "adapter"

r8_dest = root / "r8_root"
r8_prefix = "issue23_best_checkpoint_tree_full_r8/run"
with tarfile.open(r8_artifact, "r:gz") as archive:
    result_member = archive.getmember(f"{r8_prefix}/result.json")
    result = json.load(archive.extractfile(result_member))
    accepted = [row for row in result["progress"] if row.get("status") == "accepted"]
    adapter_paths = []
    adapter_hashes = []
    for row in accepted:
        remote = Path(row["selected_adapter"])
        relative = Path(*remote.parts[remote.parts.index("run") + 1:])
        member_adapter = Path(r8_prefix) / relative
        for filename in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
            name = str(member_adapter / filename)
            archive.extract(archive.getmember(name), path=r8_dest, filter="data")
        local_adapter = r8_dest / member_adapter
        observed = sha256(local_adapter / "adapter_model.safetensors")
        expected = row["selected_adapter_sha256"]
        if observed != expected:
            raise RuntimeError(f"accepted adapter mismatch for round {row['round']}: {observed}")
        adapter_paths.append(local_adapter)
        adapter_hashes.append(observed)
    mask_member = archive.getmember(f"{r8_prefix}/round_08/selected_mask.npz")
    archive.extract(mask_member, path=r8_dest, filter="data")

root_mask = r8_dest / r8_prefix / "round_08" / "selected_mask.npz"
chain = [r1_adapter, *adapter_paths]
model = model_root / "checkpoints/checkpoint-b"
baseline_mask = data_root / os.environ["BASELINE_MASK_RELATIVE"]
required = [
    model / "config.json",
    model / "model.safetensors",
    tokenizer_root / "tokenizer.json",
    baseline_mask,
    root_mask,
    *[path / "adapter_config.json" for path in chain],
    *[path / "adapter_model.safetensors" for path in chain],
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"missing staged inputs: {missing}")
(root / "adapter_chain.txt").write_text("".join(f"{path}\n" for path in chain))
receipt = {
    "model": str(model),
    "model_revision": os.environ["MODEL_REVISION"],
    "tokenizer": str(tokenizer_root),
    "tokenizer_revision": os.environ["TOKENIZER_REVISION"],
    "baseline_mask": str(baseline_mask),
    "r1_artifact": str(r1_artifact),
    "r1_artifact_sha256": r1_hash,
    "r8_artifact": str(r8_artifact),
    "r8_artifact_sha256": r8_hash,
    "root_mask": str(root_mask),
    "root_mask_sha256": sha256(root_mask),
    "adapter_chain": [str(path) for path in chain],
    "adapter_chain_sha256": [sha256(path / "adapter_model.safetensors") for path in chain],
    "accepted_rounds_in_chain": [1, *[int(row["round"]) for row in accepted]],
}
(root / "staging_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY

"${PYTHON}" -m pip freeze > "${RUN_ROOT}/setup/pip_freeze.txt"
cp "${INPUT_ROOT}/staging_receipt.json" "${RUN_ROOT}/setup/staging_receipt.json"
MODEL="${INPUT_ROOT}/model_repo/checkpoints/checkpoint-b"
TOKENIZER="${INPUT_ROOT}/tokenizer_repo"
ROOT_MASK="${INPUT_ROOT}/r8_root/issue23_best_checkpoint_tree_full_r8/run/round_08/selected_mask.npz"
BASELINE_MASK="${INPUT_ROOT}/data_repo/${BASELINE_MASK_RELATIVE}"
mapfile -t ROOT_ADAPTERS < "${INPUT_ROOT}/adapter_chain.txt"
ROOT_ARGS=()
for adapter in "${ROOT_ADAPTERS[@]}"; do
  ROOT_ARGS+=(--root-adapter "${adapter}")
done

set +e
timeout --signal=TERM --kill-after=60s 43200s \
  "${PYTHON}" "${RUNNER}" \
    --config "${CONFIG}" \
    --model "${MODEL}" \
    --tokenizer "${TOKENIZER}" \
    "${ROOT_ARGS[@]}" \
    --root-mask "${ROOT_MASK}" \
    --baseline-mask "${BASELINE_MASK}" \
    --baseline-mask-key mlp_rel_0.001 \
    --output-dir "${RUN_ROOT}/run" \
    --device cuda \
    --device-count 8 \
    --dtype bfloat16 \
    --eval-batch-size 128 \
    --seed 42 \
    --deadline-seconds 42000 \
  2>&1 | tee "${RUN_ROOT}/run.log"
STATUS="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${STATUS}" > "${RUN_ROOT}/setup/run_exit_code.txt"
printf '%s\n' "$(( $(date +%s) - STARTED_EPOCH ))" > "${RUN_ROOT}/setup/elapsed_wall_seconds.txt"
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RUN_ROOT}.tar.gz" "$(basename "${RUN_ROOT}")"
printf 'run_exit_code=%s artifact=%s\n' "${STATUS}" "${RUN_ROOT}.tar.gz"
exit "${STATUS}"
