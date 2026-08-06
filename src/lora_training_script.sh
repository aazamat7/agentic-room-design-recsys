cat > /home/jupyter/DiffSynth-Studio/train_qwen2511_hgtv_lora.sh <<'SH'
#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# Qwen-Image-Edit-2511 room-renovation LoRA training
# Reconstructs the run that produced step-2880.safetensors
# ============================================================

PYTHON="/opt/conda/bin/python"

REPO_ROOT="/home/jupyter/DiffSynth-Studio"

DATASET_ROOT="/home/jupyter/qwen2511_hgtv_augmented_lora_dataset"
METADATA_PATH="${DATASET_ROOT}/metadata_train.json"

OUTPUT_DIR="/home/jupyter/qwen2511_hgtv_lora_output"
LOG_PATH="${OUTPUT_DIR}/qwen2511_hgtv_lora_training.log"

EXPECTED_COMMIT="c02022681b09424e778b2d6275ee657a532834d3"

GCS_OUTPUT_URI="gs://adsp-s26-reccys-bucket/qwen2511-hgtv-lora/trained-lora-v1"

# ------------------------------------------------------------
# Cache configuration
# ------------------------------------------------------------

export HF_HOME="/home/jupyter/data/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

export DIFFSYNTH_DOWNLOAD_SOURCE="HuggingFace"

mkdir -p \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${OUTPUT_DIR}"

# ------------------------------------------------------------
# GPU/runtime configuration
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ------------------------------------------------------------
# Validate repository version
# ------------------------------------------------------------

cd "${REPO_ROOT}"

CURRENT_COMMIT="$(git rev-parse HEAD)"

echo "DiffSynth commit: ${CURRENT_COMMIT}"

if [[ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "ERROR: Incorrect DiffSynth commit."
    echo "Expected: ${EXPECTED_COMMIT}"
    echo "Actual:   ${CURRENT_COMMIT}"
    exit 1
fi

# ------------------------------------------------------------
# Validate required files
# ------------------------------------------------------------

if [[ ! -f "${METADATA_PATH}" ]]; then
    echo "ERROR: Metadata file not found:"
    echo "${METADATA_PATH}"
    exit 1
fi

# Validate metadata count and referenced images.
"${PYTHON}" - <<PY
import json
from pathlib import Path

dataset_root = Path("${DATASET_ROOT}")
metadata_path = Path("${METADATA_PATH}")

text = metadata_path.read_text(encoding="utf-8").strip()

if text.startswith("["):
    records = json.loads(text)
else:
    records = [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]

print("Training records:", len(records))

if len(records) != 576:
    raise RuntimeError(
        f"Expected 576 training records, found {len(records)}"
    )

required_keys = {
    "image",
    "edit_image",
    "prompt",
}

missing_files = []

for index, record in enumerate(records):
    missing_keys = required_keys - record.keys()

    if missing_keys:
        raise RuntimeError(
            f"Record {index} is missing keys: {missing_keys}"
        )

    # image is the target/after image.
    target_path = dataset_root / record["image"]

    # edit_image is the source/before image.
    source_path = dataset_root / record["edit_image"]

    if not target_path.exists():
        missing_files.append(str(target_path))

    if not source_path.exists():
        missing_files.append(str(source_path))

if missing_files:
    raise RuntimeError(
        "Missing referenced images. First examples:\n"
        + "\n".join(missing_files[:20])
    )

print("Dataset validation passed.")
print("Expected image presentations:", len(records) * 5)
PY

# ------------------------------------------------------------
# Save reproducibility configuration
# ------------------------------------------------------------

cat > "${OUTPUT_DIR}/training_config.json" <<JSON
{
  "base_model": "Qwen/Qwen-Image-Edit-2511",
  "dataset_root": "${DATASET_ROOT}",
  "metadata_path": "${METADATA_PATH}",
  "training_records": 576,
  "unique_training_pairs": 96,
  "dataset_repeat": 1,
  "num_epochs": 5,
  "expected_image_presentations": 2880,
  "learning_rate": 0.0001,
  "max_pixels": 1048576,
  "gradient_accumulation_steps": 2,
  "dataset_num_workers": 4,
  "save_steps": 250,
  "lora_rank": 32,
  "lora_base_model": "dit",
  "enable_model_cpu_offload": false,
  "zero_cond_t": true,
  "diffsynth_commit": "${EXPECTED_COMMIT}"
}
JSON

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

set -o pipefail

"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    examples/qwen_image/model_training/train.py \
    \
    --dataset_base_path "${DATASET_ROOT}" \
    --dataset_metadata_path "${METADATA_PATH}" \
    --data_file_keys "image,edit_image" \
    --extra_inputs "edit_image" \
    --dataset_repeat 1 \
    --dataset_num_workers 4 \
    --max_pixels 1048576 \
    \
    --model_id_with_origin_paths \
    "Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
    \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --gradient_accumulation_steps 2 \
    --save_steps 250 \
    \
    --output_path "${OUTPUT_DIR}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    \
    --lora_base_model "dit" \
    --lora_target_modules \
    "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
    --lora_rank 32 \
    \
    --use_gradient_checkpointing \
    --find_unused_parameters \
    --zero_cond_t \
    2>&1 | tee "${LOG_PATH}"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}."
    exit "${TRAIN_EXIT_CODE}"
fi

# ------------------------------------------------------------
# Verify final checkpoint
# ------------------------------------------------------------

FINAL_CHECKPOINT="${OUTPUT_DIR}/step-2880.safetensors"

if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
    echo "WARNING: Expected final checkpoint was not found:"
    echo "${FINAL_CHECKPOINT}"
    echo
    echo "Available checkpoints:"
    find "${OUTPUT_DIR}" \
        -maxdepth 1 \
        -name "step-*.safetensors" \
        -print \
        | sort -V
else
    echo "Final checkpoint created:"
    ls -lh "${FINAL_CHECKPOINT}"
fi

# ------------------------------------------------------------
# Back up results to GCS
# ------------------------------------------------------------

gcloud storage rsync -r \
    "${OUTPUT_DIR}" \
    "${GCS_OUTPUT_URI}"

echo
echo "Training and GCS backup complete."
echo "Local output: ${OUTPUT_DIR}"
echo "GCS output:   ${GCS_OUTPUT_URI}"
SH

chmod +x \
    /home/jupyter/DiffSynth-Studio/train_qwen2511_hgtv_lora.sh
