#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Unified SkeletonLLM training launcher. Select the stage with STAGE=1..4.
#
#   STAGE 1  Render Warm-up          [REQUIRED]  train DrAction + projector
#   STAGE 2  Discriminative FT       [optional]  train DrAction + projector
#   STAGE 3  Causal Reasoning Distil [optional]  train DrAction + projector + LLM LoRA
#   STAGE 4  Recognition Refinement  [REQUIRED]  freeze DrAction; train projector + LLM LoRA
#
# The LLM and vision backbone are always frozen; only the parts listed above are
# trained. Stages 1/2/3/4 run for 1/1/1/3 epochs by default (paper setting).
#
# Required environment variables:
#   MODEL            starting checkpoint. Stage 1: base InternVL3-8B. Later
#                    stages: the previous stage's OUTPUT_DIR checkpoint.
#   DATA_ROOT        directory containing the .skeleton files.
#   ANNOTATION_FILE  this stage's annotation JSONL (see tools/build_*_annotation.py).
#   OUTPUT_DIR       where to write checkpoints and logs.
# Optional: GPUS(2) BATCH_SIZE(2) PER_DEVICE_BATCH_SIZE(1) MASTER_PORT EPOCHS.
# ---------------------------------------------------------------------------
set -euo pipefail
set -x

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

STAGE=${STAGE:?Set STAGE to 1, 2, 3, or 4.}
: "${MODEL:?Set MODEL to the starting checkpoint (Stage 1: base InternVL3-8B; later stages: previous stage OUTPUT_DIR).}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the .skeleton files.}"
: "${ANNOTATION_FILE:?Set ANNOTATION_FILE to the annotation JSONL for this stage.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the checkpoint/log directory.}"

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
MASTER_PORT=${MASTER_PORT:-32100}

# Per-stage trainable components and default epoch count.
case "${STAGE}" in
  1) RENDERER_TRAINABLE=True;  FREEZE_MLP=False; USE_LLM_LORA=0;  DEFAULT_EPOCHS=1 ;;
  2) RENDERER_TRAINABLE=True;  FREEZE_MLP=False; USE_LLM_LORA=0;  DEFAULT_EPOCHS=1 ;;
  3) RENDERER_TRAINABLE=True;  FREEZE_MLP=False; USE_LLM_LORA=32; DEFAULT_EPOCHS=1 ;;
  4) RENDERER_TRAINABLE=False; FREEZE_MLP=False; USE_LLM_LORA=32; DEFAULT_EPOCHS=3 ;;
  *) echo "STAGE must be one of 1, 2, 3, 4." >&2; exit 2 ;;
esac
EPOCHS=${EPOCHS:-${DEFAULT_EPOCHS}}

if (( GPUS <= 0 || PER_DEVICE_BATCH_SIZE <= 0 || BATCH_SIZE <= 0 )); then
  echo "GPUS, BATCH_SIZE, and PER_DEVICE_BATCH_SIZE must be positive integers." >&2
  exit 2
fi
DENOMINATOR=$((PER_DEVICE_BATCH_SIZE * GPUS))
if (( BATCH_SIZE % DENOMINATOR != 0 )); then
  echo "BATCH_SIZE must be divisible by GPUS * PER_DEVICE_BATCH_SIZE." >&2
  exit 2
fi
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export LAUNCHER=${LAUNCHER:-pytorch}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-20000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

OUTPUT_DIR=${OUTPUT_DIR%/}
META_CONFIG=${META_CONFIG:-"${OUTPUT_DIR}/meta_config.json"}
mkdir -p "${OUTPUT_DIR}"
cat > "${META_CONFIG}" <<EOF
{
  "ntu60_finetune": {
    "root": "${DATA_ROOT}",
    "annotation": "${ANNOTATION_FILE}",
    "data_augment": false,
    "max_dynamic_patch": 1,
    "repeat_time": 1,
    "length": 0
  }
}
EOF

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  skeletonllm/train/skeletonllm_chat_finetune.py \
  --model_name_or_path "${MODEL}" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_CONFIG}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 1 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --use_skeleton True \
  --skeleton_renderer_trainable "${RENDERER_TRAINABLE}" \
  --skeleton_enable_nfm True \
  --skeleton_num_line_samples 10 \
  --skeleton_target_num_frames 12 \
  --skeleton_fovx_deg 60.0 \
  --skeleton_fovy_deg 60.0 \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_mlp "${FREEZE_MLP}" \
  --use_llm_lora "${USE_LLM_LORA}" \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version "v2" \
  --deepspeed "${REPO_ROOT}/zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
