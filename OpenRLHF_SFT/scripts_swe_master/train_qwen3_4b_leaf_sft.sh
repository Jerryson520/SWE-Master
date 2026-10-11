#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENRLHF_DIR="$(cd -- "${SCRIPT_DIR}/../OpenRLHF" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-${HOME}}"

MODEL_PATH="${MODEL_PATH:-${STORAGE_ROOT}/models/Qwen3-4B-Instruct-2507}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DATA_PATH:-${STORAGE_ROOT}/leaf-sft/tokenized/train.jsonl}}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${STORAGE_ROOT}/leaf-sft/tokenized/validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/sft_models/qwen3-4b-leaf-sft}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${STORAGE_ROOT}/tensorboard/qwen3-4b-leaf-sft}"
CKPT_DIR="${CKPT_DIR:-${STORAGE_ROOT}/sft_checkpoints/qwen3-4b-leaf-sft}"
SAVE_STEPS="${SAVE_STEPS:-10}"
MAX_CKPT_NUM="${MAX_CKPT_NUM:-1}"
MAX_CKPT_MEM_GB="${MAX_CKPT_MEM_GB:-20}"
CKPT_LOAD_ENABLE="${CKPT_LOAD_ENABLE:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
RING_ATTN_SIZE="${RING_ATTN_SIZE:-4}"
RING_ATTN_HEAD_STRIDE="${RING_ATTN_HEAD_STRIDE:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGER="${LOGGER:-tensorboard}"
WANDB_PROJECT="${WANDB_PROJECT:-swe-leaf-sft}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4b-leaf}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-4b-leaf-sft}"

[[ -d "${MODEL_PATH}" ]] || { echo "模型目录不存在: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${TRAIN_DATA_PATH}" ]] || { echo "训练数据不存在: ${TRAIN_DATA_PATH}" >&2; exit 1; }
[[ -f "${EVAL_DATA_PATH}" ]] || { echo "验证数据不存在: ${EVAL_DATA_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${CKPT_DIR}"
(( NUM_GPUS >= 2 )) || { echo "80K Ring Attention 训练至少需要 2 张 GPU" >&2; exit 1; }
(( RING_ATTN_SIZE >= 2 )) || { echo "RING_ATTN_SIZE 必须至少为 2" >&2; exit 1; }
(( NUM_GPUS % RING_ATTN_SIZE == 0 )) || {
  echo "NUM_GPUS (${NUM_GPUS}) 必须能被 RING_ATTN_SIZE (${RING_ATTN_SIZE}) 整除" >&2
  exit 1
}
[[ "${ZERO_STAGE}" == "2" || "${ZERO_STAGE}" == "3" ]] || {
  echo "ZERO_STAGE 只能是 2 或 3，当前为: ${ZERO_STAGE}" >&2
  exit 1
}

CKPT_LOAD_ARGS=()
if [[ "${CKPT_LOAD_ENABLE}" == "1" ]]; then
  CKPT_LOAD_ARGS+=(--ckpt.load_enable)
fi

LOGGER_ARGS=()
case "${LOGGER}" in
  tensorboard)
    LOGGER_ARGS+=(--logger.tensorboard_dir "${TENSORBOARD_DIR}" --logger.wandb.run_name "${WANDB_RUN_NAME}")
    ;;
  wandb)
    : "${WANDB_API_KEY:?LOGGER=wandb 时必须设置 WANDB_API_KEY}"
    LOGGER_ARGS+=(
      --logger.wandb.key "${WANDB_API_KEY}"
      --logger.wandb.project "${WANDB_PROJECT}"
      --logger.wandb.group "${WANDB_GROUP}"
      --logger.wandb.run_name "${WANDB_RUN_NAME}"
    )
    [[ -z "${WANDB_ORG:-}" ]] || LOGGER_ARGS+=(--logger.wandb.org "${WANDB_ORG}")
    ;;
  *) echo "LOGGER 只能是 tensorboard 或 wandb，当前为: ${LOGGER}" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "${OPENRLHF_DIR}"

exec deepspeed --num_gpus "${NUM_GPUS}" --module openrlhf.cli.train_sft \
  --model.model_name_or_path "${MODEL_PATH}" \
  --data.dataset "${TRAIN_DATA_PATH}" \
  --eval.dataset "${EVAL_DATA_PATH}" \
  --data.input_key input \
  --data.max_len 81920 \
  --data.multiturn \
  --train.max_epochs 1 \
  --train.batch_size 16 \
  --train.micro_batch_size 1 \
  --adam.lr 5e-6 \
  --ds.packing_samples \
  --ds.ring_attn_size "${RING_ATTN_SIZE}" \
  --ds.ring_attn_head_stride "${RING_ATTN_HEAD_STRIDE}" \
  --ds.zero_stage "${ZERO_STAGE}" \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --model.gradient_checkpointing_enable \
  --ckpt.output_dir "${OUTPUT_DIR}" \
  --ckpt.save_steps "${SAVE_STEPS}" \
  --ckpt.path "${CKPT_DIR}" \
  --ckpt.max_num "${MAX_CKPT_NUM}" \
  --ckpt.max_mem "${MAX_CKPT_MEM_GB}" \
  --logger.logging_steps 1 \
  --eval.steps "${EVAL_STEPS}" \
  "${LOGGER_ARGS[@]}" \
  "${CKPT_LOAD_ARGS[@]}"
