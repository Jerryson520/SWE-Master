#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENRLHF_DIR="$(cd -- "${SCRIPT_DIR}/../OpenRLHF" && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-/root/SWE-Master/OpenRLHF_SFT/SFT_data_pre_process/outputs/teacher-deepseek-flash-official-300x3-rsft/qwen3-80k/train_qwen3_80k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/sft_models/qwen3-4b-rsft-80k-lora}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-/root/autodl-tmp/tensorboard/qwen3-4b-rsft-80k-lora}"
CKPT_DIR="${CKPT_DIR:-/root/autodl-tmp/sft_checkpoints/qwen3-4b-rsft-80k-4gpu}"
SAVE_STEPS="${SAVE_STEPS:-10}"
MAX_CKPT_NUM="${MAX_CKPT_NUM:-1}"
MAX_CKPT_MEM_GB="${MAX_CKPT_MEM_GB:-20}"
CKPT_LOAD_ENABLE="${CKPT_LOAD_ENABLE:-0}"
NUM_GPUS="${NUM_GPUS:-2}"
RING_ATTN_SIZE="${RING_ATTN_SIZE:-2}"
RING_ATTN_HEAD_STRIDE="${RING_ATTN_HEAD_STRIDE:-1}"

[[ -d "${MODEL_PATH}" ]] || { echo "模型目录不存在: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "训练数据不存在: ${DATA_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${CKPT_DIR}"
(( NUM_GPUS >= 2 )) || { echo "80K Ring Attention 训练至少需要 2 张 GPU" >&2; exit 1; }
(( RING_ATTN_SIZE >= 2 )) || { echo "RING_ATTN_SIZE 必须至少为 2" >&2; exit 1; }
(( NUM_GPUS % RING_ATTN_SIZE == 0 )) || {
  echo "NUM_GPUS (${NUM_GPUS}) 必须能被 RING_ATTN_SIZE (${RING_ATTN_SIZE}) 整除" >&2
  exit 1
}

CKPT_LOAD_ARGS=()
if [[ "${CKPT_LOAD_ENABLE}" == "1" ]]; then
  CKPT_LOAD_ARGS+=(--ckpt.load_enable)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "${OPENRLHF_DIR}"

exec deepspeed --num_gpus "${NUM_GPUS}" --module openrlhf.cli.train_sft \
  --model.model_name_or_path "${MODEL_PATH}" \
  --data.dataset "${DATA_PATH}" \
  --data.input_key input \
  --data.max_len 81920 \
  --data.apply_chat_template \
  --data.multiturn \
  --train.max_epochs 1 \
  --train.batch_size 16 \
  --train.micro_batch_size 1 \
  --adam.lr 5e-6 \
  --ds.packing_samples \
  --ds.ring_attn_size "${RING_ATTN_SIZE}" \
  --ds.ring_attn_head_stride "${RING_ATTN_HEAD_STRIDE}" \
  --ds.zero_stage 2 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --ds.lora.rank 16 \
  --ds.lora.alpha 32 \
  --ds.lora.dropout 0.05 \
  --model.gradient_checkpointing_enable \
  --ckpt.output_dir "${OUTPUT_DIR}" \
  --ckpt.save_steps "${SAVE_STEPS}" \
  --ckpt.path "${CKPT_DIR}" \
  --ckpt.max_num "${MAX_CKPT_NUM}" \
  --ckpt.max_mem "${MAX_CKPT_MEM_GB}" \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "${TENSORBOARD_DIR}" \
  --eval.steps -1 \
  "${CKPT_LOAD_ARGS[@]}"
