#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENRLHF_DIR="${OPENRLHF_DIR:-$(cd -- "${SCRIPT_DIR}/../OpenRLHF" && pwd)}"
VENV_PATH="${VENV_PATH:-/root/autodl-tmp/venvs/swe-master-sft}"

source "${VENV_PATH}/bin/activate"
cd "${OPENRLHF_DIR}"

export OMP_NUM_THREADS=1

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a local smoke-test JSONL file}"
RUN_DIR="${OUTPUT_DIR:-/root/autodl-tmp/sft-runs/qwen3-4b-lora-smoke-$(date +%Y%m%d-%H%M%S)}"

[[ -d "${MODEL_PATH}" ]] || { echo "Model directory does not exist: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "Dataset does not exist: ${DATA_PATH}" >&2; exit 1; }
mkdir -p "${RUN_DIR}"


deepspeed --module openrlhf.cli.train_sft \
    --model.model_name_or_path "$MODEL_PATH" \
    --data.dataset "$DATA_PATH" \
    --data.input_key input \
    --data.apply_chat_template \
    --data.multiturn \
    --data.max_len 32768 \
    --data.dataloader_num_workers 0 \
    --train.micro_batch_size 1 \
    --train.batch_size 1 \
    --train.max_epochs 1 \
    --ds.zero_stage 2 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --ds.lora.rank 16 \
    --ds.lora.alpha 32 \
    --ds.lora.target_modules q_proj k_proj v_proj o_proj \
    --model.gradient_checkpointing_enable \
    --adam.lr 5e-5 \
    --logger.logging_steps 1 \
    --ckpt.output_dir "$RUN_DIR/adapter" \
    --ckpt.path "$RUN_DIR/checkpoints" \
    --ckpt.save_hf \
    --ckpt.disable_ds \
    2>&1 | tee "$RUN_DIR/train.log"

echo "训练完成，输出目录: $RUN_DIR"
