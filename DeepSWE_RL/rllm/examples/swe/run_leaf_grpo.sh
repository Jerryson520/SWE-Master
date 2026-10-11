#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
STORAGE_ROOT="${STORAGE_ROOT:-${HOME}}"
cd "$ROOT"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

RLLM_VENV="${RLLM_VENV:-${HOME}/venvs/rllm}"
export PATH="${RLLM_VENV}/bin:${PATH}"
export PYTHONPATH="$ROOT/DeepSWE_RL/rllm:$ROOT/R2E-Gym/src${PYTHONPATH:+:${PYTHONPATH}}"
export DOCKER_HOST="tcp://127.0.0.1:2375"
unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# vLLM's CuMemAllocator sleep/wake memory pool is incompatible with
# expandable_segments in vLLM 0.8.5.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export HYDRA_FULL_ERROR=1

# W&B credentials are supplied by `wandb login` or WANDB_API_KEY. Never put
# the key in this script. Set WANDB_MODE=offline only for an intentional local run.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-swe-leaf-grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RESUME="allow"

PYTHON="${PYTHON:-${RLLM_VENV}/bin/python}"
DATA="${DATA:-${ROOT}/rl_readiness/teacher_300_grpo.parquet}"
IMAGE_LIST="${IMAGE_LIST:-${ROOT}/rl_readiness/teacher_300_images.txt}"
MODEL="${MODEL:-${STORAGE_ROOT}/sft_models/qwen3-4b-leaf-sft}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/grpo_runs/leaf}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

if ! curl --silent --fail http://127.0.0.1:2375/_ping | grep -q '^OK$'; then
    echo "Docker SSH tunnel is unavailable at 127.0.0.1:2375" >&2
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "Training data is missing: $DATA" >&2
    exit 1
fi
if [ ! -f "$IMAGE_LIST" ]; then
    echo "Image list is missing: $IMAGE_LIST" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "Model is missing: $MODEL" >&2
    exit 1
fi
LORA_ARGS=()
RUN_KIND=full
if [ -n "$LORA_ADAPTER" ]; then
    if [ ! -f "$LORA_ADAPTER/adapter_config.json" ] || \
       { [ ! -f "$LORA_ADAPTER/adapter_model.safetensors" ] && [ ! -f "$LORA_ADAPTER/adapter_model.bin" ]; }; then
        echo "LoRA adapter is incomplete: $LORA_ADAPTER" >&2
        exit 1
    fi
    read -r LORA_RANK LORA_ALPHA < <(
        "$PYTHON" - "$LORA_ADAPTER/adapter_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
print(config["r"], config["lora_alpha"])
PY
    )
    RUN_KIND=lora
    ACTOR_LR="${ACTOR_LR:-5e-6}"
    LORA_ARGS=(
        actor_rollout_ref.model.lora_rank="$LORA_RANK"
        actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
        actor_rollout_ref.model.lora_dropout=0.0
        actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER"
        actor_rollout_ref.rollout.load_format=safetensors
    )
    "$PYTHON" "$ROOT/DeepSWE_RL/rllm/examples/swe/preflight_grpo_teacher_300_lora.py" \
        --base "$MODEL" \
        --adapter "$LORA_ADAPTER" \
        --data "$DATA" \
        --image-list "$IMAGE_LIST"
else
    ACTOR_LR="${ACTOR_LR:-1e-6}"
fi
TRAINING_LIMIT_ARGS=()
if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    TRAINING_LIMIT_ARGS=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi
if [ "$WANDB_MODE" = online ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -q 'api.wandb.ai' /root/.netrc 2>/dev/null; then
    echo "W&B is not logged in. Run: /root/venvs/rllm/bin/wandb login" >&2
    exit 1
fi

if [ -n "${RESUME_RUN_DIR:-}" ]; then
    RUN_DIR="$RESUME_RUN_DIR"
    TS="$(basename "$RUN_DIR")"
else
    TS="$(TZ=Asia/Shanghai date +'%Y-%m-%d_%H-%M-%S')"
    RUN_DIR="${OUTPUT_ROOT}/${TS}"
fi

PROJECT_NAME="$WANDB_PROJECT"
EXPERIMENT_NAME="swe-leaf-grpo-${RUN_KIND}-n4-tp4-${TS}"
DEFAULT_LOCAL_DIR="${RUN_DIR}/checkpoints"
ROLLOUT_DATA_DIR="${RUN_DIR}/rollouts"
export WANDB_RUN_ID="${WANDB_RUN_ID:-swe-leaf-grpo-${RUN_KIND}-n4-tp4-${TS}}"

mkdir -p "$DEFAULT_LOCAL_DIR" "$ROLLOUT_DATA_DIR" "${RUN_DIR}/logs"
printf '%s\n' "$RUN_DIR" > "${OUTPUT_ROOT}/latest_run_dir"

echo "Run directory: $RUN_DIR"
echo "W&B project/run: $PROJECT_NAME / $EXPERIMENT_NAME"
echo "Policy initialization: $MODEL"
echo "RL mode: $RUN_KIND; actor LR: $ACTOR_LR"
if [ -n "$LORA_ADAPTER" ]; then
    echo "SFT adapter: $LORA_ADAPTER (rank=$LORA_RANK, alpha=$LORA_ALPHA)"
fi
echo "Epochs: $TOTAL_EPOCHS; checkpoint every $SAVE_FREQ trainer steps"
if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    echo "Pilot limit: $TOTAL_TRAINING_STEPS trainer steps"
fi

exec "$PYTHON" -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=loop \
    data.train_files="$DATA" \
    data.val_files="$DATA" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=8192 \
    data.max_response_length=73728 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.model.path="$MODEL" \
    "${LORA_ARGS[@]}" \
    "${TRAINING_LIMIT_ARGS[@]}" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=81920 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.max_model_len=81920 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.mask_truncated_samples=False \
    algorithm.clip_advantages=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    actor_rollout_ref.actor.checkpoint.contents="['model','optimizer','extra']" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    trainer.rl_readiness_only=False \
    trainer.default_hdfs_dir=null \
    env.name=swe_leaf \
    +env.env_args.delete_image=False \
    +env.env_args.backend=docker \
    agent.name=leafagent \
    agent.max_steps=150 \
    agent.overlong_filter=True \
    agent.trajectory_timeout=3600 \
    agent.async_engine=True \
    +agent.engine_args.max_tokens_per_request=8192 \
    +agent.engine_args.context_safety_margin=8192 \
    +agent.agent_args.parallel_tool_calls=True \
    trainer.total_epochs="$TOTAL_EPOCHS"
