# Leaf training runbook

The implementation is pinned to `microsoft/FrogNano` commit
`5038f91560c635acb50daa452eb62c12daa737ad`. It keeps the existing R2E Docker
runtime and reward calculation, while replacing the agent protocol with Leaf.

## 1. Teacher rollout

Start the existing Docker SSH tunnel, then provide the teacher endpoint through
environment variables. Do not put credentials in configs or trajectory files.

```bash
cd /home/ecs-user/SWE-Master
export LEAF_TEACHER_BASE_URL=https://api.deepseek.com/v1
export LEAF_TEACHER_MODEL=deepseek-flash
read -rsp 'DeepSeek API key: ' LEAF_TEACHER_API_KEY; echo
export LEAF_TEACHER_API_KEY
export DOCKER_HOST=unix:///var/run/docker.sock

PYTHONPATH=R2E-Gym/src:DeepSWE_RL/rllm \
  R2E-Gym/.venv/bin/python \
  DeepSWE_RL/rllm/examples/swe/generate_leaf_sft.py \
  --input /home/ecs-user/swe-smith-teacher-deepseek-flash-2000x3/selected-tasks.jsonl \
  --output /home/ecs-user/leaf-sft/raw/swesmith-2000x3.jsonl \
  --seeds 0,1,2 --workers 4
```

Every raw rollout is retained. Only resolved, non-empty, naturally terminated
trajectories enter SFT.

The input is the existing stratified 2,000-task SWE-smith selection pinned to
dataset revision `ea6d7173829c7ec8fa16c22055699ff2e9188091`. Three seeds produce
6,000 raw Leaf trajectories. Do not substitute the R2E-Gym RL-readiness parquet.

## 2. Validate, replay, and tokenize

```bash
PYTHONPATH=R2E-Gym/src:DeepSWE_RL/rllm \
  R2E-Gym/.venv/bin/python \
  DeepSWE_RL/rllm/examples/swe/build_leaf_sft.py \
  --input /home/ecs-user/leaf-sft/raw/swesmith-2000x3.jsonl \
  --output-dir /home/ecs-user/leaf-sft/split --replay

for split in train validation; do
  PYTHONPATH=R2E-Gym/src:DeepSWE_RL/rllm \
    R2E-Gym/.venv/bin/python \
    OpenRLHF_SFT/SFT_data_pre_process/prepare_leaf_sft.py \
    --input /home/ecs-user/leaf-sft/split/${split}.jsonl \
    --output /home/ecs-user/leaf-sft/tokenized/${split}.jsonl \
    --model /home/ecs-user/models/Qwen3-4B-Instruct-2507
done
```

The tokenizer stage writes exact `input_ids` and assistant-only `loss_mask`
arrays. These are the same serialization used by Leaf RL rollouts.

## 3. SFT and RL gates

Run a small SFT overfit first, then the full job:

```bash
STORAGE_ROOT=/home/ecs-user \
  bash OpenRLHF_SFT/scripts_swe_master/train_qwen3_4b_leaf_sft.sh
```

Run four-task RL readiness by overriding `DATA` with a four-task parquet and
setting `TOTAL_TRAINING_STEPS=1`. Only after it produces valid rewards, masks,
patches, and resumable checkpoints should the full job start:

```bash
DATA=/path/to/four_tasks.parquet TOTAL_TRAINING_STEPS=1 \
  bash DeepSWE_RL/rllm/examples/swe/run_leaf_grpo.sh

bash DeepSWE_RL/rllm/examples/swe/run_leaf_grpo.sh
```

Leaf permits several calls in one assistant turn but executes them sequentially.
No reward is attached to the number of calls.
