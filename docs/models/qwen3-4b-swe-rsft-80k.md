---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-4B-Instruct-2507
- lora
- transformers
- software-engineering
- tool-use
- supervised-fine-tuning
---

# Qwen3-4B SWE RSFT 80K

Private LoRA adapter trained for long-context, tool-using software-engineering trajectories in SWE-Master.

## Important: this is an adapter, not a full model

This repository contains approximately 66 MB of LoRA weights. It does **not** contain the Qwen3 base-model weights and cannot be loaded by itself.

```text
Qwen/Qwen3-4B-Instruct-2507      base model
               +
adapter_model.safetensors        this repository
               =
Qwen3-4B SWE RSFT 80K            runnable adapted model
```

The base model must be available separately and is subject to its own license and usage terms.

## Artifact lineage

```text
JJerry0000/swe-master-teacher-rollout-300x3
  processed/.../qwen3-80k/train_qwen3_80k.jsonl (775 rows)
               │
               │ OpenRLHF SFT, one epoch, LoRA
               ▼
JJerry0000/Qwen3-4B-SWE-RSFT-80K
  adapter_model.safetensors
```

- Training dataset: [`JJerry0000/swe-master-teacher-rollout-300x3`](https://huggingface.co/datasets/JJerry0000/swe-master-teacher-rollout-300x3)
- Dataset input: `processed/teacher-deepseek-flash-official-300x3-rsft/qwen3-80k/train_qwen3_80k.jsonl`
- Dataset size: 775 training trajectories after validation and the 81,920-token limit.
- Source code: [`Jerryson520/SWE-Master`](https://github.com/Jerryson520/SWE-Master)
- Training entry point: `OpenRLHF_SFT/scripts_swe_master/train_qwen3_4b_rsft_80k.sh`
- Serving entry point: `scripts/serving/serve_qwen3_4b_rsft_80k.sh`
- OpenRLHF source revision: `46f8c119a1ec4cb885196f85b8243bd70eee953b`

## Repository contents

| Path | Purpose |
|---|---|
| `adapter_model.safetensors` | LoRA trainable weights; this is the primary model artifact. |
| `adapter_config.json` | PEFT configuration and base-model reference. |
| `config.json` | Qwen3 architecture/configuration snapshot used by the run. |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Tokenizer and chat-template snapshot for reproducible formatting. |
| `training_config/train_qwen3_4b_rsft_80k.sh` | Exact training command and defaults. |
| `tensorboard/` | TensorBoard event logs from training attempts/runs; useful for audit, not required for inference. |

DeepSpeed optimizer/checkpoint state is intentionally not stored in this repository. The latest resumable checkpoint is retained on private infrastructure because it is much larger and is only needed to resume training, not to run the adapter. Its checksum and location class are recorded in `artifacts/qwen3-4b-rsft-80k-lora.json` in SWE-Master.

## Load with Transformers and PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-4B-Instruct-2507"
adapter_id = "JJerry0000/Qwen3-4B-SWE-RSFT-80K"
revision = "qwen3-4b-rsft-80k-lora-v1"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, revision=revision)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id, revision=revision)
```

The Hugging Face account must have access to both private artifacts where applicable. Pin the adapter revision for reproducible use.

## Serve with vLLM

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --max-model-len 81920 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules qwen3-4b-rsft=JJerry0000/Qwen3-4B-SWE-RSFT-80K
```

The checked-in SWE-Master serving script uses the same structure and defaults to localhost-only serving.

## Training configuration

| Setting | Value |
|---|---|
| Objective | Supervised fine-tuning on multi-turn tool trajectories |
| Framework | OpenRLHF + DeepSpeed |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Examples | 775 |
| Maximum sequence length | 81,920 tokens |
| Epochs | 1 |
| Global batch size | 16 |
| Micro-batch size | 1 |
| Learning rate | `5e-6` |
| Precision | bfloat16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| DeepSpeed | ZeRO stage 2 |
| Attention | Flash Attention 2 + Ring Attention size 2 |
| Packing | Enabled |
| Gradient checkpointing | Enabled |

The default launch configuration uses two GPUs. Machine paths in the script are overridable through environment variables and are not portable identifiers.

## Intended use

- Research and internal evaluation of tool-using software-engineering agents.
- Continued SWE-focused experimentation from the matching Qwen3 base model.
- Inference through PEFT/Transformers or a LoRA-capable serving engine such as vLLM.

## Limitations and evaluation status

- This adapter was trained on only 775 trajectories from 298 software-engineering tasks; coverage is narrow.
- Training examples contain long agent/tool interactions and may reproduce ineffective actions, formatting artifacts or environment-specific behavior.
- An 80K training context limit does not guarantee reliable behavior at every long-context length.
- No authoritative benchmark result is recorded with this artifact yet. Do not infer SWE-bench performance from the repository name or training loss.
- The model can generate code and shell commands. Execute outputs only in an appropriately isolated environment and review changes before applying them.

Evaluation results should be added only with the evaluation dataset, harness revision, decoding settings and raw result artifact so that the numbers are reproducible.

## Version and integrity

- Adapter revision: `qwen3-4b-rsft-80k-lora-v1`
- Adapter SHA-256: `372e14114b5e971332845dcc26c7fba0c3ae41c9386e564ee637147fca982d65`
- PEFT version recorded by the adapter: `0.20.0`
- Artifact manifest: `artifacts/qwen3-4b-rsft-80k-lora.json` in SWE-Master
- Visibility: private

Use the version tag and verify the checksum for a reproducible deployment.
