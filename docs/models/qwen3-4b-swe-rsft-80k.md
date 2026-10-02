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

这是一个面向长上下文、工具调用软件工程轨迹训练的私有 LoRA Adapter，属于 SWE-Master 项目。

## 重要说明：这是 Adapter，不是完整模型

本仓库包含约 66 MB 的 LoRA 权重，不包含 Qwen3 基座模型权重，因此不能单独加载运行。

```text
Qwen/Qwen3-4B-Instruct-2507      基座模型
               +
adapter_model.safetensors        本仓库的 LoRA 权重
               =
Qwen3-4B SWE RSFT 80K            可运行的适配后模型
```

使用时必须另行获取基座模型，并遵守基座模型自身的 License 和使用条款。

## 模型与数据血缘

```text
JJerry0000/swe-master-teacher-rollout-300x3
  processed/.../qwen3-80k/train_qwen3_80k.jsonl（775 条）
               │
               │ OpenRLHF SFT，1 epoch，LoRA
               ▼
JJerry0000/Qwen3-4B-SWE-RSFT-80K
  adapter_model.safetensors
```

- 训练数据：[`JJerry0000/swe-master-teacher-rollout-300x3`](https://huggingface.co/datasets/JJerry0000/swe-master-teacher-rollout-300x3)
- 训练文件：`processed/teacher-deepseek-flash-official-300x3-rsft/qwen3-80k/train_qwen3_80k.jsonl`
- 数据规模：经过结构检查和 81,920 token 长度过滤后，共 775 条训练轨迹。
- 源码仓库：[`Jerryson520/SWE-Master`](https://github.com/Jerryson520/SWE-Master)
- 训练入口：`OpenRLHF_SFT/scripts_swe_master/train_qwen3_4b_rsft_80k.sh`
- 推理服务入口：`scripts/serving/serve_qwen3_4b_rsft_80k.sh`
- OpenRLHF 源码版本：`46f8c119a1ec4cb885196f85b8243bd70eee953b`

## 仓库文件说明

| 路径 | 作用 |
|---|---|
| `adapter_model.safetensors` | LoRA 可训练权重，是本仓库最核心的模型文件。 |
| `adapter_config.json` | PEFT 配置及基座模型引用。 |
| `config.json` | 本次训练使用的 Qwen3 模型结构和配置快照。 |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | 用于复现输入格式的 tokenizer 和 chat template 快照。 |
| `training_config/train_qwen3_4b_rsft_80k.sh` | 完整训练命令及默认参数。 |
| `tensorboard/` | 训练过程中产生的 TensorBoard event logs，仅用于检查训练过程，推理时不需要。 |

DeepSpeed optimizer/checkpoint state 没有存放在本仓库。最新可恢复 checkpoint 保留在私有计算环境中，因为它体积较大，并且只在恢复训练时需要，运行 LoRA Adapter 时不需要。其 checksum 和存储类型记录在 SWE-Master 的 `artifacts/qwen3-4b-rsft-80k-lora.json` 中。

## 使用 Transformers 和 PEFT 加载

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

Hugging Face 账号需要拥有相应私有仓库的访问权限。为了保证结果可复现，应固定 Adapter revision。

## 使用 vLLM 部署

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --max-model-len 81920 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules qwen3-4b-rsft=JJerry0000/Qwen3-4B-SWE-RSFT-80K
```

SWE-Master 中的部署脚本使用相同的结构，并默认仅监听 localhost。

## 训练配置

| 配置项 | 值 |
|---|---|
| 训练目标 | 对多轮工具调用轨迹进行 Supervised Fine-Tuning |
| 训练框架 | OpenRLHF + DeepSpeed |
| 基座模型 | `Qwen/Qwen3-4B-Instruct-2507` |
| 训练样本数 | 775 |
| 最大序列长度 | 81,920 tokens |
| Epochs | 1 |
| Global batch size | 16 |
| Micro-batch size | 1 |
| Learning rate | `5e-6` |
| Precision | bfloat16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| DeepSpeed | ZeRO stage 2 |
| Attention | Flash Attention 2 + Ring Attention size 2 |
| Packing | 开启 |
| Gradient checkpointing | 开启 |

默认启动配置使用 2 张 GPU。训练脚本中的本机路径均可通过环境变量覆盖，它们不是可移植的模型标识。

## 适用范围

- 研究和内部评测使用工具的软件工程 Agent。
- 基于相同 Qwen3 基座模型继续进行 SWE 方向实验。
- 通过 PEFT/Transformers 或支持 LoRA 的 vLLM 等推理引擎运行。

## 局限与评测状态

- 本 Adapter 仅使用来自 298 个软件工程任务的 775 条轨迹训练，覆盖范围有限。
- 训练数据包含较长的 Agent/Tool 交互，可能保留无效操作、格式噪声或与特定环境相关的行为。
- 80K 的训练上下文上限并不意味着模型在所有长上下文长度下都能保持可靠表现。
- 本仓库目前没有附带权威 benchmark 结果。不能根据仓库名称或 training loss 推断 SWE-bench 性能。
- 模型能够生成代码和 shell 命令。执行前应人工检查，并仅在适当隔离的环境中运行。

后续增加评测结果时，必须同时记录评测数据集、harness 版本、解码参数和原始结果文件，确保指标可以复现。

## 版本与完整性

- Adapter revision：`qwen3-4b-rsft-80k-lora-v1`
- Adapter SHA-256：`372e14114b5e971332845dcc26c7fba0c3ae41c9386e564ee637147fca982d65`
- Adapter 记录的 PEFT 版本：`0.20.0`
- Artifact manifest：SWE-Master 中的 `artifacts/qwen3-4b-rsft-80k-lora.json`
- 可见性：private

部署时应使用版本标签并校验 checksum，以保证结果可复现。
