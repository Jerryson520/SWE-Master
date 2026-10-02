---
language:
- en
pretty_name: SWE-Master Teacher Rollout 300x3
tags:
- software-engineering
- trajectories
- tool-use
- supervised-fine-tuning
size_categories:
- n<1K
---

# SWE-Master Teacher Rollout 300x3

这是 SWE-Master 第一批 Teacher Rollout 的私有、版本化训练数据，包含不可变的原始 rollout 导出，以及处理后的 Qwen3 SFT 数据。

## `300x3` 的含义

- 选取了 300 个软件工程任务。
- 每个任务请求 3 次相互独立的 Teacher Rollout。
- 原始轨迹共 900 条。
- 本次运行的标识为 `teacher-deepseek-flash-official-300x3`。

这里的 `80k` 并不表示 80,000 条数据，而是指 Qwen3 使用 81,920 token 的上下文长度上限。

## 数据处理链路

```text
300 个任务 × 3 次 rollout
        │
        ▼
raw/.../batches/batch-0000..0009.jsonl        900 条原始轨迹
        │  检查成功状态、工具调用、submit 格式和 ≤100 步限制
        ▼
processed/.../clean_success_all.jsonl          776 条合格轨迹
        │  使用 Qwen3 tokenizer 计算长度，要求 ≤81,920 tokens
        ├── processed/.../qwen3-80k/token_length_rejected.jsonl   1 条被拒绝
        ▼
processed/.../qwen3-80k/clean_success_80k_qwen3.jsonl            775 条轨迹
        │  生成训练所需的 prompt、response 和 loss ranges
        ▼
processed/.../qwen3-80k/train_qwen3_80k.jsonl                    最终 SFT 输入
```

Qwen3-4B RSFT 实际使用的最终训练文件是：

`processed/teacher-deepseek-flash-official-300x3-rsft/qwen3-80k/train_qwen3_80k.jsonl`

## 目录及文件关系

| 路径 | 作用 | 与其他文件的关系 |
|---|---|---|
| `raw/.../manifest.json` | 运行清单 | 记录任务选择、配置、任务列表、fingerprint、Docker 信息和 run ID。 |
| `raw/.../execution.json` | 执行设置 | 记录 rollout 并发数，本次为 `max_workers: 4`。 |
| `raw/.../pulled_images.json` | 环境清单 | 记录这些任务使用并拉取过的 Docker images。 |
| `raw/.../batches/batch-NNNN.json` | Batch 索引 | 按顺序列出对应 JSONL batch 中的任务 ID。 |
| `raw/.../batches/batch-NNNN.jsonl` | 原始数据源 | 保存完整 rollout；十个 batch 合计 900 条轨迹。 |
| `processed/.../clean_success_all.jsonl` | 第一阶段合格数据 | 从 298 个任务中筛出的 776 条成功且结构合法的轨迹。 |
| `processed/.../rejected.jsonl` | 第一阶段审计记录 | 每条被拒绝的原始轨迹及其来源位置、拒绝原因。 |
| `processed/.../report.json` | 第一阶段统计报告 | 包括数量、仓库分布、步数分布和拒绝原因汇总。 |
| `processed/.../qwen3-80k/clean_success_80k_qwen3.jsonl` | 长度检查后的数据 | 775 条数据，并包含 Qwen3 序列长度及 assistant loss token 统计。 |
| `processed/.../qwen3-80k/token_length_rejected.jsonl` | 长度审计记录 | 唯一一条超过 81,920 token 上限的数据。 |
| `processed/.../qwen3-80k/token_length_report.json` | 长度统计报告 | 记录 tokenizer、长度阈值、数量和序列长度分布。 |
| `processed/.../qwen3-80k/train_qwen3_80k.jsonl` | 可直接训练的数据 | 增加 `prompt`、`response`、`response_ranges` 和 `prompt_ids_len`，是 SFT 的入口文件。 |

## 处理后数据如何追溯到原始 Rollout

每条合格数据均包含以下字段：

- `source_file`：原始 batch 文件名，例如 `batch-0000.jsonl`。
- `source_line`：该数据在 batch 中从 1 开始计算的行号。
- `exp_name`：包含任务和 rollout 编号的运行标识。
- `trajectory_fingerprint`：稳定的轨迹内容 fingerprint。
- `instance_id`：软件工程任务标识。

因此，任意一条处理后数据都能精确追溯至：

```text
raw/teacher-deepseek-flash-official-300x3/batches/<source_file>
                                                        第 <source_line> 行
```

这些身份字段和 fingerprint 会贯穿三个处理阶段。关联或审计数据时应使用这些字段，不应依赖文件中的行序。

## 数据字段

### 原始 Rollout

原始 JSONL 保留 rollout 输入、Agent 与环境配置、完整 `trajectory_steps`、输出 patch、测试结果、reward、token 统计和时间信息。它是归档数据源，不直接作为 SFT 输入。

### 第一阶段合格数据（`clean_success_all.jsonl`）

| 字段 | 含义 |
|---|---|
| `docker_image` | 可复现的任务运行环境。 |
| `instance_id` | R2E-Gym 任务标识。 |
| `input` | 按顺序保存的 chat/tool 多轮轨迹。 |
| `step_count` | Agent 执行步数。 |
| `token_usage_total` | Teacher Rollout 的 token 使用量。 |
| `reward`, `exit_reason` | Rollout 结果信息。 |
| `exp_name` | 唯一的实验及 rollout 名称。 |
| `source_file`, `source_line` | 精确的原始数据来源。 |
| `trajectory_fingerprint` | 稳定的轨迹 fingerprint。 |

### Qwen3 长度检查数据

在上述字段基础上增加 `qwen3_sequence_length` 和 `qwen3_assistant_loss_tokens`。

### 最终训练数据（`train_qwen3_80k.jsonl`）

进一步增加 `prompt`、`response`、`response_ranges` 和 `prompt_ids_len`。这些是 RSFT 直接使用的渲染后训练字段；同时保留 `input`，以便检查和重新处理。

## 质量控制结果

- 原始数据：300 个任务，共 900 条轨迹。
- 第一阶段合格：298 个任务，共 776 条轨迹。
- 第一阶段拒绝：124 条。单条数据可能有多个拒绝原因，因此各原因数量相加不一定等于 124。
- Qwen3 上下文长度过滤：保留 775 条，1 条超过 81,920 tokens 被拒绝。
- 最终 SFT 数据：775 条。
- 每个保留任务的成功 rollout 数量：23 个任务有 1 条，72 个任务有 2 条，203 个任务有 3 条。

权威的详细统计保存在 `report.json` 和 `token_length_report.json` 中，不通过文件名重复表达。

## 版本与复现

- Dataset revision：`teacher-rollout-300x3-v1`
- 对应 SWE-Master 源码提交：`157c845424ce100711bef85ba833ee489efff51a`
- 完整性清单：SWE-Master 仓库中的 `artifacts/teacher-rollout-300x3.json`
- 可见性：private。Rollout 可能包含源代码、模型轨迹和环境信息，未经单独审查不应公开。

训练和评测时应使用不可变的版本标签，不要使用未锁定的 `main` revision。
