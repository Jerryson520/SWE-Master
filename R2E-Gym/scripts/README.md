# Teacher Rollout 数据采集复现说明

本文档说明如何使用 `collect_teacher_rollouts.py` 从本地
`R2E-Gym/R2E-Gym-Subset` Parquet 数据中抽取任务，通过 OpenAI 兼容的教师模型接口生成轨迹，并在 Docker 环境中计算 reward。

本文记录的是本项目实际完成的 `300 tasks × 3 rollouts` 配置。真实 API Key 不得写入本文、代码、Git、日志或结果目录。

## 1. 本次实验配置

| 项目 | 配置 |
| --- | --- |
| 教师模型 | `openai/deepseek-flash` |
| API Base | `https://api.deepseek.com/v1` |
| 数据集 | `R2E-Gym/R2E-Gym-Subset`，本地 Parquet `train` split |
| 任务选择 | 按仓库比例分层抽样，稳定哈希排序 |
| 任务数 | 300 |
| 抽样 seed | 42 |
| 每题 rollout 数 | 3 |
| 并发 | 4 |
| 每批任务数 | 30 |
| temperature | 1.0 |
| 最大交互步数 | 150 |
| scaffold | OpenHands |
| function calling | 开启 |
| context window | 131072 |
| 单次最大输出 | 8192 tokens |
| context safety margin | 4096 tokens |
| 轨迹累计输出预算 | 131072 tokens |
| LSP / memory compression | 均关闭 |

`temperature`、步数、scaffold 和 function calling 等参数目前固定在采集脚本的 `config` 中；其余预算和任务参数由 CLI 显式传入。

## 2. 前置条件

需要准备：

1. Linux 主机和本机 Docker daemon；
2. 已克隆的 SWE-Master 仓库；
3. 可运行 R2E-Gym 的 Python 3.10+ 环境；
4. 已下载的 `R2E-Gym/R2E-Gym-Subset` Parquet 文件；
5. 可访问教师模型的 OpenAI 兼容 API；
6. 足够的 Docker data-root 磁盘空间。

本次服务器路径为：

```text
/home/ecs-user/SWE-Master/R2E-Gym
/home/ecs-user/swe_datasets/R2E-Gym-Subset
```

路径不是数据身份的一部分，可以在其他机器上改变；脚本会计算所有 Parquet 文件的 SHA-256 指纹。使用已有输出目录恢复时，数据指纹必须与 `manifest.json` 一致。

进入环境：

```bash
cd /home/ecs-user/SWE-Master/R2E-Gym
source .venv/bin/activate
python --version
docker info >/dev/null
```

如果要从空环境安装，应使用仓库锁定的依赖方式创建环境；不要只复制结果目录后假设 Python 依赖已经存在。

## 3. 安全设置教师 API

API Key 只通过当前进程环境传入。关机、退出 shell 或重新登录后，需要重新设置。

```bash
export OPENAI_API_BASE="https://api.deepseek.com/v1"
read -rsp "OPENAI_API_KEY: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
```

不要把真实 Key 放入：

- README 或脚本；
- shell 历史；
- `.bashrc`、`.zshrc` 等会长期明文保存的文件；
- `manifest.json`、轨迹 JSON 或 Git commit。

如使用无鉴权的本地兼容接口，可以显式设置：

```bash
export OPENAI_API_KEY=EMPTY
```

## 4. Docker 镜像网络

脚本需要拉取每个 R2E 任务对应的 `namanjain12/*_final:<commit>` 镜像。本次服务器采用两级策略：

1. 首先按原始 Docker Hub 名称拉取，由 dockerd 的 registry mirror 加速；
2. mirror 失败时，脚本显式访问 `registry-1.docker.io`，由 dockerd 的 HTTPS 代理兜底，并重试。

示例 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": ["<你的阿里云镜像加速器地址>"],
  "max-concurrent-downloads": 1
}
```

如果所在网络能直接稳定访问 Docker Hub，不需要配置代理。若必须使用代理，代理应运行在服务器本身，dockerd 的 systemd 环境可配置为：

```ini
HTTP_PROXY=http://127.0.0.1:1082
HTTPS_PROXY=http://127.0.0.1:1082
NO_PROXY=localhost,127.0.0.1,::1,.aliyuncs.com
```

修改 daemon 配置后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
docker info
```

这些是主机级配置，不会被 Git 保存。换机器时必须重新检查。不要直接复制代理端口，除非新机器确实在该端口运行代理。

## 5. 只读检查

正式请求 API 前，先验证数据加载和确定性抽样：

```bash
cd /home/ecs-user/SWE-Master/R2E-Gym
source .venv/bin/activate

python scripts/collect_teacher_rollouts.py \
  --source-dir /home/ecs-user/swe_datasets/R2E-Gym-Subset \
  --output-dir ./results/teacher-dry-run \
  --task-count 300 \
  --selection-seed 42 \
  --batch-size 30 \
  --max-workers 4 \
  --rollouts 3 \
  --min-free-gb 10 \
  --llm-name "openai/deepseek-flash" \
  --context-window 131072 \
  --max-output-tokens 8192 \
  --context-safety-margin 4096 \
  --max-trajectory-output-tokens 131072 \
  --dry-run
```

`--dry-run` 不写结果、不拉镜像，也不调用模型。

## 6. 一题 smoke test

首次在新机器运行时，先使用独立输出目录完成 `1 task × 3 rollouts`：

```bash
python scripts/collect_teacher_rollouts.py \
  --source-dir /home/ecs-user/swe_datasets/R2E-Gym-Subset \
  --output-dir ./results/teacher-deepseek-flash-smoke-1x3 \
  --task-count 1 \
  --selection-seed 42 \
  --batch-size 1 \
  --max-workers 1 \
  --rollouts 3 \
  --min-free-gb 10 \
  --llm-name "openai/deepseek-flash" \
  --context-window 131072 \
  --max-output-tokens 8192 \
  --context-safety-margin 4096 \
  --max-trajectory-output-tokens 131072
```

验收：三个 `rollout-*.json` 都能解析；模型产生工具 action；环境返回 observation；`submit` 正常结束；reward 是明确的 0 或 1，而不是协议或程序异常。

## 7. 正式 300 × 3 采集命令

```bash
cd /home/ecs-user/SWE-Master/R2E-Gym
source .venv/bin/activate

python scripts/collect_teacher_rollouts.py \
  --source-dir /home/ecs-user/swe_datasets/R2E-Gym-Subset \
  --output-dir ./results/teacher-deepseek-flash-official-300x3 \
  --task-count 300 \
  --selection-seed 42 \
  --batch-size 30 \
  --max-workers 4 \
  --rollouts 3 \
  --min-free-gb 10 \
  --llm-name "openai/deepseek-flash" \
  --context-window 131072 \
  --max-output-tokens 8192 \
  --context-safety-margin 4096 \
  --max-trajectory-output-tokens 131072
```

建议在 tmux 中运行：

```bash
tmux new -s teacher-rollout
# 在 tmux 内设置 API 环境变量并执行上面的正式命令。
# Ctrl-b d：脱离；tmux attach -t teacher-rollout：重新进入。
```

## 8. 中断与恢复

每份轨迹先原子写入独立 JSON。进程异常、SSH 断开或服务器重启后，使用相同输出目录和相同参数重跑即可：

- 已存在且有效的结果会被跳过；
- 缺失的 rollout 会继续执行；
- `manifest.json` 会检查数据指纹、任务选择、模型配置和 Docker daemon 身份；
- 参数或 Docker daemon 身份不一致时，脚本会拒绝覆盖已有结果。

恢复时只有 `--max-workers` 可以按机器负载调整；它单独记录在 `execution.json`。任务数、seed、rollouts、batch size、模型、token 预算、API Base 和输入数据必须保持一致，否则应使用新的输出目录。

## 9. 结果目录

关键文件：

```text
results/<run-name>/
├── manifest.json
├── execution.json
├── pulled_images.json
├── batches/
│   ├── batch-0000.json
│   └── batch-0000.jsonl
└── <instance_id>/
    ├── rollout-0.json
    ├── rollout-1.json
    └── rollout-2.json
```

- 单题目录下的 `rollout-*.json` 是断点恢复和后续筛选使用的原始结果；
- `batches/*.jsonl` 是每批完成后按固定顺序生成的汇总；
- `manifest.json` 保存数据指纹、固定任务列表、镜像、抽样方法和模型配置，但不保存 API Key；
- `execution.json` 保存可调整的执行并发；
- 运行前已有镜像不会被全局 prune；任务完成后脚本只清理本轮任务对应且未被容器引用的镜像。

完成数量检查：

```bash
RUN_DIR=./results/teacher-deepseek-flash-official-300x3

find "$RUN_DIR" -mindepth 2 -maxdepth 2 \
  -name 'rollout-*.json' -type f | wc -l
```

正式实验应得到 `900`。

检查 reward：

```bash
find "$RUN_DIR" -mindepth 2 -maxdepth 2 \
  -name 'rollout-*.json' -type f -print0 \
  | xargs -0 jq -r '.reward' \
  | sort | uniq -c
```

本次完成结果为：900 条轨迹，其中 `reward=1` 为 886 条，`reward=0` 为 14 条。

## 10. 备份

从运行服务器增量备份到本机：

```bash
rsync -a --partial \
  swe-master:/home/ecs-user/SWE-Master/R2E-Gym/results/teacher-deepseek-flash-official-300x3/ \
  ~/Project/teacher-deepseek-flash-official-300x3/
```

结果目录体积较大，不应提交到 Git。应提交采集脚本、必要源码修改和本文档，并把数据保存到独立存储或备份目录。

## 11. 复现边界

完全复现不仅需要这条命令，还需要：

- 与实验一致的仓库 commit；
- 相同版本的 `collect_teacher_rollouts.py`、`edit.py`、`docker.py` 和 OpenHands function-calling YAML；
- 输入 Parquet 文件与 `manifest.json` 指纹一致；
- Docker Hub 上对应镜像仍可获取；
- 教师 API 的模型 ID、协议和服务端实现未发生不兼容变化。

API 服务是外部依赖。即使参数、任务和代码完全相同，采样式模型及服务端更新也可能使生成文本不逐 token 一致；本流程保证任务选择、运行配置、环境和产物结构可追踪，而不是保证随机生成内容逐字节相同。
