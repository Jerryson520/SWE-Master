#!/usr/bin/env python3
"""按批采集 R2E teacher 轨迹；默认两个独立进程，不启动模型服务。

在已激活的 R2E-Gym 环境运行 --help。API key 只从环境读取。
结果文件是恢复依据；发生异常即停止，修复后用相同参数重启。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from contextlib import closing
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time


def atomic_write(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if jsonl:
                for row in value:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _stable_task_key(task, seed):
    """返回不依赖 Parquet 行顺序的确定性任务排序键。"""
    value = f"{seed}\0{task['instance_id']}".encode()
    return hashlib.sha256(value).digest(), task["instance_id"]


def select_tasks_stratified(tasks, count=None, seed=42):
    """按仓库规模等比例分层抽样，并用稳定哈希固定每次选择结果。"""
    if count is None:
        count = len(tasks)
    if count > len(tasks):
        raise ValueError("task-count 超过任务总数")

    groups = defaultdict(list)
    for task in tasks:
        groups[task["repo_name"]].append(task)

    # 最大余数法：先取各仓库比例配额的整数部分，再按小数余数补齐。
    total = len(tasks)
    exact = {repo: len(group) * count / total for repo, group in groups.items()}
    quotas = {repo: math.floor(value) for repo, value in exact.items()}
    remaining = count - sum(quotas.values())
    remainder_order = sorted(
        groups,
        key=lambda repo: (-(exact[repo] - quotas[repo]), repo),
    )
    for repo in remainder_order[:remaining]:
        quotas[repo] += 1

    selected = []
    for repo, group in groups.items():
        ranked = sorted(group, key=lambda task: _stable_task_key(task, seed))
        selected.extend(ranked[:quotas[repo]])

    # 混合不同仓库，避免每批镜像和任务都集中在单一仓库。
    selected.sort(key=lambda task: _stable_task_key(task, seed))
    if len(selected) != count:
        raise RuntimeError("分层抽样数量与 task-count 不一致")
    return selected


def load_tasks(source, count=None, selection_seed=42):
    import pyarrow.parquet as pq

    source = Path(source)
    files = sorted((source / "data").glob("train-*.parquet"))
    if not files:
        raise ValueError("source-dir/data 中没有 train-*.parquet")
    tasks, fingerprints, ids = [], {}, set()
    required = ("docker_image", "repo_name", "commit_hash", "problem_statement",
                "parsed_commit_content", "expected_output_json")
    for path in files:
        with path.open("rb") as f:
            fingerprints[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
        if not set(required).issubset(pq.read_schema(path).names):
            raise ValueError(f"分片缺少必要字段: {path.name}")
        # 仅常驻任务索引；完整 commit/test 数据按当前任务读取。
        for index, row in enumerate(pq.read_table(path, columns=["repo_name", "commit_hash", "docker_image"]).to_pylist()):
            if any(not isinstance(row.get(k), str) or not row[k]
                   for k in ("repo_name", "commit_hash", "docker_image")):
                raise ValueError("任务索引缺少必要字符串字段")
            row["instance_id"] = f"r2egym__{row['repo_name']}__{row['commit_hash']}"
            ident = row["instance_id"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", ident) or ident in ids:
                raise ValueError(f"任务 ID 不安全或重复: {ident}")
            ids.add(ident)
            row["_source_file"] = str(path.resolve())
            row["_source_row"] = index
            tasks.append(row)
    return select_tasks_stratified(tasks, count, selection_seed), fingerprints


class TaskReader:
    """只缓存当前 Parquet row group，向 runagent 传递完整原始字段。"""
    def __init__(self):
        self.key, self.table = None, None

    def load(self, task):
        import pyarrow.parquet as pq
        source = pq.ParquetFile(task["_source_file"])
        index = task["_source_row"]
        for group in range(source.num_row_groups):
            size = source.metadata.row_group(group).num_rows
            if index >= size:
                index -= size
                continue
            key = (task["_source_file"], group)
            if self.key != key:
                self.table = None
                self.table = source.read_row_group(group)
                self.key = key
            row = self.table.slice(index, 1).to_pylist()[0]
            row["instance_id"] = task["instance_id"]
            if row["docker_image"] != task["docker_image"] or not row["problem_statement"]:
                raise ValueError("任务数据与索引不匹配")
            json.loads(row["parsed_commit_content"])
            expected = json.loads(row["expected_output_json"])
            if not isinstance(expected, dict) or not expected:
                raise ValueError("任务缺少非空测试预期")
            return row
        raise ValueError("Parquet 行号超出范围")


def result_path(out, task, index):
    return out / task["instance_id"] / f"rollout-{index}.json"


def validate_result(value, task, exp_name):
    if not isinstance(value, dict) or value.get("reward") not in (0, 1):
        raise ValueError("没有有效的二元 reward")
    if value.get("ds", {}).get("instance_id") != task["instance_id"]:
        raise ValueError("trajectory 任务不匹配")
    if value.get("exp_name") != exp_name:
        raise ValueError("trajectory rollout 标识不匹配")
    if not isinstance(value.get("trajectory_steps"), list) or not value["trajectory_steps"]:
        raise ValueError("trajectory_steps 缺失或为空")
    if not isinstance(value.get("test_output"), str):
        raise ValueError("test_output 缺失")
    return value


def attempt_name(run_id, task, index):
    return f"teacher-{run_id}/{task['instance_id']}/rollout-{index}"


class InsufficientSpace(RuntimeError):
    """等待在途任务结束并清理后，可以重新检查空间。"""


class Images:
    """清理本批任务镜像（含预下载）；保留被容器引用或多标签的镜像。"""
    def __init__(self, client, root, out, manifest, min_free_gb):
        self.client, self.root, self.out = client, root, out
        self.minimum = int(min_free_gb * 1024**3)
        self.baseline = manifest["initial_images"]
        self.ledger_path = out / "pulled_images.json"
        self.owned = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}

    def free(self):
        return shutil.disk_usage(self.root).free

    def get(self, tag):
        import docker
        try:
            return self.client.images.get(tag)
        except docker.errors.ImageNotFound:
            return None

    def pull_with_retry(self, tag, retry_delays=(5, 15)):
        """有限重试镜像拉取；不跳过任务，也不把基础设施失败记为 reward=0。"""
        import docker

        attempts = len(retry_delays) + 1
        for attempt in range(1, attempts + 1):
            # 上一次请求可能已下载成功，但 docker-py 在最终 get() 时短暂失败。
            image = self.get(tag)
            if image is not None:
                return image
            try:
                self.client.images.pull(tag)
                image = self.get(tag)
                if image is None:
                    raise docker.errors.ImageNotFound(
                        f"拉取结束但镜像未注册: {tag}"
                    )
                return image
            except docker.errors.DockerException as exc:
                # 再检查一次，避免将“已落盘但 SDK 返回异常”误判为失败。
                image = self.get(tag)
                if image is not None:
                    return image
                status = getattr(exc, "status_code", None)
                # 参数、鉴权和权限错误不会通过等待恢复，应立即暴露。
                if status in (400, 401, 403) or attempt == attempts:
                    raise
                delay = retry_delays[attempt - 1]
                print(
                    f"镜像拉取失败，{delay} 秒后重试 "
                    f"{attempt + 1}/{attempts}: {tag} ({type(exc).__name__})",
                    flush=True,
                )
                time.sleep(delay)

        raise RuntimeError(f"无法拉取镜像: {tag}")

    def pull_with_fallback(self, tag):
        """先使用 daemon mirror；失败后显式绕过 mirror，经代理访问 Docker Hub。"""
        import docker

        try:
            # 原始 docker.io 名称会使用 daemon.json 中的 registry-mirrors。
            return self.pull_with_retry(tag, retry_delays=())
        except docker.errors.DockerException as mirror_error:
            mirror_failure = mirror_error
            image = self.get(tag)
            if image is not None:
                return image
            print(f"镜像加速器未命中，切换 Docker Hub 原站: {tag}", flush=True)

        slash = tag.rfind("/")
        colon = tag.rfind(":")
        if colon <= slash or not tag[colon + 1:]:
            raise ValueError(f"镜像必须包含明确 tag: {tag}") from mirror_failure
        repository, tag_name = tag[:colon], tag[colon + 1:]
        direct_tag = f"registry-1.docker.io/{tag}"

        # 显式 registry-1.docker.io 主机名不会套用 docker.io mirror；
        # dockerd 的 HTTPS_PROXY 会让这条路径通过 127.0.0.1:1082。
        image = self.pull_with_retry(direct_tag)
        if not image.tag(repository, tag=tag_name):
            raise RuntimeError(f"Docker Hub 镜像无法标记为原始名称: {tag}")
        original = self.get(tag)
        if original is None:
            raise docker.errors.ImageNotFound(f"回源成功但原始标签未注册: {tag}")
        try:
            # 避免临时 registry-1 标签让后续清理误判为多标签镜像。
            self.client.images.remove(direct_tag, force=False, noprune=True)
        except docker.errors.DockerException as exc:
            print(f"临时回源标签未删除: {tag} ({type(exc).__name__})", flush=True)
        return original

    def prepare(self, tasks):
        ready = []
        # 用最近镜像大小作为保守提示，不承诺精确预测压缩层展开空间。
        estimate = 2 * 1024**3
        for task in tasks:
            if self.free() < self.minimum:
                break
            tag = task["docker_image"]
            if self.get(tag) is None:
                if ready and self.free() < self.minimum + estimate:
                    break
                print(f"拉取 {tag}", flush=True)
                image = self.pull_with_fallback(tag)
                self.owned[tag] = image.id
                atomic_write(self.ledger_path, self.owned)
                size = image.attrs.get("Size", 0)
                if isinstance(size, int):
                    estimate = max(estimate, size)
            ready.append(task)
            # 该题已拉取，但低于安全余量时不能启动测试；暂停供用户处理。
            if self.free() < self.minimum:
                raise InsufficientSpace("拉取后空间低于安全余量；保留镜像和状态，降低批次或释放空间后重启")
        if not ready:
            raise InsufficientSpace("空间不足，无法安全准备一个任务")
        return ready

    def cleanup(self, tasks):
        """结果落盘后，仅按本批任务的准确标签清理，不执行全局 prune。"""
        for tag in sorted({x["docker_image"] for x in tasks}):
            image = self.get(tag)
            if image is None:
                continue
            expected_id = self.owned.get(tag)
            if expected_id is not None and image.id != expected_id:
                print(f"保留标签指向已变化的镜像 {tag}", flush=True)
                continue
            # 存在其他标签时保守保留，即使其中一些标签也属于本轮。
            if any(t != tag for t in image.tags):
                print(f"保留多标签镜像 {tag}", flush=True)
                continue
            used = any(c.attrs.get("Image") == image.id
                       for c in self.client.containers.list(all=True))
            if used:
                print(f"保留被容器引用的镜像 {tag}", flush=True)
                continue
            try:
                self.client.images.remove(tag, force=False, noprune=True)
                print(f"已删除本批任务镜像 {tag}", flush=True)
            except Exception as exc:
                print(f"镜像未删除 {tag}: {type(exc).__name__}", flush=True)


def init_rollout_worker():
    # Ctrl-C 由主进程处理：停止派发，等待已派发轨迹保存并关闭环境。
    signal.signal(signal.SIGINT, signal.SIG_IGN)


_task_reader = None


def run_full_task(ds, **kwargs):
    """spawn 子进程内加载数据和 Agent，隔离日志、客户端与可变状态。"""
    global _task_reader
    from r2egym.agenthub.run.edit import runagent, configure_loopback_llm_no_proxy
    configure_loopback_llm_no_proxy()
    if _task_reader is None:
        _task_reader = TaskReader()
    return runagent(ds=_task_reader.load(ds), **kwargs)


def save_attempt(task, index, out, config, run, run_id):
    """每个槽位仅派发一次；子进程完成时立即落盘，不等待整批。"""
    exp = attempt_name(run_id, task, index)
    try:
        raw = run(ds=task, exp_name=exp, traj_dir=str(out), **config["agent"])
        if raw is None:
            raise RuntimeError("runagent 返回 None，详见单任务日志")
        value = validate_result(json.loads(raw), task, exp)
        atomic_write(result_path(out, task, index), value)
    except BaseException as exc:
        # 不保存异常文本，避免第三方 HTTP 错误中携带鉴权信息。
        atomic_write(out / "errors" / task["instance_id"] / f"rollout-{index}.json",
                     {"instance_id": task["instance_id"], "rollout_index": index,
                      "error_type": type(exc).__name__, "status": "stopped"})
        raise


def collect(tasks, out, config, images, run, run_id, max_workers=2, executor_factory=None):
    """有界并发：主进程管理镜像，按任务完成情况清理，不等待整批。"""
    if max_workers < 1:
        raise ValueError("max_workers 必须为正数")
    if executor_factory is None:
        executor_factory = lambda: ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_rollout_worker)
    cursor, batch_index = 0, 0
    while cursor < len(tasks):
        batch_file = out / "batches" / f"batch-{batch_index:04d}.json"
        if batch_file.exists():
            batch_ids = json.loads(batch_file.read_text())
            batch = tasks[cursor:cursor + len(batch_ids)]
            if not batch_ids or [t["instance_id"] for t in batch] != batch_ids:
                raise ValueError("批次清单不匹配")
        else:
            # 此处只固定本批任务清单，不整批预拉镜像。下面向 worker
            # 派发每个 rollout 前会执行 images.prepare([task])：第一个
            # 镜像就绪后即可开始推理，后续拉取可与在途 rollout 重叠。
            batch = tasks[cursor:cursor + config["batch_size"]]
            atomic_write(batch_file, [t["instance_id"] for t in batch])

        slots = []
        completed = {t["instance_id"]: 0 for t in batch}
        cleaned_tags = set()

        def cleanup_completed():
            # 同镜像可能被本批多题共享：必须全部完成，且无在途 worker。
            # completed 只计已验证的旧结果或 future 成功返回后的新结果。
            for tag in dict.fromkeys(t["docker_image"] for t in batch):
                shared = [t for t in batch if t["docker_image"] == tag]
                if tag not in cleaned_tags and all(
                    completed[t["instance_id"]] == config["rollouts"] for t in shared
                ):
                    images.cleanup(shared)
                    cleaned_tags.add(tag)

        for offset, task in enumerate(batch):
            for index in range(config["rollouts"]):
                path = result_path(out, task, index)
                exp = attempt_name(run_id, task, index)
                if path.exists():
                    validate_result(json.loads(path.read_text()), task, exp)
                    completed[task["instance_id"]] += 1
                    continue
                slots.append((cursor + offset + 1, task, index))
        # 恢复时先释放已完成题目的镜像，再检查未完成题目的启动空间。
        cleanup_completed()
        if slots:
            # 队列中最多 max_workers 个任务；观察到失败后不再派发新任务。
            # with 退出会等待在途任务落盘并执行 runagent 的 finally。
            with executor_factory() as pool:
                pending, next_slot = {}, 0
                while next_slot < len(slots) or pending:
                    while next_slot < len(slots) and len(pending) < max_workers:
                        if any(f.done() and f.exception() is not None for f in pending):
                            break
                        number, task, index = slots[next_slot]
                        try:
                            images.prepare([task])  # 仅主进程拉取、记录镜像并检查空间
                        except InsufficientSpace:
                            if not pending:
                                raise
                            print("空间不足，暂停派发，等待在途任务完成并清理镜像", flush=True)
                            break
                        # 拉取期间 worker 也可能失败；此时不能继续派发。
                        for f in pending:
                            if f.done():
                                f.result()
                        print(f"任务 {number}/{len(tasks)} {task['instance_id']} "
                              f"rollout={index + 1}/{config['rollouts']} "
                              f"max_workers={max_workers}", flush=True)
                        f = pool.submit(save_attempt, task, index, out, config, run, run_id)
                        pending[f] = (task, index)
                        next_slot += 1
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # 先检查这一轮所有完成项，不能在检查到失败前补充新任务。
                    for f in done:
                        f.result()
                    for f in done:
                        task, index = pending.pop(f)
                        completed[task["instance_id"]] += 1
                        print(f"已保存 {task['instance_id']} rollout={index + 1}", flush=True)
                    cleanup_completed()
        # 汇总按固定任务/rollout 顺序输出，不受完成先后影响。
        values = [validate_result(json.loads(result_path(out, t, i).read_text()),
                                  t, attempt_name(run_id, t, i))
                  for t in batch for i in range(config["rollouts"])]
        atomic_write(batch_file.with_suffix(".jsonl"), values, jsonl=True)
        print(f"批次 {batch_index}: {len(values)} 条，reward=1: "
              f"{sum(v['reward'] == 1 for v in values)}", flush=True)
        cursor += len(batch)
        batch_index += 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--max-workers", type=int, default=2,
                   help="同时执行的轨迹数，默认 2；可设 1 恢复串行，重启时可调整")
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--task-count", type=int)
    p.add_argument("--selection-seed", type=int, default=42,
                   help="分层抽样的稳定哈希 seed，默认 42；恢复时不可修改")
    p.add_argument("--min-free-gb", type=float, default=10)
    p.add_argument("--llm-name", required=True, help="openai/实际接口模型ID")
    p.add_argument("--context-window", type=int, required=True)
    p.add_argument("--max-output-tokens", type=int, required=True)
    p.add_argument("--context-safety-margin", type=int, required=True)
    p.add_argument("--max-trajectory-output-tokens", type=int, required=True)
    p.add_argument("--dry-run", action="store_true", help="只读检查，无 Docker/模型调用")
    a = p.parse_args(argv)
    positive = (a.batch_size, a.max_workers, a.rollouts, a.context_window, a.max_output_tokens,
                a.max_trajectory_output_tokens, a.task_count if a.task_count is not None else 1)
    if min(positive) < 1 or a.selection_seed < 0 or not math.isfinite(a.min_free_gb) or a.min_free_gb <= 0:
        p.error("数量、预算及磁盘余量必须为正数")
    if not 0 <= a.context_safety_margin < a.context_window:
        p.error("context-safety-margin 必须小于 context-window")
    if a.max_output_tokens > a.context_window - a.context_safety_margin:
        p.error("单次输出不能超过上下文减安全余量")
    if not a.llm_name.startswith("openai/"):
        p.error("教师兼容接口模型名应使用 openai/ 前缀")
    return a


def main(argv=None):
    a = parse_args(argv)
    tasks, fingerprints = load_tasks(a.source_dir, a.task_count, a.selection_seed)
    repo_counts = dict(sorted(Counter(task["repo_name"] for task in tasks).items()))
    print(f"任务 {len(tasks)}，每题 {a.rollouts} 次，总槽位 {len(tasks) * a.rollouts}，并发 {a.max_workers}")
    print(f"任务选择：按仓库比例分层，seed={a.selection_seed}")
    print("仓库分布：" + ", ".join(f"{repo}={count}" for repo, count in repo_counts.items()))
    if a.dry_run:
        print("dry-run：未写文件、未拉镜像、未调用模型")
        return
    from urllib.parse import urlsplit
    endpoint = os.environ.get("OPENAI_API_BASE", "")
    url = urlsplit(endpoint)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("OPENAI_API_BASE 必须是不含凭据或查询参数的 HTTP(S) 地址")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("请先设置 OPENAI_API_KEY；无鉴权服务可显式使用 EMPTY")
    if os.environ.get("DOCKER_HOST") and not os.environ["DOCKER_HOST"].startswith("unix://"):
        raise ValueError("本脚本仅支持本机 Docker，避免误判远程 daemon 的磁盘空间")

    import docker
    from r2egym.agenthub.run.edit import runagent, configure_loopback_llm_no_proxy

    config = {"batch_size": a.batch_size, "rollouts": a.rollouts,
              "min_free_gb": a.min_free_gb, "endpoint": endpoint,
              "agent": {"llm_name": a.llm_name, "temperature": 1.0,
                        "max_steps": 150, "max_steps_absolute": 150,
                        "backend": "docker", "scaffold": "openhands",
                        # Teacher 采集固定使用正式 API 的标准 tools/tool_calls 协议。
                        "use_fn_calling": True, "use_lsp": False,
                        "enable_compression": False, "max_reward_calc_time": 1800,
                        "context_window": a.context_window,
                        "max_output_tokens": a.max_output_tokens,
                        "context_safety_margin": a.context_safety_margin,
                        "max_trajectory_output_tokens": a.max_trajectory_output_tokens,
                        "used_yaml": "./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml"}}
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # DockerClient 有 close()，但当前 SDK 没有 __enter__/__exit__。
        with closing(docker.from_env(timeout=120)) as client:
            info = client.info()
            root = Path(info["DockerRootDir"])
            if not root.is_dir():
                raise ValueError("无法访问 Docker data-root，不能安全检查磁盘")
            identity = {"fingerprints": fingerprints, "config": config,
                        "selection": {"method": "proportional_repo_stable_hash_v1",
                                      "seed": a.selection_seed,
                                      "task_count": len(tasks),
                                      "repo_counts": repo_counts},
                        "tasks": [{"instance_id": t["instance_id"], "docker_image": t["docker_image"]} for t in tasks],
                        "docker_id": info.get("ID"), "docker_root": str(root)}
            manifest_file = out / "manifest.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text())
                if any(manifest.get(k) != v for k, v in identity.items()):
                    raise ValueError("数据、配置或 Docker daemon 已变化；请使用新输出目录")
            else:
                if any(x.name != ".lock" for x in out.iterdir()):
                    raise ValueError("输出目录非空但缺少 manifest，不覆盖已有内容")
                import uuid
                manifest = dict(identity, run_id=uuid.uuid4().hex[:12],
                                initial_images=[{"id": i.id, "tags": i.tags} for i in client.images.list(all=True)])
                atomic_write(manifest_file, manifest)
            os.chdir(Path(__file__).resolve().parents[1])
            configure_loopback_llm_no_proxy()
            # 并发属于执行设置，不改变任务/模型预算；兼容旧 manifest 和已存结果。
            atomic_write(out / "execution.json", {"max_workers": a.max_workers})
            collect(tasks, out, config, Images(client, root, out, manifest, a.min_free_gb),
                    run_full_task, manifest["run_id"], max_workers=a.max_workers)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已中断；保留结果，使用相同参数恢复。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        message = str(exc)
        for secret in (os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_API_BASE")):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        message = re.sub(r"https?://\S+", "[URL]", message)
        print(f"采集停止: {type(exc).__name__}: {message}。检查错误记录及单任务日志；保留未完成任务镜像。", file=sys.stderr)
        sys.exit(1)
