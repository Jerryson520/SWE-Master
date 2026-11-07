#!/usr/bin/env python3
"""Collect resumable R2E-Gym teacher rollouts for SWE-smith.

By default the agent and containers use the local Docker Unix socket. Valid
(instance_id, rollout_index) results are verified and skipped. This script
never pulls or deletes images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
from urllib.parse import urlsplit
import uuid


DATASET = "SWE-bench/SWE-smith"
REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"
ACR_REPOSITORY = (
    "crpi-jme0bqk6txpdw1cg-vpc.cn-hangzhou.personal.cr.aliyuncs.com/"
    "swe-master-personal/swe-smith"
)


def atomic_write(path: Path, value, jsonl: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in value:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def acr_image(source: str) -> str:
    leaf = source.rsplit("/", 1)[-1]
    if ":" in leaf:
        project, tag = leaf.rsplit(":", 1)
    else:
        project, tag = leaf, "latest"
    return f"{ACR_REPOSITORY}:{project}-{tag}"


def stable_key(task: dict, seed: int):
    digest = hashlib.sha256(f"{seed}\0{task['instance_id']}".encode()).digest()
    return digest, task["instance_id"]


def select_tasks(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"task-count={count} exceeds dataset size={len(rows)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["repo"]].append(row)
    exact = {repo: len(group) * count / len(rows) for repo, group in groups.items()}
    quotas = {repo: math.floor(value) for repo, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda repo: (-(exact[repo] - quotas[repo]), repo))
    for repo in order[:remaining]:
        quotas[repo] += 1
    selected = []
    for repo, group in groups.items():
        selected.extend(
            sorted(group, key=lambda item: stable_key(item, seed))[: quotas[repo]]
        )
    selected.sort(key=lambda item: stable_key(item, seed))
    if len(selected) != count:
        raise RuntimeError("stratified selection produced an incorrect task count")
    return selected


def normalize_row(row: dict) -> dict:
    required = (
        "instance_id",
        "repo",
        "image_name",
        "problem_statement",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"SWE-smith row is missing fields: {missing}")
    strings = ("instance_id", "repo", "image_name", "problem_statement")
    if not all(isinstance(row[key], str) and row[key] for key in strings):
        raise ValueError("SWE-smith row contains an empty required string")
    source_image = row["image_name"]
    value = dict(row)
    value.update(
        {
            "dataset_name": "swe-smith",
            "dataset_revision": REVISION,
            "source_image_name": source_image,
            # DockerRuntime prefers docker_image. Keep image_name unchanged for
            # SWE-smith registry/profile helpers.
            "docker_image": acr_image(source_image),
        }
    )
    return value


def load_or_create_tasks(out: Path, count: int, seed: int) -> list[dict]:
    path = out / "selected-tasks.jsonl"
    if path.exists():
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(tasks) != count:
            raise ValueError("existing task count differs; use a new output directory")
        if any(task.get("dataset_revision") != REVISION for task in tasks):
            raise ValueError("existing task list uses a different dataset revision")
        return tasks

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    # Download only the pinned Parquet shards, then scan them in small record
    # batches. The high-level datasets builder materializes a large Arrow file
    # and can be killed on memory-constrained orchestration containers.
    snapshot = Path(
        snapshot_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["data/*.parquet"],
        )
    )
    shards = sorted(snapshot.glob("data/*.parquet"))
    if not shards:
        raise RuntimeError("the pinned SWE-smith snapshot contains no Parquet shards")

    # Compute repository quotas in one pass and retain only the selected rows
    # in a second pass.
    counts: Counter[str] = Counter()
    total = 0
    index_columns = ["instance_id", "repo", "image_name", "problem_statement"]
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=4096, columns=index_columns
        ):
            for row in batch.to_pylist():
                if not all(
                    isinstance(row[key], str) and row[key] for key in index_columns
                ):
                    continue
                counts[row["repo"]] += 1
                total += 1
    if count > total:
        raise ValueError(f"task-count={count} exceeds dataset size={total}")
    exact = {repo: size * count / total for repo, size in counts.items()}
    quotas = {repo: math.floor(value) for repo, value in exact.items()}
    remaining = count - sum(quotas.values())
    for repo in sorted(
        counts, key=lambda name: (-(exact[name] - quotas[name]), name)
    )[:remaining]:
        quotas[repo] += 1

    heaps: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
            for raw in batch.to_pylist():
                if not all(
                    isinstance(raw[key], str) and raw[key] for key in index_columns
                ):
                    continue
                repo = raw["repo"]
                quota = quotas[repo]
                if quota == 0:
                    continue
                row = normalize_row(raw)
                rank = int.from_bytes(stable_key(row, seed)[0], "big")
                entry = (-rank, row["instance_id"], row)
                heap = heaps[repo]
                if len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    tasks = [entry[2] for heap in heaps.values() for entry in heap]
    tasks.sort(key=lambda item: stable_key(item, seed))
    if len(tasks) != count:
        raise RuntimeError("streaming selection produced an incorrect task count")
    atomic_write(path, tasks, jsonl=True)
    return tasks


def result_path(out: Path, task: dict, rollout_index: int) -> Path:
    # Match the established R2E-Gym dataset layout: one directory per task,
    # with rollout-0/1/2.json directly below it.
    return out / task["instance_id"] / f"rollout-{rollout_index}.json"


def attempt_name(run_id: str, task: dict, rollout_index: int) -> str:
    return f"teacher-{run_id}/{task['instance_id']}/rollout-{rollout_index}"


def validate_result(
    value: dict, task: dict, rollout_index: int, run_id: str
) -> dict:
    expected_name = attempt_name(run_id, task, rollout_index)
    if not isinstance(value, dict) or value.get("reward") not in (0, 1, 0.0, 1.0):
        raise ValueError("result has no binary reward")
    if value.get("ds", {}).get("instance_id") != task["instance_id"]:
        raise ValueError("result instance_id does not match its slot")
    if value.get("exp_name") != expected_name:
        raise ValueError("result rollout identity does not match its slot")
    if not isinstance(value.get("trajectory_steps"), list) or not value["trajectory_steps"]:
        raise ValueError("result trajectory_steps is empty")
    if not isinstance(value.get("test_output"), str):
        raise ValueError("result test_output is missing")
    return value


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_attempt(
    task: dict, rollout_index: int, out: Path, config: dict, run_id: str
) -> None:
    from r2egym.agenthub.run.edit import configure_loopback_llm_no_proxy, runagent

    configure_loopback_llm_no_proxy()
    exp_name = attempt_name(run_id, task, rollout_index)
    try:
        raw = runagent(
            ds=task,
            exp_name=exp_name,
            traj_dir=str(out / "raw"),
            **config["agent"],
        )
        if raw is None:
            raise RuntimeError("runagent returned no trajectory")
        value = validate_result(json.loads(raw), task, rollout_index, run_id)
        atomic_write(result_path(out, task, rollout_index), value)
    except BaseException as exc:
        atomic_write(
            out / "errors" / task["instance_id"] / f"rollout-{rollout_index}.json",
            {
                "instance_id": task["instance_id"],
                "rollout_index": rollout_index,
                "error_type": type(exc).__name__,
                "status": "stopped",
            },
        )
        raise


def verify_remote_images(tasks: list[dict], docker_ip: str) -> dict:
    import docker

    client = (
        docker.DockerClient(base_url=f"tcp://{docker_ip}:2375", timeout=120)
        if docker_ip
        else docker.from_env(timeout=120)
    )
    try:
        info = client.info()
        images = sorted({task["docker_image"] for task in tasks})
        missing = []
        for image in images:
            try:
                client.images.get(image)
            except docker.errors.ImageNotFound:
                missing.append(image)
        if missing:
            raise RuntimeError(
                f"remote Docker is missing {len(missing)} required images; first: {missing[0]}"
            )
        return {
            "docker_id": info.get("ID"),
            "docker_root": info.get("DockerRootDir"),
            "images": len(images),
        }
    finally:
        client.close()


def collect(
    tasks: list[dict], out: Path, config: dict, run_id: str, workers: int
) -> None:
    slots = []
    completed = 0
    for task_number, task in enumerate(tasks, 1):
        for rollout_index in range(config["rollouts"]):
            path = result_path(out, task, rollout_index)
            if path.exists():
                validate_result(
                    json.loads(path.read_text()), task, rollout_index, run_id
                )
                completed += 1
            else:
                slots.append((task_number, task, rollout_index))
    total = len(tasks) * config["rollouts"]
    print(f"resume: valid={completed}, pending={len(slots)}, total={total}", flush=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        pending = {}
        cursor = 0
        while cursor < len(slots) or pending:
            while cursor < len(slots) and len(pending) < workers:
                task_number, task, rollout_index = slots[cursor]
                print(
                    f"start task={task_number}/{len(tasks)} "
                    f"instance={task['instance_id']} "
                    f"rollout={rollout_index + 1}/{config['rollouts']}",
                    flush=True,
                )
                future = pool.submit(
                    run_attempt, task, rollout_index, out, config, run_id
                )
                pending[future] = (task, rollout_index)
                cursor += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            for future in done:
                task, rollout_index = pending.pop(future)
                completed += 1
                print(
                    f"saved {completed}/{total} {task['instance_id']} "
                    f"rollout={rollout_index + 1}",
                    flush=True,
                )

    values = []
    for task in tasks:
        for rollout_index in range(config["rollouts"]):
            values.append(
                validate_result(
                    json.loads(result_path(out, task, rollout_index).read_text()),
                    task,
                    rollout_index,
                    run_id,
                )
            )
    atomic_write(out / "trajectories.jsonl", values, jsonl=True)
    atomic_write(
        out / "summary.json",
        {
            "tasks": len(tasks),
            "rollouts_per_task": config["rollouts"],
            "trajectories": len(values),
            "reward_1": sum(value["reward"] == 1 for value in values),
            "reward_0": sum(value["reward"] == 0 for value in values),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=2000)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--docker-ip",
        default="",
        help="optional remote Docker host; empty uses the local Unix socket",
    )
    parser.add_argument("--llm-name", default="openai/deepseek-flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.task_count, args.rollouts, args.max_workers) < 1:
        parser.error("counts and max-workers must be positive")
    if args.selection_seed < 0:
        parser.error("selection-seed must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks = load_or_create_tasks(out, args.task_count, args.selection_seed)
        repo_counts = dict(sorted(Counter(task["repo"] for task in tasks).items()))
        image_count = len({task["docker_image"] for task in tasks})
        print(
            f"tasks={len(tasks)} rollouts={args.rollouts} "
            f"total={len(tasks) * args.rollouts} workers={args.max_workers}"
        )
        print(
            f"dataset={DATASET}@{REVISION} repos={len(repo_counts)} "
            f"images={image_count}"
        )
        if args.dry_run:
            return

        endpoint = os.environ.get("OPENAI_API_BASE", "")
        url = urlsplit(endpoint)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "OPENAI_API_BASE must be a credential-free HTTP(S) URL"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required")

        remote = verify_remote_images(tasks, args.docker_ip)
        config = {
            "rollouts": args.rollouts,
            "agent": {
                "llm_name": args.llm_name,
                "temperature": 1.0,
                "max_steps": 150,
                "max_steps_absolute": 150,
                "backend": "docker",
                "scaffold": "openhands",
                "use_fn_calling": True,
                "use_lsp": False,
                "enable_compression": False,
                "max_reward_calc_time": 1800,
                "context_window": 131072,
                "max_output_tokens": 8192,
                "context_safety_margin": 4096,
                "max_trajectory_output_tokens": 131072,
                "ip": args.docker_ip,
                "used_yaml": (
                    "./src/r2egym/agenthub/config/openhands/"
                    "openhands_sp_fn_calling.yaml"
                ),
            },
        }
        identity = {
            "dataset": DATASET,
            "revision": REVISION,
            "selection_seed": args.selection_seed,
            "task_count": len(tasks),
            "repo_counts": repo_counts,
            "tasks_sha256": hashlib.sha256(
                (out / "selected-tasks.jsonl").read_bytes()
            ).hexdigest(),
            "config": config,
            "remote_docker": remote,
        }
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    "dataset/config/remote Docker changed; use a new output directory"
                )
        else:
            # ``tee -a <output>/run.log`` opens the log before Python reaches
            # this check, so the collector's own fixed log file is safe here.
            allowed = {".lock", "selected-tasks.jsonl", "run.log"}
            unexpected = [item.name for item in out.iterdir() if item.name not in allowed]
            if unexpected:
                raise ValueError(
                    "output directory contains unrelated files and has no manifest"
                )
            manifest = dict(identity, run_id=uuid.uuid4().hex[:12])
            atomic_write(manifest_path, manifest)
        atomic_write(out / "execution.json", {"max_workers": args.max_workers})
        os.chdir(Path(__file__).resolve().parents[2])
        collect(tasks, out, config, manifest["run_id"], args.max_workers)


if __name__ == "__main__":
    main()
