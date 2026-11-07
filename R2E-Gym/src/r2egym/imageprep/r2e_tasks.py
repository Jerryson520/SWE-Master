"""Deterministic task loading for R2E-Gym-Subset."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from r2egym.commit_models.diff_classes import ParsedCommit


REQUIRED_COLUMNS = (
    "docker_image",
    "repo_name",
    "commit_hash",
    "problem_statement",
    "parsed_commit_content",
    "expected_output_json",
)


def _stable_task_key(task: dict[str, Any], seed: int) -> tuple[bytes, str]:
    value = f"{seed}\0{task['instance_id']}".encode()
    return hashlib.sha256(value).digest(), str(task["instance_id"])


def select_tasks_stratified(
    tasks: list[dict[str, Any]], count: int | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    if count is None:
        count = len(tasks)
    if count < 1 or count > len(tasks):
        raise ValueError("task-count must be between 1 and the dataset size")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[str(task["repo_name"])].append(task)
    exact = {repo: len(group) * count / len(tasks) for repo, group in groups.items()}
    quotas = {repo: math.floor(value) for repo, value in exact.items()}
    remaining = count - sum(quotas.values())
    for repo in sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))[:remaining]:
        quotas[repo] += 1
    selected: list[dict[str, Any]] = []
    for repo, group in groups.items():
        selected.extend(sorted(group, key=lambda task: _stable_task_key(task, seed))[: quotas[repo]])
    selected.sort(key=lambda task: _stable_task_key(task, seed))
    if len(selected) != count:
        raise RuntimeError("stratified selection returned the wrong number of tasks")
    return selected


def _validate_full_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads(row["expected_output_json"])
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"task {row['instance_id']} has no expected tests")
    commit = ParsedCommit(**json.loads(row["parsed_commit_content"]))
    if commit.new_commit_hash != row["commit_hash"]:
        raise ValueError(f"task {row['instance_id']} commit hash mismatch")
    return row


def load_r2e_tasks(
    source: str | Path,
    count: int | None = None,
    selection_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import pyarrow.parquet as pq

    source = Path(source)
    files = sorted((source / "data").glob("train-*.parquet"))
    if not files:
        raise ValueError("source-dir/data contains no train-*.parquet files")
    indexes: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    instance_ids: set[str] = set()
    for path in files:
        with path.open("rb") as stream:
            fingerprints[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        schema = set(pq.read_schema(path).names)
        if not set(REQUIRED_COLUMNS).issubset(schema):
            raise ValueError(f"parquet shard is missing required columns: {path.name}")
        table = pq.read_table(path, columns=["repo_name", "commit_hash", "docker_image"])
        for index, row in enumerate(table.to_pylist()):
            if any(not isinstance(row.get(key), str) or not row[key] for key in row):
                raise ValueError(f"invalid task index in {path.name}:{index}")
            row["instance_id"] = f"r2egym__{row['repo_name']}__{row['commit_hash']}"
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["instance_id"]):
                raise ValueError(f"unsafe instance id: {row['instance_id']}")
            if row["instance_id"] in instance_ids:
                raise ValueError(f"duplicate instance id: {row['instance_id']}")
            instance_ids.add(row["instance_id"])
            row["_source_file"] = str(path.resolve())
            row["_source_row"] = index
            indexes.append(row)
    selected = select_tasks_stratified(indexes, count, selection_seed)
    full_rows: list[dict[str, Any]] = []
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in selected:
        by_file[task["_source_file"]].append(task)
    for file_name, tasks in by_file.items():
        table = pq.read_table(file_name)
        for task in tasks:
            row = table.slice(task["_source_row"], 1).to_pylist()[0]
            row["instance_id"] = task["instance_id"]
            full_rows.append(_validate_full_row(row))
    full_rows.sort(key=lambda task: _stable_task_key(task, selection_seed))
    if len({task["docker_image"] for task in full_rows}) != len(full_rows):
        raise ValueError("R2E prepared images require one unique source image per task")
    return full_rows, fingerprints


def r2e_source_mirror_reference(docker_image: str, source_registry: str) -> str:
    repository, commit_hash = docker_image.rsplit(":", 1)
    source_tag = repository.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_tag):
        raise ValueError(f"unsafe source repository tag: {source_tag}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit_hash):
        raise ValueError(f"invalid R2E commit hash: {commit_hash}")
    return f"{source_registry.rstrip('/')}:{source_tag}-{commit_hash}"
