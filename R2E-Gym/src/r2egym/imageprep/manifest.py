"""Manifest types and lookup helpers for prepared task images."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = "swesmith-offline-image-v2"
LEGACY_SWESMITH_MANIFEST_SCHEMA_VERSION = "swesmith-offline-image-v1"
R2E_MANIFEST_SCHEMA_VERSION = "r2e-gym-subset-paired-image-v1"
SWEBENCH_VERIFIED_MANIFEST_SCHEMA_VERSION = "swebench-verified-paired-image-v1"
SINGLE_IMAGE_R2E_MANIFEST_SCHEMA_VERSION = "r2e-gym-subset-offline-image-v2"
LEGACY_R2E_MANIFEST_SCHEMA_VERSION = "r2e-gym-subset-offline-image-v1"
DEFAULT_PREPARED_REGISTRY = (
    "crpi-jme0bqk6txpdw1cg-vpc.cn-hangzhou.personal.cr.aliyuncs.com/"
    "swe-master-personal/swe-smith-offline"
)
_SAFE_COMPONENT = re.compile(r"[^a-z0-9_.-]+")
_SAFE_INSTANCE = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class PreparedImageSpec:
    source_image: str
    source_digest: str
    prepared_repository: str
    prepared_tag: str
    dataset_type: str
    repo: str
    profile: str
    architecture: str
    instance_ids: tuple[str, ...]
    prepare_version: str
    sanitization_version: str = ""

    @property
    def remote_tag(self) -> str:
        return f"{self.prepared_repository}:{self.prepared_tag}"

    @property
    def resume_key(self) -> str:
        material = "\n".join(
            [
                self.source_digest,
                self.prepare_version,
                self.sanitization_version,
                self.architecture,
                *sorted(self.instance_ids),
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["instance_ids"] = list(self.instance_ids)
        value["resume_key"] = self.resume_key
        return value


def validate_instance_id(value: str) -> str:
    parts = value.split("/")
    if (
        not value
        or not _SAFE_INSTANCE.fullmatch(value)
        or value.startswith("-")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"unsafe prepared-image instance_id: {value!r}")
    return value


def make_prepared_tag(
    repo: str,
    source_digest: str,
    prepare_version: str,
    architecture: str = "linux/x86_64",
    profile_commit: str | None = None,
) -> str:
    repo_key = repo.removeprefix("swesmith/").replace("/", "-")
    repo_key = repo_key.replace("__", "-").replace("_1776_", "-")
    repo_key = _SAFE_COMPONENT.sub("-", repo_key.lower()).strip("-._")
    arch = architecture.rsplit("/", 1)[-1].replace("amd64", "x86_64")
    version = _SAFE_COMPONENT.sub("-", prepare_version.lower()).strip("-._")
    digest = source_digest.removeprefix("sha256:")[:12]
    commit = ""
    if profile_commit:
        normalized_commit = _SAFE_COMPONENT.sub("-", profile_commit.lower()).strip("-._")[:12]
        repo_key = re.sub(rf"[.-]{re.escape(normalized_commit)}$", "", repo_key)
        commit = "-" + normalized_commit
    tag = f"{arch}-{repo_key}{commit}-{digest}-{version}"
    if len(tag) > 128:
        tag = f"{tag[:110].rstrip('-')}-{digest}-{version}"[:128]
    return tag


def load_task_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            row = raw.get("extra_info", raw)
            if isinstance(row, str):
                row = json.loads(row)
            if not isinstance(row, dict):
                raise ValueError("task row extra_info must be an object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def task_source_image(row: dict[str, Any]) -> str:
    image = row.get("docker_image") or row.get("image_name")
    if not image:
        raise ValueError(f"task {row.get('instance_id')} has no docker image")
    return str(image)


def group_tasks_by_image(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance_id = validate_instance_id(str(row.get("instance_id", "")))
        row = dict(row)
        row["instance_id"] = instance_id
        grouped.setdefault(task_source_image(row), []).append(row)
    return grouped


def new_manifest(
    prepare_version: str,
    registry: str,
    *,
    dataset_type: str = "swesmith",
    source_registry: str | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = (
        R2E_MANIFEST_SCHEMA_VERSION
        if dataset_type == "r2e-gym-subset"
        else SWEBENCH_VERIFIED_MANIFEST_SCHEMA_VERSION
        if dataset_type == "swebench-verified"
        else MANIFEST_SCHEMA_VERSION
    )
    value = {
        "schema_version": schema,
        "dataset_type": dataset_type,
        "prepare_version": prepare_version,
        "registry": registry.rstrip("/"),
        "images": {},
    }
    if source_registry:
        value["source_registry"] = source_registry.rstrip("/")
    if selection is not None:
        value["selection"] = selection
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") not in {
        MANIFEST_SCHEMA_VERSION,
        LEGACY_SWESMITH_MANIFEST_SCHEMA_VERSION,
        R2E_MANIFEST_SCHEMA_VERSION,
        SWEBENCH_VERIFIED_MANIFEST_SCHEMA_VERSION,
        SINGLE_IMAGE_R2E_MANIFEST_SCHEMA_VERSION,
        LEGACY_R2E_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"unsupported prepared image manifest schema: {value.get('schema_version')!r}"
        )
    if not isinstance(value.get("images"), dict):
        raise ValueError("prepared image manifest images must be an object")
    return value


def write_manifest_atomic(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, destination)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def resolve_prepared_image(
    manifest_path: str | Path,
    source_image: str,
    instance_id: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    is_swesmith = manifest.get("schema_version") in {
        MANIFEST_SCHEMA_VERSION,
        LEGACY_SWESMITH_MANIFEST_SCHEMA_VERSION,
    }
    if is_swesmith:
        from .sanitization import SWESMITH_SANITIZATION_VERSION

        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise RuntimeError("legacy SWE-smith manifests are not rollout eligible")
    if (
        manifest.get("dataset_type") == "r2e-gym-subset"
        and manifest.get("schema_version") != R2E_MANIFEST_SCHEMA_VERSION
    ):
        raise RuntimeError("single-image R2E manifests do not provide strict verifier isolation")
    record = manifest["images"].get(source_image)
    if not isinstance(record, dict):
        raise RuntimeError(f"source image is not prepared: {source_image}")
    if instance_id not in record.get("instance_ids", []):
        raise RuntimeError(
            f"prepared image for {source_image} does not cover instance {instance_id}"
        )
    if is_swesmith:
        if record.get("sanitization_version") != SWESMITH_SANITIZATION_VERSION:
            raise RuntimeError(f"prepared image has stale sanitization contract: {source_image}")
        if record.get("agent_exposable") is not False:
            raise RuntimeError(f"prepared image is not marked control-plane-only: {source_image}")
        if instance_id not in record.get("sanitized_instance_ids", []):
            raise RuntimeError(f"instance has not passed sanitization audit: {instance_id}")
        if record.get("isolation_verified") is not True:
            raise RuntimeError(f"prepared image isolation is not verified: {source_image}")
    verification = record.get("offline_verification", {})
    if not verification.get("passed") or not verification.get("network_blocked"):
        raise RuntimeError(f"prepared image is not offline verified: {source_image}")
    paired = manifest.get("dataset_type") in {"r2e-gym-subset", "swebench-verified"}
    digest = str(record.get("agent_digest" if paired else "prepared_digest", ""))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(f"prepared image has invalid digest: {source_image}")
    tag = record.get("agent_reference" if paired else "prepared_tag_reference")
    if not tag:
        raise RuntimeError(f"prepared image has no tag reference: {source_image}")
    result = dict(record)
    result["instance_id"] = instance_id
    result["prepared_reference"] = f"{tag}@{digest}"
    if paired:
        verifier_digest = str(record.get("verifier_digest", ""))
        verifier_tag = record.get("verifier_reference")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", verifier_digest) or not verifier_tag:
            raise RuntimeError(f"prepared image has invalid verifier identity: {source_image}")
        if record.get("isolation_verified") is not True:
            raise RuntimeError(f"prepared image has not passed isolation checks: {source_image}")
        result["verifier_prepared_reference"] = f"{verifier_tag}@{verifier_digest}"
    return result
