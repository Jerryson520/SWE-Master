#!/usr/bin/env python3
"""Build, audit once offline, and optionally publish R2E prepared images."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading

import docker

from r2egym.imageprep.builder import PreparedImageBuilder, registry_digest, remote_digest, source_digest
from r2egym.imageprep.manifest import (
    R2E_MANIFEST_SCHEMA_VERSION,
    PreparedImageSpec,
    load_manifest,
    make_prepared_tag,
    new_manifest,
    write_manifest_atomic,
)
from r2egym.imageprep.r2e import R2EImageAdapter, SANITIZATION_VERSION
from r2egym.imageprep.r2e_tasks import load_r2e_tasks, r2e_source_mirror_reference


_PUBLISH_LOCK = threading.Lock()


def _build_one(task, args, resume_record):
    client = docker.from_env(timeout=900)
    try:
        source_image = task["docker_image"]
        mirror = r2e_source_mirror_reference(source_image, args.source_registry)
        digest, source_reference = source_digest(
            client, source_image, registry_reference=mirror
        )
        repo = task["repo_name"]
        spec = PreparedImageSpec(
            source_image=source_image,
            source_digest=digest,
            prepared_repository=args.prepared_registry.rstrip("/"),
            prepared_tag=make_prepared_tag(
                repo,
                digest,
                args.prepare_version,
                args.architecture,
                task["commit_hash"],
            ),
            dataset_type="r2e-gym-subset",
            repo=repo,
            profile="r2e",
            architecture=args.architecture,
            instance_ids=(task["instance_id"],),
            prepare_version=args.prepare_version,
            sanitization_version=SANITIZATION_VERSION,
        )
        if resume_record and resume_record.get("resume_key") == spec.resume_key:
            if resume_record.get("status") == "published" and resume_record.get("agent_digest"):
                for role in ("agent", "verifier"):
                    current = registry_digest(client, resume_record[f"{role}_reference"])
                    if current != resume_record[f"{role}_digest"]:
                        raise RuntimeError(f"{role} registry digest drift")
                return source_image, resume_record, True
            if resume_record.get("status") == "verified-local" and not args.push:
                for role in ("agent", "verifier"):
                    image = client.images.get(resume_record.get(f"{role}_local_reference"))
                    if image.id != resume_record.get(f"{role}_image_id"):
                        raise RuntimeError(f"local {role} image ID differs from audited ID")
                return source_image, resume_record, True
            if args.publish_verified:
                if resume_record.get("status") != "verified-local":
                    raise RuntimeError("publish-verified requires a verified-local record")
                published = dict(resume_record)
                for role in ("agent", "verifier"):
                    image = client.images.get(resume_record[f"{role}_local_reference"])
                    if image.id != resume_record[f"{role}_image_id"]:
                        raise RuntimeError(f"local {role} image ID differs from audited ID")
                    reference = resume_record[f"{role}_reference"]
                    repository, tag = reference.rsplit(":", 1)
                    image.tag(repository, tag)
                    with _PUBLISH_LOCK:
                        errors = [
                            event["error"] for event in client.images.push(
                                repository, tag=tag, stream=True, decode=True,
                            ) if event.get("error")
                        ]
                    if errors:
                        raise RuntimeError("registry push failed: " + "; ".join(errors))
                    published[f"{role}_digest"] = remote_digest(
                        client, reference, expected_image_id=image.id
                    )
                published["prepared_digest"] = published["verifier_digest"]
                published["status"] = "published"
                return source_image, published, False
        record = PreparedImageBuilder(client).build_and_verify(
            spec,
            source_reference=source_reference,
            test_command="",
            tasks=[task],
            push=args.push,
            adapter=R2EImageAdapter(),
        )
        record["source_mirror_image"] = mirror
        return source_image, record, False
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--task-count", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument(
        "--prepared-registry",
        default=os.environ.get(
            "R2EGYM_R2E_PREPARED_REGISTRY",
            "crpi-jme0bqk6txpdw1cg-vpc.cn-hangzhou.personal.cr.aliyuncs.com/"
            "swe-master-personal/r2e-gym-offline",
        ),
        required=False,
    )
    parser.add_argument("--prepare-version", default="offline-v2")
    parser.add_argument("--architecture", default="linux/x86_64")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--publish-verified", action="store_true")
    parser.add_argument("--audit-report")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.publish_verified:
        if not args.push or not args.resume:
            parser.error("--publish-verified requires --push --resume")
        if args.audit_report:
            audit = json.loads(Path(args.audit_report).read_text())
            if audit.get("passed") != args.task_count or audit.get("failed") != 0:
                parser.error("publish refused: audit must have passed=task-count and failed=0")
    if not args.prepared_registry:
        parser.error("--prepared-registry is required")
    if args.source_registry.rstrip("/") == args.prepared_registry.rstrip("/"):
        parser.error("source registry is read-only and cannot be the prepared registry")
    tasks, fingerprints = load_r2e_tasks(
        args.source_dir, args.task_count, args.selection_seed
    )
    manifest_path = Path(args.manifest)
    if args.resume and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("dataset_type") != "r2e-gym-subset":
            parser.error("existing manifest belongs to a different dataset")
        if manifest.get("schema_version") != R2E_MANIFEST_SCHEMA_VERSION:
            parser.error("legacy R2E manifest cannot be resumed as a secure v2 build")
        if manifest.get("selection", {}).get("fingerprints") != fingerprints:
            parser.error("dataset fingerprint drift")
    else:
        manifest = new_manifest(
            args.prepare_version,
            args.prepared_registry,
            dataset_type="r2e-gym-subset",
            source_registry=args.source_registry,
            selection={
                "task_count": len(tasks),
                "selection_seed": args.selection_seed,
                "fingerprints": fingerprints,
            },
        )
    print(f"Preparing {len(tasks)} R2E tasks across {len(tasks)} source images", flush=True)
    lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _build_one,
                task,
                args,
                manifest["images"].get(task["docker_image"]) if args.resume else None,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                source_image, record, skipped = future.result()
            except Exception as exc:
                failures.append((task["instance_id"], f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {task['instance_id']} {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                manifest["images"][source_image] = record
                write_manifest_atomic(manifest_path, manifest)
            print(
                f"{'SKIP' if skipped else 'DONE'} {completed}/{len(tasks)} "
                f"{task['instance_id']}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} image preparations failed")


if __name__ == "__main__":
    main()
