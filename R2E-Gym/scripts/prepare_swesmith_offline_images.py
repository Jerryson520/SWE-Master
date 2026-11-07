#!/usr/bin/env python3
"""Build, verify, and optionally publish SWE-smith offline images."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

from r2egym.imageprep.builder import PreparedImageBuilder, registry_digest, source_digest
from r2egym.imageprep.manifest import (
    DEFAULT_PREPARED_REGISTRY,
    PreparedImageSpec,
    group_tasks_by_image,
    load_manifest,
    load_task_rows,
    make_prepared_tag,
    new_manifest,
    write_manifest_atomic,
)
from r2egym.imageprep.sanitization import SWESMITH_SANITIZATION_VERSION
from swesmith.profiles import registry as swesmith_registry
from swesmith.profiles.golang import GoProfile


def _build_one(source_image, tasks, args):
    client = docker.from_env(timeout=900)
    try:
        print(f"RESOLVE {source_image}", flush=True)
        digest, source_reference = source_digest(client, source_image)
        profile = swesmith_registry.get_from_inst(tasks[0])
        profile_module = type(profile).__module__.rsplit(".", 1)[-1]
        profile_name = "go" if isinstance(profile, GoProfile) else profile_module
        test_command, _ = profile.get_test_cmd(tasks[0])
        repo = str(tasks[0].get("repo") or tasks[0].get("repo_name"))
        instance_ids = tuple(sorted({str(task["instance_id"]) for task in tasks}))
        match = re.search(r"\.([0-9a-f]{8,40})-latest$", source_image)
        profile_commit = match.group(1) if match else None
        tag = make_prepared_tag(
            repo, digest, args.prepare_version, args.architecture, profile_commit
        )
        spec = PreparedImageSpec(
            source_image=source_image,
            source_digest=digest,
            prepared_repository=args.registry.rstrip("/"),
            prepared_tag=tag,
            dataset_type="swesmith",
            repo=repo,
            profile=profile_name,
            architecture=args.architecture,
            instance_ids=instance_ids,
            prepare_version=args.prepare_version,
            sanitization_version=SWESMITH_SANITIZATION_VERSION,
        )
        if args.resume_record and args.resume_record.get("resume_key") == spec.resume_key:
            old = args.resume_record
            if old.get("status") == "published" and old.get("prepared_digest"):
                # Resume validation only needs the immutable registry manifest
                # digest. Pulling all layers causes avoidable I/O and 429s.
                digest_at_registry = registry_digest(
                    client, old["prepared_tag_reference"]
                )
                if digest_at_registry != old["prepared_digest"]:
                    raise RuntimeError(
                        "registry digest drift for immutable prepared tag: "
                        f"{old['prepared_tag_reference']}"
                    )
                return source_image, old, True
        record = PreparedImageBuilder(client).build_and_verify(
            spec,
            source_reference=source_reference,
            test_command=test_command,
            tasks=tasks,
            push=args.push,
        )
        if args.cleanup_local and args.push:
            client.images.remove(record["audited_image_id"], force=True)
        return source_image, record, False
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task-count", type=int)
    parser.add_argument(
        "--source-image",
        action="append",
        help="Prepare only the exact source image; repeat for multiple smoke images.",
    )
    parser.add_argument(
        "--registry",
        default=os.environ.get("R2EGYM_PREPARED_REGISTRY", DEFAULT_PREPARED_REGISTRY),
    )
    parser.add_argument("--prepare-version", default="offline-v1")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--architecture", default="linux/x86_64")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--cleanup-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After a verified push, remove the local Prepared Image and dangling layers.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = load_task_rows(args.input, args.task_count)
    if args.source_image:
        selected = set(args.source_image)
        rows = [
            row for row in rows
            if (row.get("docker_image") or row.get("image_name")) in selected
        ]
        missing = selected - {
            str(row.get("docker_image") or row.get("image_name")) for row in rows
        }
        if missing:
            parser.error("source image is not present in selected tasks: " + ", ".join(sorted(missing)))
    grouped = group_tasks_by_image(rows)
    path = Path(args.manifest)
    if args.resume and path.exists():
        manifest = load_manifest(path)
        if manifest.get("prepare_version") != args.prepare_version:
            parser.error("existing manifest prepare_version does not match")
        if manifest.get("registry") != args.registry.rstrip("/"):
            parser.error("existing manifest registry does not match")
    else:
        manifest = new_manifest(args.prepare_version, args.registry)
    print(f"Preparing {len(rows)} tasks across {len(grouped)} source images", flush=True)
    lock = threading.Lock()
    failures = []
    processed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for source_image, tasks in grouped.items():
            task_args = argparse.Namespace(**vars(args))
            task_args.resume_record = manifest["images"].get(source_image) if args.resume else None
            future = pool.submit(_build_one, source_image, tasks, task_args)
            futures[future] = source_image
        for future in as_completed(futures):
            expected_source = futures[future]
            try:
                source_image, record, skipped = future.result()
            except Exception as exc:
                failures.append((expected_source, f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {expected_source} {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                manifest["images"][source_image] = record
                write_manifest_atomic(path, manifest)
                processed += 1
                print(
                    f"{'SKIP' if skipped else 'DONE'} {processed}/{len(grouped)} "
                    f"manifest_total={len(manifest['images'])} "
                    f"{source_image} -> {record['prepared_tag_reference']}",
                    flush=True,
                )
    if args.cleanup_local and args.push:
        # Pruning from worker threads can race with Docker's classic builder
        # while it commits another image. Clean only after every build stops.
        cleanup_client = docker.from_env(timeout=900)
        try:
            for image in cleanup_client.images.list():
                if any(
                    tag.startswith("r2egym-local/swe-smith-offline:")
                    for tag in image.tags
                ):
                    cleanup_client.images.remove(image.id, force=True)
            cleanup_client.images.prune(filters={"dangling": True})
        finally:
            cleanup_client.close()
    if failures:
        print(f"FAILED {len(failures)}/{len(grouped)} image groups", flush=True)
        for source_image, error in failures:
            print(f"FAILED_IMAGE {source_image} {error}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
