#!/usr/bin/env python3
"""Read-only, post-publication audit of R2E prepared images."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

from r2egym.imageprep.base import VerificationContext
from r2egym.imageprep.manifest import (
    PreparedImageSpec,
    load_manifest,
    write_manifest_atomic,
)
from r2egym.imageprep.r2e_tasks import load_r2e_tasks
from r2egym.imageprep.r2e_verifier import verify_r2e_image


def _audit_one(task, manifest):
    client = docker.from_env(timeout=900)
    try:
        record = manifest["images"].get(task["docker_image"])
        if not record:
            return task["instance_id"], False, {"reason_code": "prepared_image_missing"}
        if record.get("verifier_digest"):
            image = f"{record['verifier_reference']}@{record['verifier_digest']}"
        elif record.get("status") == "verified-local" and record.get("verifier_local_reference"):
            local = client.images.get(record["verifier_local_reference"])
            if local.id != record.get("verifier_image_id"):
                return task["instance_id"], False, {"reason_code": "audited_image_id_mismatch"}
            image = record["verifier_local_reference"]
        else:
            return task["instance_id"], False, {"reason_code": "prepared_image_missing"}
        spec = PreparedImageSpec(
            source_image=record["source_image"],
            source_digest=record["source_digest"],
            prepared_repository=record["prepared_repository"],
            prepared_tag=record["prepared_tag"],
            dataset_type=record["dataset_type"],
            repo=record["repo"],
            profile=record["profile"],
            architecture=record["architecture"],
            instance_ids=tuple(record["instance_ids"]),
            prepare_version=record["prepare_version"],
            sanitization_version=record.get("sanitization_version", ""),
        )
        result = verify_r2e_image(client, image, VerificationContext(spec, (task,)))
        return task["instance_id"], True, result.to_dict()
    except Exception as exc:
        return task["instance_id"], False, {
            "reason_code": getattr(exc, "reason_code", "audit_error"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--task-count", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.get("dataset_type") != "r2e-gym-subset":
        parser.error("manifest is not an R2E-Gym-Subset manifest")
    tasks, fingerprints = load_r2e_tasks(
        args.source_dir, args.task_count, args.selection_seed
    )
    if manifest.get("selection", {}).get("fingerprints") != fingerprints:
        parser.error("dataset fingerprint drift")
    report = {
        "schema_version": "r2e-gym-subset-offline-audit-v1",
        "source_manifest": str(Path(args.manifest).resolve()),
        "passed": 0,
        "failed": 0,
        "results": {},
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_audit_one, task, manifest) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            instance_id, passed, result = future.result()
            report["passed" if passed else "failed"] += 1
            report["results"][instance_id] = {"passed": passed, **result}
            write_manifest_atomic(args.report, report)
            print(f"AUDIT {index}/{len(tasks)} {instance_id}", flush=True)


if __name__ == "__main__":
    main()
