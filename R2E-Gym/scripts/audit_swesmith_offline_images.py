#!/usr/bin/env python3
"""Re-verify prepared SWE-smith images using strict per-task offline gates."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

from r2egym.imageprep.manifest import (
    PreparedImageSpec,
    group_tasks_by_image,
    load_manifest,
    load_task_rows,
    write_manifest_atomic,
)
from r2egym.imageprep.swesmith import VerificationContext
from r2egym.imageprep.verifier import verify_image
from r2egym.imageprep.manifest import MANIFEST_SCHEMA_VERSION
from r2egym.imageprep.sanitization import SWESMITH_SANITIZATION_VERSION
from swesmith.profiles import registry as swesmith_registry


REPORT_SCHEMA = "swesmith-offline-audit-v3"
TERMINAL_STATUSES = {"passed", "failed", "prepared_image_missing", "task_data_missing"}


def _spec_from_record(source_image: str, record: dict) -> PreparedImageSpec:
    return PreparedImageSpec(
        source_image=source_image,
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


def _audit_one(source_image: str, tasks: list[dict], record: dict, task_workers: int) -> tuple[str, dict]:
    client = docker.from_env(timeout=900)
    pulled_image = None
    try:
        digest = record["prepared_digest"]
        reference = f"{record['prepared_tag_reference']}@{digest}"
        pulled_image = client.images.pull(reference)
        profile = swesmith_registry.get_from_inst(tasks[0])
        test_command, _ = profile.get_test_cmd(tasks[0])
        result = verify_image(
            client,
            reference,
            VerificationContext(
                _spec_from_record(source_image, record),
                test_command,
                tuple(tasks),
                task_workers,
            ),
        )
        return source_image, {"status": "passed", "verification": result.to_dict()}
    finally:
        if pulled_image is not None:
            try:
                client.images.remove(pulled_image.id, force=True)
            except docker.errors.APIError as exc:
                print(
                    f"AUDIT_CLEANUP_WARNING {source_image} {type(exc).__name__}: {exc}",
                    flush=True,
                )
        client.close()


def _load_or_create_report(path: str, source_manifest: str) -> dict:
    report_path = Path(path)
    resolved_manifest = str(Path(source_manifest).resolve())
    if not report_path.exists():
        return {
            "schema_version": REPORT_SCHEMA,
            "source_manifest": resolved_manifest,
            "images": {},
        }
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"cannot resume report with schema {report.get('schema_version')!r}")
    if report.get("source_manifest") != resolved_manifest:
        raise ValueError("cannot resume report created from a different source manifest")
    if not isinstance(report.get("images"), dict):
        raise ValueError("cannot resume report without an images mapping")
    report.pop("summary", None)
    return report


def _build_summary(report: dict) -> dict:
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    strict_passed_tasks = 0
    pr_empty_p2p_accepted = 0
    warning_count = 0
    warning_tasks = 0
    for image in report["images"].values():
        status = str(image.get("status", "unknown"))
        statuses[status] += 1
        reason_code = image.get("reason_code")
        if status != "passed" and reason_code:
            reasons[str(reason_code)] += 1
        if status != "passed":
            continue
        tasks = image.get("verification", {}).get("checks", {}).get("tasks", {})
        strict_passed_tasks += sum(
            task.get("status", "passed") == "passed" for task in tasks.values()
        )
        for task in tasks.values():
            modes[str(task.get("testing_mode", "unknown"))] += 1
            pr_empty_p2p_accepted += int(bool(task.get("pr_empty_p2p")))
            warnings = task.get("warnings") or []
            warning_count += len(warnings)
            warning_tasks += int(bool(warnings))
    return {
        "statuses": dict(sorted(statuses.items())),
        "strict_passed_tasks": strict_passed_tasks,
        "testing_modes": dict(sorted(modes.items())),
        "pr_empty_p2p_accepted": pr_empty_p2p_accepted,
        "full_suite_warning_count": warning_count,
        "full_suite_warning_tasks": warning_tasks,
        "failure_reasons": dict(sorted(reasons.items())),
    }


def _build_gated_manifest(source: dict, report: dict) -> dict:
    gated = copy.deepcopy(source)
    gated["schema_version"] = MANIFEST_SCHEMA_VERSION
    gated["sanitization_version"] = SWESMITH_SANITIZATION_VERSION
    for source_image, record in gated["images"].items():
        task_results = (
            report.get("images", {}).get(source_image, {}).get("verification", {})
            .get("checks", {}).get("tasks", {})
        )
        sanitized = sorted(
            instance_id for instance_id, result in task_results.items()
            if result.get("status") == "passed"
            and result.get("functional_audit_passed") is True
            and result.get("sanitization_passed") is True
        )
        record["sanitization_version"] = SWESMITH_SANITIZATION_VERSION
        record["agent_exposable"] = False
        record["sanitized_instance_ids"] = sanitized
        record["isolation_verified"] = bool(sanitized)
        record["task_audit"] = task_results
    return gated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gated-manifest", required=True)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent image groups")
    parser.add_argument("--task-workers", type=int, default=12, help="Concurrent per-task containers")
    parser.add_argument("--source-image", action="append")
    args = parser.parse_args()

    rows = load_task_rows(args.input, args.task_count)
    grouped = group_tasks_by_image(rows)
    selected = set(args.source_image or grouped)
    manifest = load_manifest(args.manifest)
    report = _load_or_create_report(args.report, args.manifest)
    lock = threading.Lock()
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source_image in sorted(selected):
            previous = report["images"].get(source_image, {})
            if previous.get("status") in TERMINAL_STATUSES:
                continue
            tasks = grouped.get(source_image)
            record = manifest["images"].get(source_image)
            if not tasks:
                report["images"][source_image] = {
                    "status": "task_data_missing",
                    "reason_code": "task_data_missing",
                }
            elif not record:
                report["images"][source_image] = {
                    "status": "prepared_image_missing",
                    "reason_code": "prepared_image_missing",
                    "task_count": len(tasks),
                }
            else:
                futures[pool.submit(_audit_one, source_image, tasks, record, args.task_workers)] = source_image
        write_manifest_atomic(args.report, report)
        for future in as_completed(futures):
            source_image = futures[future]
            try:
                _, result = future.result()
            except Exception as exc:
                result = {
                    "status": "failed",
                    "reason_code": getattr(
                        exc, "reason_code", "offline_verification_error"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            with lock:
                report["images"][source_image] = result
                write_manifest_atomic(args.report, report)
                print(f"AUDIT {result['status'].upper()} {source_image}", flush=True)

    summary = _build_summary(report)
    report["summary"] = summary
    write_manifest_atomic(args.report, report)
    write_manifest_atomic(args.gated_manifest, _build_gated_manifest(manifest, report))
    print(json.dumps(summary, sort_keys=True), flush=True)
    statuses = summary["statuses"]
    if statuses.get("failed") or statuses.get("task_data_missing"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
