#!/usr/bin/env python3
"""Audit paired R2E Agent images without exposing verifier assets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile

import docker


PROTECTED_PATHS = (
    "/r2e_tests",
    "/root/r2e_tests",
    "/testbed/r2e_tests",
    "/root/run_tests.sh",
    "/testbed/run_tests.sh",
    "/root/expected_test_output.json",
    "/testbed/expected_test_output.json",
)


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def audit_one(source: str, record: dict) -> tuple[str, dict]:
    client = docker.from_env(timeout=900)
    container = None
    try:
        reference = f"{record['agent_reference']}@{record['agent_digest']}"
        container = client.containers.run(
            reference,
            command=["/bin/bash", "-lc", "sleep infinity"],
            detach=True,
            network_mode="none",
        )
        container.reload()
        if container.attrs.get("HostConfig", {}).get("NetworkMode") != "none":
            raise RuntimeError("agent audit container is not offline")
        paths = " ".join(PROTECTED_PATHS)
        command = (
            "set -e; "
            "test \"$(cat /opt/r2egym/dataset-adapter)\" = r2e-gym-subset-agent-v1; "
            f"for path in {paths}; do "
            "test ! -e \"$path\" && test ! -L \"$path\"; done; "
            "test -z \"$(git -C /testbed status --porcelain --untracked-files=all)\""
        )
        result = container.exec_run(["/bin/bash", "-lc", command])
        if int(result.exit_code) != 0:
            raise RuntimeError(result.output.decode(errors="replace")[-4000:])
        return source, {
            "passed": True,
            "agent_reference": record["agent_reference"],
            "agent_digest": record["agent_digest"],
            "network_mode": "none",
            "dangling_links_checked": True,
        }
    except Exception as exc:
        return source, {
            "passed": False,
            "agent_reference": record.get("agent_reference"),
            "agent_digest": record.get("agent_digest"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if container is not None:
            container.remove(force=True)
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    images = {
        source: record
        for source, record in manifest.get("images", {}).items()
        if record.get("agent_reference") and record.get("agent_digest")
    }
    report = {
        "schema_version": "r2e-gym-subset-agent-isolation-audit-v1",
        "manifest": str(Path(args.manifest).resolve()),
        "total": len(images),
        "passed": 0,
        "failed": 0,
        "results": {},
    }
    report_path = Path(args.report)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(audit_one, source, record)
            for source, record in images.items()
        ]
        for index, future in enumerate(as_completed(futures), 1):
            source, result = future.result()
            report["results"][source] = result
            report["passed" if result["passed"] else "failed"] += 1
            write_atomic(report_path, report)
            print(
                f"AGENT_AUDIT {index}/{len(images)} passed={report['passed']} "
                f"failed={report['failed']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
