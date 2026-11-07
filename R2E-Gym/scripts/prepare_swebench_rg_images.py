#!/usr/bin/env python3
"""Bake pinned ripgrep into SWE-bench task images and rewrite JSONL inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time


def run(command: list[str], *, timeout: int = 1800) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout[-4000:]}")
    return result.stdout


def image_exists(reference: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", reference],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def audit(reference: str) -> None:
    check = (
        "set -eu; test -f /usr/local/bin/rg; test -x /usr/local/bin/rg; "
        "test ! -L /usr/local/bin/rg; "
        "/usr/local/bin/rg --version | grep -Fx 'ripgrep 15.1.0 (rev af60c2de9d)'; "
        "test ! -e /run_tests.sh; test ! -e /root/test.patch; test ! -e /root/gold.patch; "
        "test ! -e /testbed/test.patch; test ! -e /testbed/gold.patch"
    )
    run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh",
         reference, "-lc", check], timeout=300)


def prepare(source: str, target: str, context: Path) -> dict[str, str]:
    if image_exists(target):
        audit(target)
        return {"source": source, "target": target, "status": "reused"}

    run(["docker", "build", "--pull=false", "--network", "none",
         "--build-arg", f"BASE={source}", "--tag", target, str(context)], timeout=1800)
    audit(target)
    return {"source": source, "target": target, "status": "built"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--rg", required=True)
    parser.add_argument("--repository", default="r2egym-local/swebench-rollout-rg15")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise SystemExit("--input and --output counts must match")
    rg_path = Path(args.rg).resolve()
    version = run([str(rg_path), "--version"]).splitlines()[0]
    if version != "ripgrep 15.1.0 (rev af60c2de9d)":
        raise SystemExit(f"unexpected rg version: {version}")
    rg_sha256 = hashlib.sha256(rg_path.read_bytes()).hexdigest()

    groups: list[list[dict]] = []
    jobs: dict[str, tuple[str, str]] = {}
    for input_name in args.input:
        rows = [json.loads(line) for line in Path(input_name).read_text().splitlines() if line.strip()]
        groups.append(rows)
        for row in rows:
            source = str(row.get("docker_image") or row.get("image_name"))
            instance = str(row["instance_id"])
            tag = hashlib.sha256(f"{source}\0{instance}\0{rg_sha256}".encode()).hexdigest()[:24]
            jobs[instance] = (source, f"{args.repository}:{tag}")

    started = time.time()
    results: list[dict[str, str]] = []
    lock = threading.Lock()
    with tempfile.TemporaryDirectory(prefix="swebench-rg-build-") as directory:
        context = Path(directory)
        (context / "rg").write_bytes(rg_path.read_bytes())
        (context / "rg").chmod(0o755)
        (context / "Dockerfile").write_text(
            "ARG BASE\nFROM ${BASE}\nUSER root\n"
            "COPY --chmod=0755 rg /usr/local/bin/rg\n"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(prepare, source, target, context): instance
                       for instance, (source, target) in jobs.items()}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                instance = futures[future]
                result = future.result()
                result["instance_id"] = instance
                results.append(result)
                with lock:
                    print(f"[{index}/{len(futures)}] {result['status']} {instance}", flush=True)

    for rows, output_name in zip(groups, args.output):
        output = Path(output_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                source, target = jobs[str(row["instance_id"])]
                rewritten = dict(row)
                rewritten["source_docker_image"] = source
                rewritten["docker_image"] = target
                if "image_name" in rewritten:
                    rewritten["image_name"] = target
                stream.write(json.dumps(rewritten, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "swebench-rollout-rg-images-v1",
        "rg_version": version,
        "rg_sha256": rg_sha256,
        "tasks": len(jobs),
        "unique_targets": len({target for _, target in jobs.values()}),
        "built": sum(r["status"] == "built" for r in results),
        "reused": sum(r["status"] == "reused" for r in results),
        "elapsed_sec": time.time() - started,
        "results": sorted(results, key=lambda value: value["instance_id"]),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "tasks", "unique_targets", "built", "reused", "elapsed_sec", "rg_sha256")}, indent=2))


if __name__ == "__main__":
    main()
