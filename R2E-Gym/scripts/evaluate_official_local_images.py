#!/usr/bin/env python3
"""Run the official SWE-bench 3.0.2 evaluator against pre-pulled images.

This adapter only replaces TestSpec construction so evaluation can stay local.
Patch application, test execution, parsing, and per-instance reports are the
official ``swebench.harness.run_evaluation.run_instance`` implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
from pathlib import Path

import docker
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import TestSpec


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    return json.loads(str(value))


def make_official_eval_spec(instance: dict, namespace: str) -> TestSpec:
    """Construct the official TestSpec without generating unused build scripts."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = str(instance["version"])
    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    eval_script_list = make_eval_script_list(
        instance,
        specs,
        env_name,
        repo_directory,
        instance["base_commit"],
        instance["test_patch"],
    )
    arch = "x86_64"
    if platform.machine() in {"aarch64", "arm64"} and instance_id not in USE_X86:
        arch = "arm64"
    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        repo_script_list=[],
        eval_script_list=eval_script_list,
        env_script_list=[],
        arch=arch,
        FAIL_TO_PASS=json_list(instance.get("FAIL_TO_PASS")),
        PASS_TO_PASS=json_list(instance.get("PASS_TO_PASS")),
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=specs.get("docker_specs", {}),
        namespace=namespace,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--namespace", default="docker.1ms.run/swebench")
    parser.add_argument("--instance-id", action="append")
    args = parser.parse_args()

    dataset = {row["instance_id"]: row for row in read_jsonl(args.dataset)}
    predictions = {row[KEY_INSTANCE_ID]: row for row in read_jsonl(args.predictions)}
    selected = args.instance_id or list(predictions)
    selected = [iid for iid in selected if predictions[iid].get(KEY_PREDICTION)]
    client = docker.from_env()

    missing = []
    jobs = []
    for iid in selected:
        # The image already exists, so only the official evaluation portion of
        # TestSpec is needed. Generating env/repo build scripts would perform
        # unnecessary network access for some historical repositories.
        spec = make_official_eval_spec(dataset[iid], args.namespace)
        try:
            client.images.get(spec.instance_image_key)
        except docker.errors.ImageNotFound:
            missing.append((iid, spec.instance_image_key))
            continue
        jobs.append((spec, predictions[iid]))
    if missing:
        raise SystemExit(f"Missing {len(missing)} images; first: {missing[0]}")

    print(f"Running {len(jobs)} non-empty predictions with {args.workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_instance,
                spec,
                pred,
                False,
                False,
                client,
                args.run_id,
                args.timeout,
                False,
            ): spec.instance_id
            for spec, pred in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            iid = futures[future]
            try:
                result = future.result()
                print(f"DONE {iid} result={result is not None}", flush=True)
            except Exception as exc:
                print(f"ERROR {iid}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
