#!/usr/bin/env python3
"""Prepare one paired, offline SWE-bench Verified image and publish a manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import docker

from r2egym.agenthub.leaf.swebench_patch_verifier import SWEbenchPatchVerifier
from r2egym.agenthub.leaf.tools.ripgrep import resolve_host_rg
from r2egym.imageprep.builder import registry_digest
from r2egym.imageprep.manifest import write_manifest_atomic


SCHEMA = "swebench-verified-paired-image-v1"
PREPARE_VERSION = "leaf-offline-v1"


def push(client, image, repository: str, tag: str) -> tuple[str, str]:
    reference = f"{repository}:{tag}"
    image.tag(repository, tag)
    errors = [event["error"] for event in client.images.push(
        repository, tag=tag, stream=True, decode=True
    ) if event.get("error")]
    if errors:
        raise RuntimeError("registry push failed: " + "; ".join(errors))
    return reference, registry_digest(client, reference)


def build(client, context: Path, tag: str):
    image, logs = client.images.build(path=str(context), tag=tag, rm=True, forcerm=True)
    return image


def exec_checked(container, command: str) -> str:
    result = container.exec_run(["/bin/bash", "-lc", command])
    output = result.output.decode(errors="replace")
    if int(result.exit_code) != 0:
        raise RuntimeError(f"audit failed ({result.exit_code}): {output[-4000:]}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise SystemExit("pilot preparation requires exactly one task")
    row = rows[0]
    source = str(row["docker_image"])
    instance_id = str(row["instance_id"])
    safe_id = instance_id.replace("__", "-").replace("/", "-").replace("_", "-").lower()
    fingerprint = hashlib.sha256(source.encode()).hexdigest()[:12]
    verifier_tag = f"{safe_id}-{fingerprint}-{PREPARE_VERSION}"
    agent_tag = f"{verifier_tag}-agent"
    local_verifier = f"r2egym-local/swebench-verified:{verifier_tag}"
    local_agent = f"r2egym-local/swebench-verified:{agent_tag}"
    source_image = client = None
    client = docker.from_env(timeout=1800)
    try:
        source_image = client.images.get(source)
        source_digest = source_image.id
        run_tests = str(row["run_tests"])
        encoded_runner = base64.b64encode(run_tests.encode()).decode()
        with tempfile.TemporaryDirectory(prefix="swebench-verified-verifier-") as directory:
            root = Path(directory)
            shutil.copy2(resolve_host_rg(), root / "rg")
            (root / "rg").chmod(0o755)
            prepare = f'''#!/usr/bin/env bash
set -euo pipefail
test -d /testbed/.git
test "$(git -C /testbed rev-parse {row['base_commit']}^{{commit}})" = "{row['base_commit']}"
git config --global --add safe.directory /testbed
printf '%s' '{encoded_runner}' | base64 -d > /run_tests.sh
chmod 0755 /run_tests.sh
CURRENT_HEAD="$(git -C /testbed rev-parse HEAD^{{commit}})"
while IFS= read -r ref; do git -C /testbed update-ref -d "$ref"; done < <(git -C /testbed for-each-ref --format='%(refname)')
git -C /testbed symbolic-ref HEAD refs/heads/leaf-baseline
git -C /testbed update-ref refs/heads/leaf-baseline "$CURRENT_HEAD"
rm -rf /testbed/.git/logs /testbed/.git/refs/remotes /testbed/.git/refs/original
rm -f /testbed/.git/ORIG_HEAD /testbed/.git/FETCH_HEAD
git -C /testbed reflog expire --expire=now --all
git -C /testbed gc --prune=now
mkdir -p /opt/r2egym
printf '%s\n' 'swebench-verified-verifier-v1' > /opt/r2egym/dataset-adapter
'''
            (root / "prepare.sh").write_text(prepare)
            (root / "prepare.sh").chmod(0o755)
            (root / "Dockerfile").write_text(
                f"FROM {source}\nUSER root\n"
                "COPY rg /usr/local/bin/rg\nCOPY prepare.sh /tmp/prepare.sh\n"
                "RUN chmod 0755 /usr/local/bin/rg /tmp/prepare.sh && /tmp/prepare.sh && rm /tmp/prepare.sh\n"
                "ENV PATH=/opt/miniconda3/envs/testbed/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin "
                "PIP_NO_INDEX=1 UV_OFFLINE=1 GOPROXY=off\nWORKDIR /testbed\n"
            )
            verifier_image = build(client, root, local_verifier)

        with tempfile.TemporaryDirectory(prefix="swebench-verified-agent-") as directory:
            root = Path(directory)
            (root / "Dockerfile").write_text(
                f"FROM {local_verifier}\nUSER root\n"
                "RUN rm -f /run_tests.sh /root/run_tests.sh /testbed/run_tests.sh "
                "/root/test.patch /root/gold.patch /testbed/test.patch /testbed/gold.patch "
                "&& printf '%s\\n' 'swebench-verified-agent-v1' > /opt/r2egym/dataset-adapter "
                "&& test -z \"$(git -C /testbed status --porcelain --untracked-files=no)\"\n"
                "WORKDIR /testbed\n"
            )
            agent_image = build(client, root, local_agent)

        verifier_container = client.containers.run(
            local_verifier, ["/bin/bash", "-lc", "sleep infinity"],
            detach=True, network_mode="none",
        )
        try:
            exec_checked(verifier_container,
                "test -x /run_tests.sh; test -d /testbed/.git; "
                "test -z \"$(git -C /testbed remote)\"; "
                "test \"$(cat /opt/r2egym/dataset-adapter)\" = swebench-verified-verifier-v1")
        finally:
            verifier_container.remove(force=True)
        agent_container = client.containers.run(
            local_agent, ["/bin/bash", "-lc", "sleep infinity"],
            detach=True, network_mode="none",
        )
        try:
            exec_checked(agent_container,
                "test ! -e /run_tests.sh; test ! -e /root/test.patch; "
                "test ! -e /root/gold.patch; "
                "test \"$(cat /opt/r2egym/dataset-adapter)\" = swebench-verified-agent-v1; "
                "test -z \"$(git -C /testbed status --porcelain --untracked-files=no)\"")
        finally:
            agent_container.remove(force=True)

        verifier_reference, verifier_digest = push(
            client, verifier_image, args.repository, verifier_tag
        )
        agent_reference, agent_digest = push(
            client, agent_image, args.repository, agent_tag
        )
        verification = None
        for attempt in range(1, 4):
            verification = SWEbenchPatchVerifier(reward_timeout=1800).verify(
                entry=row,
                patch=str(row["patch"]),
                verifier_image=verifier_reference,
                verifier_digest=verifier_digest,
            )
            if verification.error_code != "verifier_infrastructure_error":
                break
            if attempt < 3:
                time.sleep(2)
        assert verification is not None
        if verification.error or verification.reward != 1.0:
            raise RuntimeError(f"gold-patch verifier audit failed: {verification.to_dict()}")
        baseline = SWEbenchPatchVerifier(reward_timeout=1800).verify(
            entry=row,
            patch="",
            verifier_image=verifier_reference,
            verifier_digest=verifier_digest,
        )
        if baseline.error or baseline.reward != 0.0:
            raise RuntimeError(f"empty-patch verifier audit failed: {baseline.to_dict()}")
        record = {
            "source_image": source,
            "source_digest": source_digest,
            "dataset_type": "swebench-verified",
            "instance_ids": [instance_id],
            "prepare_version": PREPARE_VERSION,
            "sanitization_version": "swebench-base-history-v1",
            "status": "published",
            "agent_reference": agent_reference,
            "agent_digest": agent_digest,
            "agent_image_id": agent_image.id,
            "verifier_reference": verifier_reference,
            "verifier_digest": verifier_digest,
            "verifier_image_id": verifier_image.id,
            "isolation_verified": True,
            "offline_verification": {
                "passed": True,
                "network_blocked": True,
                "gold_patch_reward": verification.reward,
                "empty_patch_reward": baseline.reward,
                "verifier": verification.to_dict(),
            },
        }
        manifest = {
            "schema_version": SCHEMA,
            "dataset_type": "swebench-verified",
            "prepare_version": PREPARE_VERSION,
            "registry": args.repository,
            "selection": {"task_count": 1, "instance_ids": [instance_id]},
            "images": {source: record},
        }
        write_manifest_atomic(args.manifest, manifest)
        print(json.dumps({
            "instance_id": instance_id,
            "agent": f"{agent_reference}@{agent_digest}",
            "verifier": f"{verifier_reference}@{verifier_digest}",
            "gold_patch_reward": verification.reward,
            "manifest": args.manifest,
        }, indent=2))
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
