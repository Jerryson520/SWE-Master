import json
from pathlib import Path

import pytest

from r2egym.agenthub.leaf.antihack import (
    POLICY_BLOCKED_ERROR_CODE,
    AntiHackGuard,
)
from r2egym.agenthub.leaf.protocol import PROMPT_VERSION, R2E_OFFLINE_SYSTEM_PROMPT
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.imageprep.manifest import resolve_prepared_image


def test_antihack_blocks_remote_git_without_recording_raw_command(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    guard = AntiHackGuard({"enabled": True, "events_path": str(events)})
    output = guard.intercept(
        {"tool": "Bash", "params": {"command": "git fetch origin secret"}},
        event_context={"tool_call_id": "call-1", "instance_id": "task-1"},
    )

    assert output is not None
    assert output.success is False
    assert output.metadata["error_code"] == POLICY_BLOCKED_ERROR_CODE
    assert guard.summary()["blocked_tool_call_ids"] == ["call-1"]
    event_text = events.read_text()
    assert "git fetch origin secret" not in event_text
    assert json.loads(event_text)["pattern_id"] == "git_network"


def test_antihack_allows_sanitized_local_head_history():
    guard = AntiHackGuard(True)
    assert guard.intercept(
        {"tool": "Bash", "params": {"command": "git log --oneline HEAD~3..HEAD"}}
    ) is None


def test_paired_manifest_resolves_agent_and_verifier_digests(tmp_path: Path):
    agent_digest = "sha256:" + "a" * 64
    verifier_digest = "sha256:" + "b" * 64
    manifest = {
        "schema_version": "swebench-verified-paired-image-v1",
        "dataset_type": "swebench-verified",
        "images": {
            "source:latest": {
                "instance_ids": ["django__django-1"],
                "agent_reference": "registry/agent:tag",
                "agent_digest": agent_digest,
                "verifier_reference": "registry/verifier:tag",
                "verifier_digest": verifier_digest,
                "isolation_verified": True,
                "offline_verification": {"passed": True, "network_blocked": True},
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    resolved = resolve_prepared_image(path, "source:latest", "django__django-1")
    assert resolved["prepared_reference"] == f"registry/agent:tag@{agent_digest}"
    assert resolved["verifier_prepared_reference"] == (
        f"registry/verifier:tag@{verifier_digest}"
    )


def test_swesmith_security_rejects_escape_surfaces():
    runtime = DockerRuntime.__new__(DockerRuntime)
    runtime.backend = "docker"
    runtime.network_mode = "none"
    runtime._validate_swesmith_container_security({})

    with pytest.raises(RuntimeError, match="privileged"):
        runtime._validate_swesmith_container_security({"privileged": True})
    with pytest.raises(RuntimeError, match="pid_mode"):
        runtime._validate_swesmith_container_security({"pid_mode": "host"})
    with pytest.raises(RuntimeError, match="host volumes"):
        runtime._validate_swesmith_container_security({"volumes": {"/": {}}})


def test_offline_prompt_declares_policy_boundary():
    assert PROMPT_VERSION == "leaf-prompt-v2"
    assert "POLICY_BLOCKED" in R2E_OFFLINE_SYSTEM_PROMPT
    assert "Network access" in R2E_OFFLINE_SYSTEM_PROMPT
    assert "reflog" in R2E_OFFLINE_SYSTEM_PROMPT
