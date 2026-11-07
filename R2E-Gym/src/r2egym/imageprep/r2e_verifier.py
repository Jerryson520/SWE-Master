"""Offline semantic and Git-sanitization gate for R2E prepared images."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import shlex
from typing import Any

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.execution_log_parser import decolor_dict_keys

from .base import VerificationContext


class R2EVerificationError(RuntimeError):
    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class R2EVerificationResult:
    passed: bool
    network_blocked: bool
    task_results: tuple[dict[str, Any], ...]
    security_checks: dict[str, bool]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "network_blocked": self.network_blocked,
            "task_results": list(self.task_results),
            "security_checks": self.security_checks,
            "warnings": list(self.warnings),
        }


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    value = decolor_dict_keys(value)
    return {key.split(" - ")[0]: value[key] for key in sorted(value)}


def _static_security_checks(
    client: Any, image: str, task: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    commit = ParsedCommit(**json.loads(task["parsed_commit_content"]))
    container = client.containers.run(
        image,
        command=["/bin/bash", "-lc", "sleep infinity"],
        detach=True,
        network_mode="none",
    )
    try:
        def run(command: str, *, check: bool = True) -> tuple[int, str]:
            result = container.exec_run(["/bin/bash", "-lc", command])
            output = result.output.decode(errors="replace")
            if check and int(result.exit_code) != 0:
                raise R2EVerificationError(output[-4000:], "security_check_failed")
            return int(result.exit_code), output

        if container.attrs.get("HostConfig", {}).get("NetworkMode") != "none":
            raise R2EVerificationError("audit container is not offline", "network_not_isolated")
        _, raw_metadata = run("cat /opt/r2egym/r2e-security.json")
        metadata = json.loads(raw_metadata)
        if metadata.get("source_base_selector") != commit.old_commit_hash:
            raise R2EVerificationError("source base SHA mismatch", "base_commit_mismatch")
        source_base_sha = metadata.get("source_base_sha")
        prepared_base_sha = metadata.get("prepared_base_sha")
        _, head = run("git -C /testbed rev-parse HEAD^{commit}")
        if head.strip() != prepared_base_sha:
            raise R2EVerificationError("prepared HEAD mismatch", "base_commit_mismatch")
        if prepared_base_sha != source_base_sha:
            _, parents = run("git -C /testbed rev-list --parents -n 1 HEAD")
            if len(parents.split()) != 2 or parents.split()[1] != source_base_sha:
                raise R2EVerificationError("prepared environment parent mismatch", "git_history_leak")
        _, commits_after_base = run(
            "git -C /testbed rev-list --all --not " + shlex.quote(source_base_sha)
        )
        unexpected = [value for value in commits_after_base.splitlines() if value]
        expected_after_base = [] if prepared_base_sha == source_base_sha else [prepared_base_sha]
        if unexpected != expected_after_base:
            raise R2EVerificationError(
                f"unexpected commits after base: {unexpected}", "git_history_leak"
            )
        _, branches = run("git -C /testbed for-each-ref refs/heads --format='%(refname)'")
        if branches.strip() != "refs/heads/r2egym-baseline":
            raise R2EVerificationError(f"unexpected branches: {branches}", "git_ref_leak")
        _, tags = run("git -C /testbed for-each-ref refs/tags --format='%(refname:strip=2)'")
        actual_tags = [value for value in tags.splitlines() if value]
        expected_tags = metadata.get("preserved_ancestor_tags") or []
        if actual_tags != expected_tags:
            raise R2EVerificationError(f"unexpected tags: {actual_tags}", "git_ref_leak")
        for expected_tag in expected_tags:
            _, tag_target = run(f"git -C /testbed rev-parse {shlex.quote(expected_tag)}^{{commit}}")
            tag_code, _ = run(
                "git -C /testbed merge-base --is-ancestor "
                f"{shlex.quote(tag_target.strip())} {shlex.quote(source_base_sha)}",
                check=False,
            )
            if tag_code != 0:
                raise R2EVerificationError("tag points beyond base", "git_ref_leak")
        _, remotes = run("git -C /testbed remote")
        if remotes.strip():
            raise R2EVerificationError("prepared repository has remotes", "git_ref_leak")
        solution_code, _ = run(
            "git -C /testbed cat-file -e "
            f"{shlex.quote(commit.new_commit_hash)}^{{commit}}",
            check=False,
        )
        if solution_code == 0:
            raise R2EVerificationError("solution commit is accessible", "solution_leak")
        run("test -f /testbed/run_tests.sh || test -f /root/run_tests.sh")
        run("test -d /r2e_tests || test -d /testbed/r2e_tests || test -d /root/r2e_tests")
        run("test ! -s /testbed/.git/logs/HEAD")
        run("test -z \"$(find /testbed/.git/hooks -mindepth 1 -maxdepth 1 -type f -print -quit)\"")
        return {
            "base_history_preserved": True,
            "future_history_removed": True,
            "solution_unreachable": True,
            "refs_removed": True,
            "ancestor_tags_preserved": True,
            "reflog_removed": True,
            "tests_preserved": True,
            "runtime_isolated": False,
            "layer_sanitized": False,
        }, metadata
    finally:
        container.remove(force=True)


def _verify_task(image: str, task: dict[str, Any], timeout: int) -> dict[str, Any]:
    expected_raw = json.loads(task["expected_output_json"])
    if not isinstance(expected_raw, dict) or not expected_raw:
        raise R2EVerificationError("expected test map is empty", "expected_tests_empty")
    expected = _normalize(expected_raw)
    commit = ParsedCommit(**json.loads(task["parsed_commit_content"]))
    if commit.new_commit_hash != task["commit_hash"]:
        raise R2EVerificationError("commit hash does not match parsed patch", "commit_mismatch")
    row = dict(task)
    row["docker_image"] = image
    row["_network_mode"] = "none"
    runtime = DockerRuntime(
        ds=row,
        docker_image=image,
        network_mode="none",
        repo_path="/testbed",
        logger=logging.getLogger("r2egym.imageprep.r2e_verifier"),
    )
    try:
        network_mode = runtime.container.attrs.get("HostConfig", {}).get("NetworkMode")
        if network_mode != "none":
            raise R2EVerificationError("verification container is not isolated", "network_not_isolated")
        head, head_error = runtime.run("git -C /testbed rev-parse HEAD")
        if str(head_error) not in {"0", ""}:
            raise R2EVerificationError("could not resolve bug baseline", "invalid_bug_baseline")
        bug_reward, bug_output = runtime._calculate_reward_r2e(
            get_test_output=True, timeout=timeout
        )
        bug_status = _normalize(runtime.parse_logs(bug_output))
        if not bug_status:
            raise R2EVerificationError("bug test parser returned no tests", "parser_empty")
        if bug_reward != 0:
            raise R2EVerificationError("bug baseline unexpectedly passes", "invalid_bug_baseline")
        patch = commit.get_patch()
        if not patch.strip():
            raise R2EVerificationError("ground-truth patch is empty", "patch_empty")
        _, apply_error = runtime.apply_patch(patch)
        if str(apply_error) not in {"0", ""}:
            raise R2EVerificationError("ground-truth patch failed to apply", "patch_apply_failed")
        fixed_reward, fixed_output = runtime._calculate_reward_r2e(
            get_test_output=True, timeout=timeout
        )
        fixed_status = _normalize(runtime.parse_logs(fixed_output))
        if not fixed_status:
            raise R2EVerificationError("fixed test parser returned no tests", "parser_empty")
        if fixed_status != expected:
            raise R2EVerificationError(
                "fixed test status does not match expected_output_json",
                "expected_output_mismatch",
            )
        if fixed_reward != 1:
            raise R2EVerificationError("fixed state reward is not one", "invalid_fixed_state")
        return {
            "instance_id": task["instance_id"],
            "baseline_head": head.strip(),
            "bug_reward": bug_reward,
            "fixed_reward": fixed_reward,
            "bug_parsed_tests": len(bug_status),
            "fixed_parsed_tests": len(fixed_status),
            "expected_tests": len(expected),
            "patch_applied": True,
        }
    finally:
        runtime.close()


def verify_r2e_image(
    client: Any,
    image: str,
    context: VerificationContext,
    *,
    timeout: int = 300,
) -> R2EVerificationResult:
    if len(context.tasks) != 1:
        raise R2EVerificationError("R2E verification requires one task", "attestation_error")
    checks, metadata = _static_security_checks(client, image, context.tasks[0])
    result = _verify_task(image, context.tasks[0], timeout)
    result.update({
        "source_base_sha": metadata["source_base_sha"],
        "prepared_base_sha": metadata["prepared_base_sha"],
        "base_tree_digest": metadata["base_tree_digest"],
    })
    return R2EVerificationResult(True, True, (result,), checks)
