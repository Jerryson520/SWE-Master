"""Strict network-none verification for prepared SWE-smith images."""

from __future__ import annotations

import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from swebench.harness.constants import (
    FAIL_TO_PASS,
    PASS_TO_PASS,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_resolution_status
from swesmith.harness.grading import get_eval_tests_report, test_failed, test_passed
from swesmith.profiles import registry as swesmith_registry
from swesmith.profiles.golang import GoProfile

from .swesmith import ImageBuildContext, VerificationContext, hooks_for
from .sanitization import SWESMITH_SANITIZATION_VERSION, build_swesmith_sanitize_command


class OfflineVerificationError(RuntimeError):
    def __init__(self, message: str, reason_code: str = "offline_verification_error"):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    test_collection_passed: bool
    network_blocked: bool
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "test_collection_passed": self.test_collection_passed,
            "network_blocked": self.network_blocked,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class TestRunResult:
    status_map: dict[str, str]
    exit_code: int
    parsed_test_count: int
    mode: str
    timed_out: bool = False


def _exec_result(container, command: str, timeout: int | None = None) -> tuple[int, str]:
    bounded = (
        f"timeout --signal=KILL {int(timeout)} bash -lc {shlex.quote(command)}"
        if timeout
        else command
    )
    result = container.exec_run(["/bin/bash", "-lc", bounded], demux=True)
    stdout, stderr = result.output if isinstance(result.output, tuple) else (result.output, b"")
    text = (stdout or b"").decode(errors="replace") + (stderr or b"").decode(errors="replace")
    return int(result.exit_code), text


def _exec(container, command: str, timeout: int | None = None) -> str:
    exit_code, text = _exec_result(container, command, timeout)
    if exit_code != 0:
        raise OfflineVerificationError(
            f"verification command failed ({exit_code}): {command}\n{text[-4000:]}"
        )
    return text


def _runtime_command(task: dict) -> tuple[Any, str]:
    profile = swesmith_registry.get_from_inst(task)
    test_command, _ = profile.get_test_cmd(task)
    if isinstance(profile, GoProfile):
        go_environment = "GOPROXY=off"
        if "denisenkom__go-mssqldb" in str(task.get("repo", "")):
            go_environment += " GOFLAGS=-mod=mod GOWORK=off"
        prefix = (
            'export PATH="/usr/local/go/bin:/usr/local/bin:$PATH" '
            f"{go_environment}; "
        )
    else:
        prefix = (
            'source /opt/miniconda3/bin/activate; conda activate testbed; '
            'export PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/r2egym/wheelhouse '
            'UV_OFFLINE=1 UV_CACHE_DIR=/opt/r2egym/cache/uv; '
        )
    return profile, prefix + "cd /testbed; " + test_command


def _expected_tests(task: dict) -> tuple[list[str], list[str], bool]:
    instance_id = str(task.get("instance_id"))
    f2p = list(task.get(FAIL_TO_PASS) or [])
    p2p = list(task.get(PASS_TO_PASS) or [])
    is_pr_mirror = ".pr_" in instance_id
    if not f2p or (not p2p and not is_pr_mirror):
        raise OfflineVerificationError(
            f"{instance_id}: expected non-empty F2P"
            f"{' and P2P' if not is_pr_mirror else ''}; got {len(f2p)} and {len(p2p)}",
            "expected_tests_empty",
        )
    return f2p, p2p, is_pr_mirror


def _parse_complete(profile, output: str, expected: list[str], instance_id: str) -> dict[str, str]:
    status_map = profile.log_parser(output)
    if not status_map:
        raise OfflineVerificationError(
            f"{instance_id}: test parser returned no tests", "parser_empty"
        )
    missing = [case for case in expected if case not in status_map]
    if missing:
        raise OfflineVerificationError(
            f"{instance_id}: parser missing {len(missing)} expected tests: {missing[:10]}",
            "expected_test_missing",
        )
    return status_map


def _assert_bug_state(status_map: dict[str, str], f2p: list[str], p2p: list[str], instance_id: str) -> None:
    wrong_f2p = [case for case in f2p if not test_failed(case, status_map)]
    wrong_p2p = [case for case in p2p if not test_passed(case, status_map)]
    if wrong_f2p or wrong_p2p:
        raise OfflineVerificationError(
            f"{instance_id}: invalid bug baseline; F2P-not-failing={wrong_f2p[:10]}, "
            f"P2P-not-passing={wrong_p2p[:10]}",
            "invalid_bug_baseline",
        )


def _assert_fixed_state(
    status_map: dict[str, str],
    f2p: list[str],
    p2p: list[str],
    instance_id: str,
) -> None:
    report = get_eval_tests_report(
        status_map,
        {FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p},
    )
    resolution = get_resolution_status(report)
    if resolution != ResolvedStatus.FULL.value:
        raise OfflineVerificationError(
            f"{instance_id}: reverse patch was not fully resolved; "
            f"resolution={resolution}, "
            f"F2P-failure={report[FAIL_TO_PASS]['failure'][:10]}, "
            f"P2P-failure={report[PASS_TO_PASS]['failure'][:10]}",
            "invalid_fixed_state",
        )


def _run_tests(container, task: dict, *, timeout: int = 1800) -> TestRunResult:
    instance_id = str(task["instance_id"])
    f2p, p2p, _ = _expected_tests(task)
    profile, command = _runtime_command(task)
    mode = "minimal" if profile.min_testing else "full_suite"
    exit_code, output = _exec_result(container, command, timeout)
    if exit_code in {124, 137}:
        raise OfflineVerificationError(
            f"{instance_id}: tests timed out after {timeout}s", "test_timeout"
        )
    status_map = _parse_complete(profile, output, f2p + p2p, instance_id)
    return TestRunResult(status_map, exit_code, len(status_map), mode)


def _full_suite_diagnostics(
    bug_status: dict[str, str], fixed_status: dict[str, str]
) -> dict[str, Any]:
    passing = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
    failing = {TestStatus.FAILED.value, TestStatus.ERROR.value}
    common = set(bug_status) & set(fixed_status)
    transitions = {"f2p": [], "p2p": [], "f2f": [], "unexpected_p2f": []}
    unclassified = []
    for case in sorted(common):
        before, after = bug_status[case], fixed_status[case]
        if before in failing and after in passing:
            transitions["f2p"].append(case)
        elif before in passing and after in passing:
            transitions["p2p"].append(case)
        elif before in failing and after in failing:
            transitions["f2f"].append(case)
        elif before in passing and after in failing:
            transitions["unexpected_p2f"].append(case)
        else:
            unclassified.append(case)
    bug_only = sorted(set(bug_status) - set(fixed_status))
    fixed_only = sorted(set(fixed_status) - set(bug_status))
    warnings = []
    if transitions["unexpected_p2f"]:
        warnings.append("unexpected_p2f")
    if bug_only:
        warnings.append("bug_only_tests")
    if fixed_only:
        warnings.append("fixed_only_tests")
    if unclassified:
        warnings.append("unclassified_transitions")
    return {
        "transitions": {name: len(cases) for name, cases in transitions.items()},
        "transition_samples": {
            "unexpected_p2f": transitions["unexpected_p2f"][:20],
            "unclassified": unclassified[:20],
        },
        "bug_only_test_count": len(bug_only),
        "bug_only_tests": bug_only[:20],
        "fixed_only_test_count": len(fixed_only),
        "fixed_only_tests": fixed_only[:20],
        "warnings": warnings,
    }


def _verify_task(container, task: dict) -> dict[str, Any]:
    instance_id = str(task["instance_id"])
    ref_name = f"refs/r2egym/instances/{instance_id}"
    ref = shlex.quote(ref_name)
    f2p, p2p, is_pr_mirror = _expected_tests(task)
    _exec(
        container,
        f"cd /testbed; git rev-parse --verify {ref}^{{commit}} >/dev/null; "
        f"git rev-parse --verify {ref}~1^{{commit}} >/dev/null; "
        f"git rev-parse --verify {ref}~2^{{commit}} >/dev/null; "
        f"git checkout --detach {ref}~1; git reset --hard {ref}~1; git clean -fdq; "
        "test -z \"$(git status --porcelain --untracked-files=no)\"",
    )
    bug_run = _run_tests(container, task)
    _assert_bug_state(bug_run.status_map, f2p, p2p, instance_id)

    _exec(
        container,
        f"cd /testbed; git diff --binary {ref}~2 {ref}~1 > /tmp/r2egym-bug.patch; "
        "test -s /tmp/r2egym-bug.patch; git apply --reverse /tmp/r2egym-bug.patch; "
        f"git diff --quiet {ref}~2 --; test -z \"$(git diff --name-only {ref}~2 --)\"",
    )
    fixed_run = _run_tests(container, task)
    _assert_fixed_state(fixed_run.status_map, f2p, p2p, instance_id)
    result = {
        "bug_commit": f"{ref_name}~1",
        "clean_commit": f"{ref_name}~2",
        "f2p": len(f2p),
        "p2p": len(p2p),
        "pr_empty_p2p": is_pr_mirror and not p2p,
        "testing_mode": bug_run.mode,
        "min_testing": bug_run.mode == "minimal",
        "bug_exit_code": bug_run.exit_code,
        "fixed_exit_code": fixed_run.exit_code,
        "bug_parsed_tests": bug_run.parsed_test_count,
        "fixed_parsed_tests": fixed_run.parsed_test_count,
        "bug_state_valid": True,
        "reverse_patch_valid": True,
    }
    if bug_run.mode == "full_suite":
        result.update(_full_suite_diagnostics(bug_run.status_map, fixed_run.status_map))
    else:
        result["warnings"] = []
    return result


def _verify_task_isolated(client, image: str, task: dict) -> tuple[str, dict[str, Any]]:
    """Run functional and Git-reinitialization gates in one disposable container."""
    instance_id = str(task["instance_id"])
    container = client.containers.run(
        image,
        command=["/bin/bash", "-lc", "sleep infinity"],
        detach=True,
        network_mode="none",
    )
    try:
        host_config = container.attrs.get("HostConfig", {})
        if host_config.get("NetworkMode") != "none" or host_config.get("Privileged"):
            raise OfflineVerificationError("unsafe audit container configuration", "container_isolation_failed")
        result = _verify_task(container, task)
        _exec(container, build_swesmith_sanitize_command(instance_id), timeout=120)
        f2p, p2p, _ = _expected_tests(task)
        bug_run = _run_tests(container, task)
        _assert_bug_state(bug_run.status_map, f2p, p2p, instance_id)
        _exec(
            container,
            "cd /testbed; test \"$(git for-each-ref --format='%(refname)')\" = refs/heads/r2egym-baseline; "
            "test \"$(git rev-list --parents -n1 HEAD | wc -w)\" -eq 1; "
            "test -z \"$(git remote)\"; test ! -d .git/logs; "
            "test ! -e .git/objects/info/alternates; test ! -d .git/worktrees; "
            "test -z \"$(git status --porcelain)\"",
        )
        result.update({
            "status": "passed",
            "functional_audit_passed": True,
            "sanitization_passed": True,
            "sanitization_version": SWESMITH_SANITIZATION_VERSION,
            "network_blocked": True,
            "bug_baseline_valid": True,
        })
        return instance_id, result
    except OfflineVerificationError as exc:
        return instance_id, {
            "status": "failed",
            "reason_code": exc.reason_code,
            "error": str(exc),
            "functional_audit_passed": False,
            "sanitization_passed": False,
            "sanitization_version": SWESMITH_SANITIZATION_VERSION,
            "network_blocked": True,
        }
    finally:
        container.remove(force=True)


def verify_image(client, image: str, context: VerificationContext) -> VerificationResult:
    image_object = client.images.get(image)
    labels = image_object.labels or {}
    expected_labels = {
        "org.r2egym.prepare-version": context.spec.prepare_version,
        "org.r2egym.source-digest": context.spec.source_digest,
        "org.r2egym.resume-key": context.spec.resume_key,
    }
    mismatched = {
        key: {"expected": expected, "actual": labels.get(key)}
        for key, expected in expected_labels.items()
        if labels.get(key) != expected
    }
    if mismatched:
        raise OfflineVerificationError(
            f"image label attestation mismatch: {mismatched}", "attestation_error"
        )
    container = client.containers.run(
        image,
        command=["/bin/bash", "-lc", "sleep infinity"],
        detach=True,
        network_mode="none",
    )
    checks: dict[str, Any] = {}
    try:
        checks["attestation"] = True
        dirty = _exec(container, "cd /testbed; git status --porcelain --untracked-files=no")
        if dirty.strip():
            raise OfflineVerificationError(
                f"prepared image has tracked worktree changes: {dirty}", "attestation_error"
            )
        checks["tracked_worktree_clean"] = True
        _exec(container, "rg --version | grep -F 'ripgrep 15.1.0'")
        checks["ripgrep"] = True
        for instance_id in context.spec.instance_ids:
            ref = shlex.quote(f"refs/r2egym/instances/{instance_id}")
            _exec(
                container,
                f"cd /testbed; git rev-parse --verify {ref}^{{commit}} >/dev/null; "
                f"git rev-parse --verify {ref}~1^{{commit}} >/dev/null; "
                f"git rev-parse --verify {ref}~2^{{commit}} >/dev/null",
            )
        checks["git_objects"] = len(context.spec.instance_ids)
        build_context = ImageBuildContext(context.spec, context.test_command)
        for hook in hooks_for(build_context):
            command = hook.verification_command(context)
            if command:
                _exec(container, command, timeout=900)
        checks["dependencies"] = True

        if not context.tasks and not context.image_only:
            raise OfflineVerificationError(
                "strict verification requires task records", "attestation_error"
            )
        task_ids = {str(task.get("instance_id")) for task in context.tasks}
        if task_ids != set(context.spec.instance_ids):
            raise OfflineVerificationError(
                "verification task set does not match prepared image spec", "attestation_error"
            )
        task_checks = {}
        if not context.image_only:
            with ThreadPoolExecutor(max_workers=max(1, context.task_workers)) as pool:
                futures = {
                    pool.submit(_verify_task_isolated, client, image, task): str(task["instance_id"])
                    for task in context.tasks
                }
                for future in as_completed(futures):
                    instance_id, task_result = future.result()
                    task_checks[instance_id] = task_result
        checks["tasks"] = task_checks
        checks["strict_task_count"] = len(task_checks)
        checks["image_only"] = context.image_only
        checks["sanitized_task_count"] = sum(
            task.get("sanitization_passed") is True for task in task_checks.values()
        )

        network_probe = container.exec_run(
            [
                "/bin/bash",
                "-lc",
                "! getent hosts github.com >/dev/null 2>&1 && "
                "! timeout 3 bash -c 'echo >/dev/tcp/1.1.1.1/443' >/dev/null 2>&1",
            ]
        )
        if network_probe.exit_code != 0:
            raise OfflineVerificationError(
                "network-none probe unexpectedly reached DNS or direct IP", "attestation_error"
            )
        checks["network"] = "none"
        return VerificationResult(True, True, True, checks)
    finally:
        container.remove(force=True)
