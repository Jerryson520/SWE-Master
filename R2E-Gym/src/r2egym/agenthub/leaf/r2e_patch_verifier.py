"""Fresh-container patch verification for R2E-Gym-Subset trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import shlex
import time
from typing import Any

from r2egym.agenthub.runtime.docker import DockerRuntime


class R2EPatchVerificationError(RuntimeError):
    """A fail-closed verifier error with a stable machine-readable code."""

    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class R2EPatchVerificationResult:
    reward: float
    test_output: str
    image: str
    digest: str
    container_id: str | None
    network_mode: str
    patch_applied: bool
    status: str
    error_code: str | None
    error: str | None
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROTECTED_PATHS = {
    "run_tests.sh",
    "expected_test_output.json",
}
_PROTECTED_PARTS = {
    ".git",
    ".eval",
    "r2e_tests",
    "hidden_tests",
    "private_tests",
    "secret_cases",
}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _patch_paths(patch: str) -> tuple[str, ...]:
    """Extract paths from the canonical git diff emitted by DockerRuntime."""
    paths: set[str] = set()
    saw_header = False
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        saw_header = True
        try:
            fields = shlex.split(line)
        except ValueError as exc:
            raise R2EPatchVerificationError(str(exc), "patch_invalid") from exc
        if len(fields) != 4:
            raise R2EPatchVerificationError("malformed diff header", "patch_invalid")
        for value, prefix in ((fields[2], "a/"), (fields[3], "b/")):
            if not value.startswith(prefix):
                raise R2EPatchVerificationError("non-relative diff path", "patch_path_rejected")
            paths.add(value[2:])
    if patch.strip() and not saw_header:
        raise R2EPatchVerificationError("patch has no git diff headers", "patch_invalid")
    for path in paths:
        parts = path.split("/")
        if (
            not path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or any(part.lower() in _PROTECTED_PARTS for part in parts)
            or parts[-1].lower() in _PROTECTED_PATHS
        ):
            raise R2EPatchVerificationError(
                f"patch targets protected path: {path}", "patch_path_rejected"
            )
    return tuple(sorted(paths))


class R2EPatchVerifier:
    """Apply one candidate patch and grade it in a pristine verifier image."""

    def __init__(self, *, reward_timeout: int = 300, backend: str = "docker"):
        self.reward_timeout = reward_timeout
        self.backend = backend

    def verify(
        self,
        *,
        entry: dict[str, Any],
        patch: str,
        verifier_image: str,
        verifier_digest: str,
    ) -> R2EPatchVerificationResult:
        started = time.monotonic()
        runtime = None
        container_id = None
        patch_applied = False
        try:
            if self.backend != "docker":
                raise R2EPatchVerificationError(
                    "R2E paired verification currently requires Docker", "unsupported_backend"
                )
            if not _DIGEST.fullmatch(verifier_digest):
                raise R2EPatchVerificationError("invalid verifier digest", "image_digest_invalid")
            expected_reference = f"{verifier_image}@{verifier_digest}"
            _patch_paths(patch)
            row = {
                key: value
                for key, value in entry.items()
                if key not in {"_prepared_image_manifest", "_require_prepared_image"}
            }
            row["docker_image"] = expected_reference
            row["_network_mode"] = "none"
            runtime = DockerRuntime(
                ds=row,
                docker_image=expected_reference,
                network_mode="none",
                backend="docker",
            )
            container_id = getattr(runtime.container, "id", None)
            actual_network = runtime.container.attrs.get("HostConfig", {}).get("NetworkMode")
            if actual_network != "none":
                raise R2EPatchVerificationError(
                    f"unexpected verifier network: {actual_network}", "network_not_isolated"
                )
            status, status_code = runtime.run(
                "git diff --quiet HEAD -- . ':(exclude)run_tests.sh' "
                "':(exclude)r2e_tests' ':(exclude)r2e_tests/**'; echo $?",
                timeout=30,
            )
            if str(status_code) not in {"0", ""} or status.strip().splitlines()[-1:] != ["0"]:
                raise R2EPatchVerificationError(
                    f"verifier baseline is not clean: {status[-1000:]}", "baseline_not_clean"
                )
            if patch.strip():
                _, check_code = runtime.apply_patch(patch, check=True)
                if str(check_code) not in {"0", ""}:
                    raise R2EPatchVerificationError(
                        "candidate patch failed git apply --check", "patch_apply_check_failed"
                    )
                _, apply_code = runtime.apply_patch(patch)
                if str(apply_code) not in {"0", ""}:
                    raise R2EPatchVerificationError(
                        "candidate patch failed to apply", "patch_apply_failed"
                    )
                patch_applied = True
            scored = runtime._calculate_reward_r2e(
                get_test_output=True, timeout=self.reward_timeout
            )
            reward, test_output = scored
            if not runtime.parse_logs(test_output):
                raise R2EPatchVerificationError(
                    "verifier test parser returned no tests", "test_parser_empty"
                )
            return R2EPatchVerificationResult(
                float(reward), str(test_output), verifier_image, verifier_digest,
                container_id, "none", patch_applied, "completed", None, None,
                time.monotonic() - started,
            )
        except R2EPatchVerificationError as exc:
            return R2EPatchVerificationResult(
                0.0, "", verifier_image, verifier_digest, container_id, "none",
                patch_applied, "error", exc.error_code,
                f"{type(exc).__name__}: {exc}", time.monotonic() - started,
            )
        except Exception as exc:
            return R2EPatchVerificationResult(
                0.0, "", verifier_image, verifier_digest, container_id, "none",
                patch_applied, "error", "verifier_infrastructure_error",
                f"{type(exc).__name__}: {exc}", time.monotonic() - started,
            )
        finally:
            if runtime is not None:
                runtime.close()
