"""Fresh-container verification for SWE-bench Verified Leaf trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import re
import shlex
import time
from typing import Any

from r2egym.agenthub.runtime.docker import DockerRuntime


class SWEbenchPatchVerificationError(RuntimeError):
    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class SWEbenchPatchVerificationResult:
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


_PROTECTED_PATHS = {"run_tests.sh", "expected_test_output.json"}
_PROTECTED_PARTS = {".git", ".eval", "r2e_tests", "hidden_tests", "private_tests"}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_offline_test_spec(entry: dict[str, Any]) -> str:
    """Serialize the grading-only TestSpec without any upstream network fetches."""
    def test_ids(key: str) -> list[str]:
        value = entry.get(key, [])
        return json.loads(value) if isinstance(value, str) else list(value)

    return json.dumps({
        "instance_id": str(entry["instance_id"]),
        "repo": str(entry["repo"]),
        "version": str(entry.get("version") or ""),
        "repo_script_list": [],
        "eval_script_list": [],
        "env_script_list": [],
        "arch": "arm64" if platform.machine() in {"aarch64", "arm64"} else "x86_64",
        "FAIL_TO_PASS": test_ids("FAIL_TO_PASS"),
        "PASS_TO_PASS": test_ids("PASS_TO_PASS"),
        "language": "python",
        "docker_specs": {},
        "namespace": None,
    })


def _validate_patch_paths(patch: str) -> None:
    saw_header = False
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        saw_header = True
        try:
            fields = shlex.split(line)
        except ValueError as exc:
            raise SWEbenchPatchVerificationError(str(exc), "patch_invalid") from exc
        if len(fields) != 4:
            raise SWEbenchPatchVerificationError("malformed diff header", "patch_invalid")
        for value, prefix in ((fields[2], "a/"), (fields[3], "b/")):
            if not value.startswith(prefix):
                raise SWEbenchPatchVerificationError("non-relative diff path", "patch_path_rejected")
            path = value[2:]
            parts = path.split("/")
            if (
                not path
                or path.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
                or any(part.lower() in _PROTECTED_PARTS for part in parts)
                or parts[-1].lower() in _PROTECTED_PATHS
            ):
                raise SWEbenchPatchVerificationError(
                    f"patch targets protected path: {path}", "patch_path_rejected"
                )
    if patch.strip() and not saw_header:
        raise SWEbenchPatchVerificationError("patch has no git diff headers", "patch_invalid")


class SWEbenchPatchVerifier:
    def __init__(self, *, reward_timeout: int = 1800, backend: str = "docker"):
        self.reward_timeout = reward_timeout
        self.backend = backend

    def verify(self, *, entry: dict[str, Any], patch: str,
               verifier_image: str, verifier_digest: str) -> SWEbenchPatchVerificationResult:
        started = time.monotonic()
        runtime = None
        container_id = None
        applied = False
        try:
            if self.backend != "docker":
                raise SWEbenchPatchVerificationError("Docker is required", "unsupported_backend")
            if not _DIGEST.fullmatch(verifier_digest):
                raise SWEbenchPatchVerificationError("invalid verifier digest", "image_digest_invalid")
            _validate_patch_paths(patch)
            immutable = f"{verifier_image}@{verifier_digest}"
            row = {k: v for k, v in entry.items()
                   if k not in {"_prepared_image_manifest", "_require_prepared_image"}}
            row["docker_image"] = immutable
            row["_network_mode"] = "none"
            # The verifier image already embeds /run_tests.sh.  Supplying a
            # minimal TestSpec prevents the upstream SWE-bench constructor from
            # fetching requirements from GitHub while still using the official
            # log parser and grading logic.
            row["make_test_spec"] = build_offline_test_spec(row)
            runtime = DockerRuntime(ds=row, docker_image=immutable,
                                    network_mode="none", backend="docker")
            container_id = getattr(runtime.container, "id", None)
            if runtime.container.attrs.get("HostConfig", {}).get("NetworkMode") != "none":
                raise SWEbenchPatchVerificationError("verifier is not offline", "network_not_isolated")
            status, code = runtime.run(
                "git -C /testbed status --porcelain --untracked-files=no", timeout=30
            )
            if str(code) not in {"0", ""} or status.strip():
                raise SWEbenchPatchVerificationError("verifier baseline is not clean", "baseline_not_clean")
            if patch.strip():
                _, code = runtime.apply_patch(patch, check=True)
                if str(code) not in {"0", ""}:
                    raise SWEbenchPatchVerificationError("patch failed git apply --check", "patch_apply_check_failed")
                _, code = runtime.apply_patch(patch)
                if str(code) not in {"0", ""}:
                    raise SWEbenchPatchVerificationError("patch failed to apply", "patch_apply_failed")
                applied = True
            reward, output = runtime._calculate_reward_swebench(
                get_test_output=True, timeout=self.reward_timeout
            )
            return SWEbenchPatchVerificationResult(
                float(reward), str(output), verifier_image, verifier_digest,
                container_id, "none", applied, "completed", None, None,
                time.monotonic() - started,
            )
        except SWEbenchPatchVerificationError as exc:
            return SWEbenchPatchVerificationResult(
                0.0, "", verifier_image, verifier_digest, container_id, "none",
                applied, "error", exc.error_code, f"{type(exc).__name__}: {exc}",
                time.monotonic() - started,
            )
        except Exception as exc:
            return SWEbenchPatchVerificationResult(
                0.0, "", verifier_image, verifier_digest, container_id, "none",
                applied, "error", "verifier_infrastructure_error",
                f"{type(exc).__name__}: {exc}", time.monotonic() - started,
            )
        finally:
            if runtime is not None:
                runtime.close()
