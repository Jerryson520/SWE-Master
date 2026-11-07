"""Docker builder for immutable prepared images."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from r2egym.agenthub.leaf.tools.ripgrep import RG_VERSION, resolve_host_rg

from .base import DatasetImageAdapter, ImageBuildContext, VerificationContext
from .manifest import PreparedImageSpec
from .swesmith import (
    ImageBuildContext as SweSmithImageBuildContext,
    VerificationContext as SweSmithVerificationContext,
    create_prepare_script,
)
from .verifier import verify_image


class ImagePreparationError(RuntimeError):
    pass


# Registries may reject concurrent uploads that finalize the same shared layer.
# Keep expensive image construction and offline verification parallel, but make
# tag publication and its read-back verification a single-file operation.
_PUSH_LOCK = threading.Lock()


def _repository(image: str) -> str:
    last_slash = image.rfind("/")
    last_colon = image.rfind(":")
    return image[:last_colon] if last_colon > last_slash else image


def registry_digest(client, image_reference: str) -> str:
    """Resolve a tag at the registry without pulling its image layers."""
    digest = None
    for attempt in range(6):
        try:
            data = client.images.get_registry_data(image_reference)
            digest = data.attrs.get("Descriptor", {}).get("digest")
            break
        except docker.errors.APIError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code != 429 or attempt == 5:
                raise ImagePreparationError(
                    f"source_image_error: could not inspect {image_reference}: {exc}"
                ) from exc
            delay = (2**attempt) + random.uniform(0.0, 1.0)
            print(
                f"REGISTRY_RETRY {image_reference} status=429 "
                f"attempt={attempt + 1}/6 delay={delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    if not digest or not str(digest).startswith("sha256:"):
        raise ImagePreparationError(
            f"source_image_error: {image_reference} has no registry digest"
        )
    return str(digest)


def _local_image_at_digest(client, repository: str, digest: str):
    immutable_reference = f"{repository}@{digest}"
    try:
        return client.images.get(immutable_reference)
    except docker.errors.ImageNotFound:
        return None


def _pull_with_retry(client, image_reference: str):
    """Pull an image while tolerating transient registry throttling."""
    for attempt in range(6):
        try:
            return client.images.pull(image_reference)
        except docker.errors.APIError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code != 429 or attempt == 5:
                raise
            delay = (2**attempt) + random.uniform(0.0, 1.0)
            print(
                f"REGISTRY_PULL_RETRY {image_reference} status=429 "
                f"attempt={attempt + 1}/6 delay={delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def source_digest(
    client, source_image: str, *, registry_reference: str | None = None
) -> tuple[str, str]:
    """Resolve an immutable source, optionally through a read-only mirror."""
    resolved_reference = registry_reference or source_image
    repository = _repository(resolved_reference)
    expected_digest = registry_digest(client, resolved_reference)
    immutable_reference = f"{repository}@{expected_digest}"
    image = _local_image_at_digest(client, repository, expected_digest)
    if image is not None:
        print(f"LOCAL_HIT {immutable_reference}", flush=True)
        return expected_digest, immutable_reference
    print(f"PULL_MISSING {resolved_reference}", flush=True)
    try:
        image = _pull_with_retry(client, resolved_reference)
    except docker.errors.APIError as exc:
        raise ImagePreparationError(
            f"source_image_error: could not pull {resolved_reference}: {exc}"
        ) from exc
    for value in image.attrs.get("RepoDigests", []) or []:
        name, digest = value.rsplit("@", 1)
        normalized = f"sha256:{digest.removeprefix('sha256:')}"
        if name == repository and normalized == expected_digest:
            return normalized, value
    raise ImagePreparationError(
        f"source_image_error: pulled {resolved_reference} does not match its registry digest"
    )


def remote_digest(
    client, tag_reference: str, *, expected_image_id: str | None = None
) -> str:
    digest = registry_digest(client, tag_reference)
    if expected_image_id is not None:
        try:
            pulled = _pull_with_retry(client, tag_reference)
        except docker.errors.APIError as exc:
            raise ImagePreparationError(
                f"registry_push_error: could not verify pushed image: {exc}"
            ) from exc
        if pulled.id != expected_image_id:
            raise ImagePreparationError(
                "registry_push_error: remote image ID does not match the audited local image"
            )
    return digest


@dataclass
class PreparedImageBuilder:
    client: Any

    def _build_agent_image(
        self, spec: PreparedImageSpec, verifier_local_tag: str, *, push: bool
    ) -> dict[str, Any]:
        """Derive an agent-only image and prove that grading assets are absent."""
        agent_tag = (spec.prepared_tag[:121].rstrip("-") + "-agent")
        local_tag = f"r2egym-local/r2e-gym-subset-agent:{agent_tag}"
        cleanup = r'''#!/usr/bin/env bash
set -euo pipefail
rm -rf /r2e_tests /root/r2e_tests /testbed/r2e_tests
rm -f /root/run_tests.sh /testbed/run_tests.sh
rm -f /root/expected_test_output.json /testbed/expected_test_output.json
rm -rf /root/.eval /testbed/.eval /opt/r2egym/verifier
git -C /testbed add -A
TREE="$(git -C /testbed write-tree)"
PARENT="$(git -C /testbed rev-parse HEAD^{commit})"
AGENT_BASE="$(printf '%s\n' 'R2EGym agent-only baseline' | git -C /testbed -c user.name=R2EGym -c user.email=r2egym@localhost commit-tree "$TREE" -p "$PARENT")"
git -C /testbed update-ref refs/heads/r2egym-baseline "$AGENT_BASE"
git -C /testbed reset --mixed "$AGENT_BASE" >/dev/null
git -C /testbed reflog expire --expire=now --all
git -C /testbed gc --prune=now
printf '%s\n' 'r2e-gym-subset-agent-v1' > /opt/r2egym/dataset-adapter
test -z "$(git -C /testbed status --porcelain --untracked-files=no)"
'''
        with tempfile.TemporaryDirectory(prefix="r2egym-agent-image-") as directory:
            root = Path(directory)
            (root / "cleanup.sh").write_text(cleanup)
            (root / "cleanup.sh").chmod(0o755)
            (root / "Dockerfile").write_text(
                f"FROM {verifier_local_tag}\n"
                "USER root\nCOPY cleanup.sh /tmp/r2egym-agent-cleanup.sh\n"
                "RUN /tmp/r2egym-agent-cleanup.sh && rm -f /tmp/r2egym-agent-cleanup.sh\n"
                "WORKDIR /testbed\n"
            )
            self.client.images.build(path=str(root), tag=local_tag, rm=True, forcerm=True)
        container = self.client.containers.run(
            local_tag, command=["/bin/bash", "-lc", "sleep infinity"],
            detach=True, network_mode="none",
        )
        try:
            check = container.exec_run([
                "/bin/bash", "-lc",
                "set -e; for path in /r2e_tests /root/r2e_tests /testbed/r2e_tests "
                "/root/run_tests.sh /testbed/run_tests.sh /root/expected_test_output.json "
                "/testbed/expected_test_output.json; do "
                "test ! -e \"$path\" && test ! -L \"$path\"; done; "
                "test -z \"$(git -C /testbed status --porcelain --untracked-files=no)\"; "
                "test \"$(cat /opt/r2egym/dataset-adapter)\" = r2e-gym-subset-agent-v1"
            ])
            if int(check.exit_code) != 0:
                raise ImagePreparationError(
                    "agent_isolation_error: " + check.output.decode(errors="replace")[-4000:]
                )
        finally:
            container.remove(force=True)
        image = self.client.images.get(local_tag)
        remote_tag = f"{spec.prepared_repository}:{agent_tag}"
        image.tag(spec.prepared_repository, agent_tag)
        digest = None
        if push:
            with _PUSH_LOCK:
                errors = [
                    event["error"] for event in self.client.images.push(
                        spec.prepared_repository, tag=agent_tag, stream=True, decode=True
                    ) if event.get("error")
                ]
                if errors:
                    raise ImagePreparationError("registry_push_error: " + "; ".join(errors))
                digest = remote_digest(self.client, remote_tag, expected_image_id=image.id)
        return {
            "agent_reference": remote_tag,
            "agent_digest": digest,
            "agent_image_id": image.id,
            "agent_local_reference": local_tag,
            "isolation_verified": True,
        }

    def build_and_verify(
        self,
        spec: PreparedImageSpec,
        *,
        source_reference: str,
        test_command: str,
        tasks: list[dict[str, Any]],
        push: bool,
        adapter: DatasetImageAdapter | None = None,
    ) -> dict[str, Any]:
        namespace = (
            "swe-smith"
            if spec.dataset_type == "swesmith"
            else spec.dataset_type.replace("_", "-")
        )
        local_tag = f"r2egym-local/{namespace}-offline:{spec.prepared_tag}"
        task_tuple = tuple(tasks)
        public_metadata: dict[str, Any] = {}
        if adapter is None:
            prepare_context = SweSmithImageBuildContext(spec, test_command)
            prepare_script = create_prepare_script(prepare_context)
        else:
            prepare_context = ImageBuildContext(spec, task_tuple, test_command)
            prepare_script = adapter.create_prepare_script(prepare_context)
        print(f"BUILD {spec.source_image} -> {local_tag}", flush=True)
        with tempfile.TemporaryDirectory(prefix="r2egym-imageprep-") as directory:
            root = Path(directory)
            shutil.copy2(resolve_host_rg(), root / "rg")
            (root / "rg").chmod(0o755)
            (root / "instances.txt").write_text("\n".join(spec.instance_ids) + "\n")
            (root / "prepare.sh").write_text(prepare_script)
            (root / "prepare.sh").chmod(0o755)
            labels = {
                "org.r2egym.prepare-version": spec.prepare_version,
                "org.r2egym.source-digest": spec.source_digest,
                "org.r2egym.resume-key": spec.resume_key,
            }
            if spec.dataset_type == "swesmith":
                labels["org.r2egym.sanitization-version"] = spec.sanitization_version
            dockerfile = f"""FROM {source_reference}
USER root
COPY rg /usr/local/bin/rg
COPY instances.txt /opt/r2egym/instances.txt
COPY prepare.sh /opt/r2egym/prepare.sh
ARG PIP_INDEX_URL
ARG UV_DEFAULT_INDEX
RUN chmod 0755 /usr/local/bin/rg /opt/r2egym/prepare.sh && /opt/r2egym/prepare.sh && rm -f /opt/r2egym/prepare.sh /opt/r2egym/instances.txt
ENV PATH=/usr/local/go/bin:/opt/miniconda3/envs/testbed/bin:/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \\
    PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/r2egym/wheelhouse \\
    UV_OFFLINE=1 UV_CACHE_DIR=/opt/r2egym/cache/uv GOPROXY=off
WORKDIR /testbed
"""
            (root / "Dockerfile").write_text(dockerfile)
            proxy = os.environ.get("R2EGYM_IMAGEPREP_PROXY", "").strip()
            build_options: dict[str, Any] = {}
            if proxy:
                # Docker treats these predefined proxy build arguments specially:
                # RUN steps inherit them, but they are not persisted in the final
                # image configuration or ordinary build history. Host networking
                # also lets an ECS build reach a loopback-only SSH proxy tunnel.
                build_options["buildargs"] = {
                    "HTTP_PROXY": proxy,
                    "HTTPS_PROXY": proxy,
                    "http_proxy": proxy,
                    "https_proxy": proxy,
                }
                build_options["network_mode"] = "host"
                print("PROXY_ENABLED image preparation downloads", flush=True)
            pip_index = os.environ.get(
                "R2EGYM_IMAGEPREP_PIP_INDEX_URL",
                "https://pypi.org/simple" if proxy else "",
            ).strip()
            if pip_index:
                build_options.setdefault("buildargs", {})["PIP_INDEX_URL"] = pip_index
                build_options["buildargs"]["UV_DEFAULT_INDEX"] = pip_index
                print("PUBLIC_INDEX_ENABLED image preparation downloads", flush=True)
            try:
                self.client.images.build(
                    path=str(root),
                    tag=local_tag,
                    rm=True,
                    forcerm=True,
                    labels=labels,
                    **build_options,
                )
            except docker.errors.BuildError as exc:
                tail = "\n".join(str(item) for item in list(exc.build_log)[-20:])
                raise ImagePreparationError(f"dependency_prepare_error: {tail}") from exc
            print(f"AUDIT {local_tag}", flush=True)
            if adapter is None:
                verification = verify_image(
                    self.client,
                    local_tag,
                    SweSmithVerificationContext(
                        spec, test_command, task_tuple, image_only=True
                    ),
                )
            else:
                verification = adapter.verify_image(
                    self.client,
                    local_tag,
                    VerificationContext(spec, task_tuple, test_command),
                )
            verification_data = verification.to_dict()
            if spec.dataset_type == "swesmith":
                public_metadata["sanitization_version"] = spec.sanitization_version
                public_metadata["agent_exposable"] = False
                public_metadata["isolation_verified"] = False
                public_metadata["sanitized_instance_ids"] = []
            if spec.dataset_type == "r2e-gym-subset":
                task_results = verification_data.get("task_results") or []
                if task_results:
                    public_metadata["source_base_sha"] = task_results[0].get("source_base_sha")
                    public_metadata["prepared_base_sha"] = task_results[0].get("prepared_base_sha")
                    public_metadata["base_tree_digest"] = task_results[0].get("base_tree_digest")
                public_metadata["sanitization_version"] = spec.sanitization_version
                public_metadata["flattened"] = False
                public_metadata["layer_sanitized"] = False
                public_metadata["security_checks"] = verification_data.get("security_checks", {})
            # The exact audited image is tagged and pushed. Verification state lives
            # in the host manifest, so no post-audit layer changes the image ID.
            image = self.client.images.get(local_tag)
            audited_image_id = image.id
            image.tag(spec.prepared_repository, spec.prepared_tag)
            prepared_digest = None
            if push:
                print(f"PUSH_WAIT {spec.remote_tag}", flush=True)
                with _PUSH_LOCK:
                    print(f"PUSH {spec.remote_tag}", flush=True)
                    errors = []
                    for event in self.client.images.push(
                        spec.prepared_repository,
                        tag=spec.prepared_tag,
                        stream=True,
                        decode=True,
                    ):
                        if event.get("error"):
                            errors.append(event["error"])
                    if errors:
                        raise ImagePreparationError(
                            "registry_push_error: " + "; ".join(errors)
                        )
                    prepared_digest = remote_digest(
                        self.client,
                        spec.remote_tag,
                        expected_image_id=audited_image_id,
                    )
            record = {
                **spec.to_dict(),
                **public_metadata,
                "source_reference": source_reference,
                "prepared_tag_reference": spec.remote_tag,
                "prepared_digest": prepared_digest,
                "audited_image_id": audited_image_id,
                "local_image_reference": local_tag,
                "offline_verification": verification_data,
                "agent_tools": {"ripgrep": RG_VERSION},
                "status": "published" if push else "verified-local",
            }
            if spec.dataset_type == "r2e-gym-subset":
                agent = self._build_agent_image(spec, local_tag, push=push)
                record.update(agent)
                record.update({
                    "verifier_reference": spec.remote_tag,
                    "verifier_digest": prepared_digest,
                    "verifier_image_id": audited_image_id,
                    "verifier_local_reference": local_tag,
                })
            return record
