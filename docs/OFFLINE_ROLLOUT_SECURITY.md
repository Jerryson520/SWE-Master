# Offline rollout security and trajectory integrity

Teacher trajectory collection uses layered controls so model-visible tools cannot access hidden evaluation evidence or fetch external solutions.

## Trust boundaries

1. `r2egym.imageprep` builds immutable agent/verifier image pairs and records their source, digest, task coverage, sanitization version, and offline verification in a manifest.
2. The agent image removes grading scripts, gold patches, private tests, future Git history, remote refs, reflogs, and other solution-bearing state. It runs with Docker network mode `none`.
3. The verifier image is never exposed to the model. Patch verification runs after rollout through `R2EPatchVerifier` or `SWEbenchPatchVerifier`.
4. `AntiHackGuard` inspects every model-controlled tool call before execution. Blocked calls receive a generic `POLICY_BLOCKED` observation; detailed telemetry is written only to a host-side sidecar.
5. The collector validates task-to-image identity, prompt/schema versions, image digests, isolation checks, and resume state before sending a teacher request.

Anti-hack is defence in depth, not a sandbox. Image sanitization and `network_mode=none` are the primary controls. The guard is adapted from Xiaomi MiMoAgent's MIT-licensed anti-hack implementation at commit [`467f0a1`](https://github.com/XiaomiMiMo/mimoagent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/agents/antihack.py).

## Main entry points

- `R2E-Gym/scripts/prepare_swebench_verified_pilot.py`: prepare and audit one SWE-bench agent/verifier image pair.
- `R2E-Gym/scripts/prepare_swebench_verified_parallel.py`: resumable parallel preparation.
- `R2E-Gym/scripts/prepare_r2e_subset_offline_images.py`: deterministic R2E-Gym subset preparation.
- `R2E-Gym/scripts/prepare_swesmith_offline_images.py`: SWE-smith preparation and sanitization.
- `R2E-Gym/scripts/collect_leaf_teacher_rollouts.py`: fail-closed Teacher rollout collection.
- `R2E-Gym/scripts/audit_r2e_policy_v2.py`: post-run policy and trajectory audit.

Every entry point exposes its machine-specific paths through command-line arguments. Use `--help` for the full interface; do not commit registry credentials, model tokens, or machine-local absolute paths.

## Required invariants

- A rollout-eligible manifest must use the current paired-image schema.
- Agent and verifier digests must be immutable `sha256:` references.
- Offline verification must report both `passed=true` and `network_blocked=true`.
- Agent images must not contain grading scripts, gold/test patches, or private verifier assets.
- SWE-smith containers reject privileged mode, host PID/IPC namespaces, mounts, volumes, and devices.
- Anti-hack telemetry must not store raw blocked commands or expose policy details to the model.
