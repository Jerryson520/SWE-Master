#!/usr/bin/env python3
"""Generate Leaf trajectories with an OpenAI-compatible teacher.

The generic ``--input`` mode accepts an existing JSON/JSONL/Parquet task file.
The fail-closed ``--r2e-source-dir`` mode deterministically reloads the exact
R2E-Gym subset used to build a prepared-image manifest and validates every
task/image association before any model request or container launch.
"""

import argparse
import fcntl
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoTokenizer

from r2egym.agenthub.leaf import TOOL_DEFINITIONS, SYSTEM_PROMPT, AgentAction, RepoToolEnvironment, ToolCall
from r2egym.agenthub.leaf.protocol import PROMPT_VERSION, R2E_OFFLINE_SYSTEM_PROMPT
from r2egym.agenthub.leaf.r2e_patch_verifier import R2EPatchVerifier
from r2egym.agenthub.leaf.swebench_patch_verifier import (
    SWEbenchPatchVerifier,
    build_offline_test_spec,
)
from r2egym.imageprep.manifest import R2E_MANIFEST_SCHEMA_VERSION, load_manifest, resolve_prepared_image
from r2egym.imageprep.r2e_tasks import load_r2e_tasks


SCHEMA_VERSION = "leaf-openai-tools-v2"
PROGRESS_LOCK = threading.Lock()
R2E_PREPARE_VERSION = "offline-v2"
R2E_SANITIZATION_VERSION = "r2e-base-history-v1"
R2E_REQUIRED_SECURITY_CHECKS = (
    "base_history_preserved",
    "future_history_removed",
    "solution_unreachable",
    "refs_removed",
    "reflog_removed",
    "ancestor_tags_preserved",
    "tests_preserved",
)


def progress(message):
    with PROGRESS_LOCK:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def acquire_run_lock(path: str | Path):
    """Hold a non-blocking host lock for one collector/output pair."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"another collector owns the run lock: {destination}") from None
    return stream


def entry_id(entry):
    return (
        entry.get("instance_id")
        or entry.get("id")
        or entry.get("commit_hash")
        or entry.get("docker_image")
        or entry.get("image_name")
    )


def load_entries(path: str):
    suffix = Path(path).suffix.lower()
    kind = "parquet" if suffix == ".parquet" else "json"
    rows = list(load_dataset(kind, data_files=path, split="train"))
    entries = []
    for row in rows:
        value = row.get("extra_info", row)
        if isinstance(value, str):
            value = json.loads(value)
        entries.append(value)
    return entries


def validate_r2e_collection_manifest(
    manifest_path: str,
    entries: list[dict],
    fingerprints: dict[str, str],
    *,
    task_count: int,
    selection_seed: int,
) -> dict:
    """Prove that the selected R2E tasks exactly match the published images."""
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != R2E_MANIFEST_SCHEMA_VERSION:
        raise ValueError("R2E teacher collection requires the paired-image manifest schema")
    if manifest.get("dataset_type") != "r2e-gym-subset":
        raise ValueError("prepared-image manifest is not for R2E-Gym-Subset")
    if manifest.get("prepare_version") != R2E_PREPARE_VERSION:
        raise ValueError(f"R2E teacher collection requires prepare_version={R2E_PREPARE_VERSION}")
    selection = manifest.get("selection") or {}
    expected_selection = {
        "task_count": task_count,
        "selection_seed": selection_seed,
        "fingerprints": fingerprints,
    }
    if selection != expected_selection:
        raise ValueError("R2E dataset selection does not match the prepared-image manifest")
    if len(entries) != task_count:
        raise ValueError(f"R2E loader returned {len(entries)} tasks; expected {task_count}")

    source_images = {str(entry["docker_image"]) for entry in entries}
    manifest_images = set(manifest.get("images", {}))
    if source_images != manifest_images:
        missing = sorted(source_images - manifest_images)
        extra = sorted(manifest_images - source_images)
        raise ValueError(
            "R2E manifest image set differs from selected tasks: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    instance_ids: set[str] = set()
    for entry in entries:
        instance_id = str(entry["instance_id"])
        if instance_id in instance_ids:
            raise ValueError(f"duplicate R2E instance_id: {instance_id}")
        instance_ids.add(instance_id)
        source_image = str(entry["docker_image"])
        resolved = resolve_prepared_image(manifest_path, source_image, instance_id)
        if resolved.get("status") != "published":
            raise ValueError(f"R2E prepared image is not published: {instance_id}")
        if resolved.get("prepare_version") != R2E_PREPARE_VERSION:
            raise ValueError(f"wrong prepare version for {instance_id}")
        if resolved.get("sanitization_version") != R2E_SANITIZATION_VERSION:
            raise ValueError(f"wrong sanitization version for {instance_id}")
        if resolved.get("source_image") != source_image:
            raise ValueError(f"source-image identity mismatch for {instance_id}")
        if len(resolved.get("instance_ids", [])) != 1:
            raise ValueError(f"R2E prepared image must cover exactly one task: {instance_id}")
        checks = resolved.get("security_checks") or {}
        failed_checks = [name for name in R2E_REQUIRED_SECURITY_CHECKS if checks.get(name) is not True]
        if failed_checks:
            raise ValueError(
                f"R2E prepared image failed security checks for {instance_id}: "
                + ", ".join(failed_checks)
            )
        if resolved.get("isolation_verified") is not True:
            raise ValueError(f"R2E agent image isolation is not verified: {instance_id}")
    return manifest


def prepull_r2e_paired_images(manifest: dict, entries: list[dict]) -> None:
    """Materialize both immutable roles before any teacher request is sent."""
    client = docker.from_env(timeout=900)
    try:
        references: set[tuple[str, str]] = set()
        for entry in entries:
            record = manifest["images"][str(entry["docker_image"])]
            for role in ("agent", "verifier"):
                references.add((record[f"{role}_reference"], record[f"{role}_digest"]))
        for tag, digest in sorted(references):
            immutable = f"{tag}@{digest}"
            try:
                client.images.get(immutable)
            except docker.errors.ImageNotFound:
                client.images.pull(tag)
                client.images.get(immutable)
    finally:
        client.close()


def message_dict(message):
    result = {"role": "assistant", "content": message.content}
    calls = []
    for call in message.tool_calls or []:
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
        )
    if calls:
        result["tool_calls"] = calls
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


RETRYABLE_FINISH_REASONS = {"insufficient_system_resource", "aborted"}


def prompt_for_run(args):
    """Select the prompt without weakening generic non-R2E collection."""
    if args.swebench_rollout_only:
        return R2E_OFFLINE_SYSTEM_PROMPT, "swebench-rollout"
    if args.r2e_source_dir or args.swebench_verified:
        return R2E_OFFLINE_SYSTEM_PROMPT, "r2e-offline"
    return SYSTEM_PROMPT, "generic"


def trajectory_policy_summary(summary):
    """Keep safe policy metadata in the trajectory; detailed rules stay sidecar-only."""
    return {
        "enabled": bool(summary.get("enabled")),
        "block_count": int(summary.get("block_count", 0)),
        "blocked_tool_call_ids": list(summary.get("blocked_tool_call_ids") or []),
        "telemetry_errors": list(summary.get("telemetry_errors") or []),
    }


def teacher_request_kwargs(args, messages):
    """Build a DeepSeek-compatible OpenAI Chat Completions request."""
    kwargs = {
        "model": args.model,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens_per_turn,
    }
    extra_body = {"top_k": args.top_k}
    if args.thinking_provider == "deepseek" and args.thinking_mode:
        extra_body["thinking"] = {"type": args.thinking_mode}
    elif args.thinking_provider == "qwen" and args.thinking_mode:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.thinking_mode == "enabled"
        }
    kwargs["extra_body"] = extra_body
    if args.reasoning_effort != "none":
        kwargs["reasoning_effort"] = args.reasoning_effort
    return kwargs


def rendered_context_tokens(args, messages):
    """Count the exact Qwen chat-template input before reserving generation space."""
    template_messages = []
    for message in messages:
        normalized = dict(message)
        normalized_calls = []
        for call in message.get("tool_calls", []):
            normalized_call = dict(call)
            function = dict(normalized_call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    function["arguments"] = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    function["arguments"] = {}
            normalized_call["function"] = function
            normalized_calls.append(normalized_call)
        if normalized_calls:
            normalized["tool_calls"] = normalized_calls
        template_messages.append(normalized)
    tokens = args.tokenizer_instance.apply_chat_template(
        template_messages,
        tools=TOOL_DEFINITIONS,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=args.thinking_mode == "enabled",
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    elif isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    return len(tokens)


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("connection", "timeout", "ratelimit"))


def create_teacher_completion(client, args, messages, *, task_id, seed, turn):
    """Retry transient provider failures without duplicating conversation state."""
    for attempt in range(args.api_max_retries + 1):
        try:
            response = client.chat.completions.create(**teacher_request_kwargs(args, messages))
            finish_reason = str(response.choices[0].finish_reason or "").lower()
            if finish_reason not in RETRYABLE_FINISH_REASONS:
                return response
            failure = RuntimeError(f"retryable finish_reason={finish_reason}")
        except Exception as exc:
            if not is_retryable_api_error(exc) or attempt >= args.api_max_retries:
                raise
            failure = exc
        if attempt >= args.api_max_retries:
            raise failure
        delay = min(args.api_retry_max_delay, args.api_retry_base_delay * (2**attempt))
        delay += random.uniform(0, min(1.0, delay * 0.1))
        progress(
            f"RETRY task={task_id} seed={seed} turn={turn} "
            f"attempt={attempt + 1}/{args.api_max_retries} delay={delay:.1f}s "
            f"reason={type(failure).__name__}: {failure}"
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def generate_one(entry, seed, args):
    task_id = entry_id(entry)
    runtime_entry = dict(entry)
    if args.swebench_rollout_only:
        runtime_entry["_swebench_rollout_only"] = True
    if args.swebench_verified:
        runtime_entry["make_test_spec"] = build_offline_test_spec(runtime_entry)
    antihack = {
        "enabled": args.antihack,
        "events_path": args.antihack_events,
        "run_id": args.antihack_run_id,
        "trajectory_id": f"{task_id}:seed:{seed}",
    }
    env_config = {
            **runtime_entry,
            "backend": args.backend,
            "step_timeout": args.step_timeout,
            "reward_timeout": args.reward_timeout,
            "prepared_image_manifest": args.prepared_image_manifest,
            "require_prepared_images": args.require_prepared_images,
            "network_mode": args.network_mode,
            "antihack": antihack,
        }
    env = RepoToolEnvironment.from_dict(env_config)
    started = time.monotonic()
    messages = []
    exit_reason = "unknown"
    error = None
    patch = ""
    reward = 0.0
    test_output = ""
    verifier_info = None
    agent_elapsed_sec = 0.0
    environment = None
    policy = trajectory_policy_summary({})
    system_prompt, prompt_variant = prompt_for_run(args)
    try:
        progress(f"START task={task_id} seed={seed} phase=container")
        if args.swebench_rollout_only:
            for create_attempt in range(3):
                try:
                    with open("/tmp/leaf-swebench-docker-create.lock", "a+") as create_lock:
                        fcntl.flock(create_lock.fileno(), fcntl.LOCK_EX)
                        instruction, _ = env.reset()
                        fcntl.flock(create_lock.fileno(), fcntl.LOCK_UN)
                    break
                except Exception as exc:
                    env.close()
                    if create_attempt == 2:
                        raise
                    progress(
                        f"DOCKER_RETRY task={task_id} seed={seed} "
                        f"attempt={create_attempt + 1}/3 reason={type(exc).__name__}"
                    )
                    time.sleep(2 * (create_attempt + 1))
                    env = RepoToolEnvironment.from_dict(env_config)
        else:
            instruction, _ = env.reset()
        environment = env.environment_info()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
        client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.base_url, timeout=args.request_timeout)
        for turn in range(1, args.max_steps + 1):
            context_tokens = rendered_context_tokens(args, messages)
            if context_tokens > args.context_window - args.context_safety_margin:
                exit_reason = "context_window"
                progress(
                    f"CONTEXT_LIMIT task={task_id} seed={seed} turn={turn} "
                    f"tokens={context_tokens} window={args.context_window} "
                    f"reserve={args.context_safety_margin}"
                )
                break
            response = create_teacher_completion(
                client,
                args,
                messages,
                task_id=task_id,
                seed=seed,
                turn=turn,
            )
            choice = response.choices[0]
            assistant = message_dict(choice.message)
            messages.append(assistant)
            if str(choice.finish_reason or "").lower() in {"length", "max_tokens", "max_new_tokens"}:
                exit_reason = "max_tokens_per_turn"
                break
            calls = []
            for call in assistant.get("tool_calls", []):
                try:
                    arguments = json.loads(call["function"].get("arguments") or "{}")
                    arguments = arguments if isinstance(arguments, dict) else {}
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                calls.append(ToolCall(call["id"], call["function"].get("name", ""), arguments))
            if args.log_turns:
                names = ",".join(call.name for call in calls) or "none"
                progress(
                    f"TURN task={task_id} seed={seed} turn={turn} "
                    f"finish={choice.finish_reason or 'none'} tools={names}"
                )
            action = AgentAction(tuple(calls), assistant.get("content") or "", assistant.get("reasoning_content") or "")
            results, _, done, info = env.step(action)
            for result in results:
                messages.append(
                    {"role": "tool", "tool_call_id": result.tool_call_id, "name": result.name, "content": result.content}
                )
            if done:
                exit_reason = info.get("exit_reason", "agent")
                break
        else:
            exit_reason = "max_turns"
        patch = env.output_patch()
        progress(f"REWARD task={task_id} seed={seed} patch_chars={len(patch)}")
        agent_elapsed_sec = time.monotonic() - started
        if args.swebench_rollout_only:
            env.close()
            verifier_info = {
                "status": "deferred",
                "backend": "official-swebench-harness",
            }
        elif args.r2e_source_dir or args.swebench_verified:
            env.close()
            resolved = resolve_prepared_image(
                args.prepared_image_manifest,
                str(entry.get("docker_image") or entry.get("image_name")),
                str(task_id),
            )
            verifier_class = (
                SWEbenchPatchVerifier if args.swebench_verified else R2EPatchVerifier
            )
            verification = verifier_class(
                reward_timeout=args.reward_timeout, backend=args.backend
            ).verify(
                entry=entry,
                patch=patch,
                verifier_image=resolved["verifier_reference"],
                verifier_digest=resolved["verifier_digest"],
            )
            reward = verification.reward
            test_output = verification.test_output
            verifier_info = verification.to_dict()
            if verification.error:
                error = verification.error
                exit_reason = "verifier_error"
        else:
            scored = env.env.runtime._calculate_reward(
                get_test_output=True, timeout=args.reward_timeout
            )
            if isinstance(scored, tuple):
                reward, test_output = scored
            else:
                reward = scored
    except Exception as exc:
        exit_reason = "environment_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        policy = trajectory_policy_summary(env.antihack_summary())
        env.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": {"version": PROMPT_VERSION, "variant": prompt_variant},
        "instance_id": entry_id(entry),
        "repo": entry.get("repo") or entry.get("repo_name"),
        "image": entry.get("docker_image") or entry.get("image_name"),
        "entry": entry,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "max_tokens_per_turn": args.max_tokens_per_turn,
            "context_window": args.context_window,
            "context_safety_margin": args.context_safety_margin,
            "tokenizer": args.tokenizer,
            "thinking_mode": args.thinking_mode,
            "thinking_provider": args.thinking_provider,
            "reasoning_effort": args.reasoning_effort,
        },
        "seed": seed,
        "messages": messages,
        "tools": TOOL_DEFINITIONS,
        "policy": policy,
        "output_patch": patch,
        "reward": float(reward),
        "test_output": str(test_output),
        "exit_reason": exit_reason,
        "error": error,
        "environment": environment,
        "verifier": verifier_info,
        "agent_elapsed_sec": agent_elapsed_sec,
        "verifier_elapsed_sec": (verifier_info or {}).get("elapsed_sec", 0.0),
        "model_turns": sum(message.get("role") == "assistant" for message in messages),
        "tool_calls": sum(len(message.get("tool_calls", [])) for message in messages),
        "elapsed_sec": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Existing JSON/JSONL/Parquet task file.")
    source.add_argument(
        "--r2e-source-dir",
        help="R2E-Gym-Subset checkout containing data/train-*.parquet shards.",
    )
    parser.add_argument("--task-count", type=int, default=300)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Restrict an --input dataset to these instance IDs.",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock-file", help="Cross-process lock; defaults to <output>.lock.")
    parser.add_argument("--base-url", default=os.environ.get("LEAF_TEACHER_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LEAF_TEACHER_MODEL"))
    parser.add_argument("--api-key-env", default="LEAF_TEACHER_API_KEY")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-tokens-per-turn", type=int, default=8192)
    parser.add_argument("--context-window", type=int, default=81920)
    parser.add_argument("--context-safety-margin", type=int, default=8192)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=int, default=7200)
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Provider thinking mode; defaults to DeepSeek thinking mode.",
    )
    parser.add_argument(
        "--thinking-provider",
        choices=("deepseek", "qwen", "none"),
        default="deepseek",
        help="Provider-specific request field used to control thinking.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "high", "max"),
        default="high",
        help="Provider reasoning effort; defaults to high for coding-agent rollouts.",
    )
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--api-retry-base-delay", type=float, default=2.0)
    parser.add_argument("--api-retry-max-delay", type=float, default=60.0)
    parser.add_argument("--step-timeout", type=int, default=180)
    parser.add_argument("--reward-timeout", type=int, default=1800)
    parser.add_argument("--backend", default="docker")
    parser.add_argument("--prepared-image-manifest")
    parser.add_argument("--require-prepared-images", action="store_true")
    parser.add_argument(
        "--swebench-verified",
        action="store_true",
        help="Use the paired offline Leaf contract for a SWE-bench Verified --input file.",
    )
    parser.add_argument(
        "--swebench-rollout-only",
        action="store_true",
        help=(
            "Run Leaf directly in the official SWE-bench task image, export a patch, "
            "and defer all scoring to the official SWE-bench evaluator."
        ),
    )
    parser.add_argument("--network-mode", choices=("host", "bridge", "none"), default="host")
    parser.add_argument(
        "--prepull-prepared-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-pull both immutable R2E image roles before starting trajectories.",
    )
    parser.add_argument("--log-turns", action="store_true")
    parser.add_argument("--antihack", action="store_true")
    parser.add_argument(
        "--antihack-events",
        help="Host-side anti-hack JSONL sidecar; defaults beside --output when enabled.",
    )
    parser.add_argument("--antihack-run-id")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate R2E task/image coverage without starting containers or calling the teacher.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.context_window <= 0:
        parser.error("--context-window must be positive")
    if not 0 <= args.context_safety_margin < args.context_window:
        parser.error("--context-safety-margin must be in [0, context-window)")
    if args.max_tokens_per_turn > args.context_safety_margin:
        parser.error("--context-safety-margin must reserve max-tokens-per-turn")
    args.tokenizer_instance = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.api_max_retries < 0:
        parser.error("--api-max-retries must be non-negative")
    if args.api_retry_base_delay <= 0 or args.api_retry_max_delay <= 0:
        parser.error("API retry delays must be positive")
    if args.swebench_verified and args.swebench_rollout_only:
        parser.error("choose only one SWE-bench mode")
    if args.swebench_rollout_only:
        if not args.input:
            parser.error("--swebench-rollout-only requires --input")
        if args.prepared_image_manifest or args.require_prepared_images:
            parser.error("rollout-only mode uses official task images, not prepared-image manifests")
        if args.network_mode != "none" or not args.antihack:
            parser.error("SWE-bench rollout-only requires --network-mode none and --antihack")
        entries = load_entries(args.input)
        if args.instance_id:
            wanted = set(args.instance_id)
            entries = [entry for entry in entries if str(entry_id(entry)) in wanted]
            found = {str(entry_id(entry)) for entry in entries}
            if found != wanted:
                parser.error(f"instance IDs not found: {sorted(wanted - found)}")
        client = docker.from_env(timeout=60)
        try:
            missing = []
            for entry in entries:
                image = str(entry.get("docker_image") or entry.get("image_name"))
                try:
                    client.images.get(image)
                except docker.errors.ImageNotFound:
                    missing.append((str(entry_id(entry)), image))
            if missing:
                parser.error(f"official task images missing locally: {len(missing)}")
        finally:
            client.close()
        progress(f"SWE-BENCH ROLLOUT PREFLIGHT passed tasks={len(entries)}")
    elif args.swebench_verified:
        if not args.input:
            parser.error("--swebench-verified requires --input")
        if not args.prepared_image_manifest or not args.require_prepared_images:
            parser.error("SWE-bench Verified requires paired prepared images")
        if args.network_mode != "none" or not args.antihack:
            parser.error("SWE-bench Verified requires --network-mode none and --antihack")
        entries = load_entries(args.input)
        if args.instance_id:
            wanted = set(args.instance_id)
            entries = [entry for entry in entries if str(entry_id(entry)) in wanted]
            found = {str(entry_id(entry)) for entry in entries}
            if found != wanted:
                parser.error(f"instance IDs not found: {sorted(wanted - found)}")
        manifest = load_manifest(args.prepared_image_manifest)
        if manifest.get("dataset_type") != "swebench-verified":
            parser.error("prepared manifest is not for SWE-bench Verified")
        for entry in entries:
            resolved = resolve_prepared_image(
                args.prepared_image_manifest,
                str(entry.get("docker_image") or entry.get("image_name")),
                str(entry_id(entry)),
            )
            if resolved.get("status") != "published":
                parser.error(f"prepared image is not published: {entry_id(entry)}")
        progress(f"SWE-BENCH VERIFIED PREFLIGHT passed tasks={len(entries)}")
        if args.prepull_prepared_images:
            prepull_r2e_paired_images(manifest, entries)
            progress(f"SWE-BENCH VERIFIED PREPULL passed image_pairs={len(entries)}")
    elif args.r2e_source_dir:
        if not args.prepared_image_manifest or not args.require_prepared_images:
            parser.error("R2E collection requires --prepared-image-manifest and --require-prepared-images")
        if args.network_mode != "none":
            parser.error("R2E prepared-image collection requires --network-mode none")
        if not args.antihack:
            parser.error("R2E prepared-image collection requires --antihack")
        entries, fingerprints = load_r2e_tasks(
            args.r2e_source_dir,
            count=args.task_count,
            selection_seed=args.selection_seed,
        )
        manifest = validate_r2e_collection_manifest(
            args.prepared_image_manifest,
            entries,
            fingerprints,
            task_count=args.task_count,
            selection_seed=args.selection_seed,
        )
        progress(
            f"R2E PREFLIGHT passed tasks={len(entries)} seed={args.selection_seed} "
            f"manifest={args.prepared_image_manifest}"
        )
        if args.prepull_prepared_images:
            prepull_r2e_paired_images(manifest, entries)
            progress(f"R2E PREPULL passed image_pairs={len(entries)}")
    else:
        entries = load_entries(args.input)
        if args.instance_id:
            wanted = set(args.instance_id)
            entries = [entry for entry in entries if str(entry_id(entry)) in wanted]
            found = {str(entry_id(entry)) for entry in entries}
            if found != wanted:
                parser.error(f"instance IDs not found: {sorted(wanted - found)}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.antihack and not args.antihack_events:
        args.antihack_events = str(output.with_suffix(".antihack.jsonl"))
    if args.preflight_only:
        return
    if not args.base_url or not args.model or not os.environ.get(args.api_key_env):
        parser.error("teacher base URL, model, and API-key environment variable are required")
    run_lock = acquire_run_lock(args.lock_file or f"{output}.lock")
    completed = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as existing_stream:
            for line in existing_stream:
                record = json.loads(line)
                expected_prompt = {
                    "version": PROMPT_VERSION,
                    "variant": (
                        "swebench-rollout" if args.swebench_rollout_only
                        else "r2e-offline" if (args.r2e_source_dir or args.swebench_verified)
                        else "generic"
                    ),
                }
                if record.get("schema_version") != SCHEMA_VERSION or record.get("prompt") != expected_prompt:
                    raise RuntimeError(
                        "existing output uses a different Leaf prompt/schema; use a new output file"
                    )
                if args.require_prepared_images:
                    environment = record.get("environment") or {}
                    verifier = record.get("verifier") or {}
                    if (
                        environment.get("network_mode") != "none"
                        or environment.get("offline_verified") is not True
                        or not environment.get("prepared_digest")
                        or ((args.r2e_source_dir or args.swebench_verified) and environment.get("image_role") != "agent")
                        or ((args.r2e_source_dir or args.swebench_verified) and verifier.get("network_mode") != "none")
                        or ((args.r2e_source_dir or args.swebench_verified) and not verifier.get("digest"))
                    ):
                        raise RuntimeError(
                            "existing output contains a non-prepared trajectory; "
                            "use a new output file"
                        )
                completed.add((record.get("instance_id"), record.get("seed")))
    desired = [(entry, seed) for entry in entries for seed in seeds]
    jobs = [(entry, seed) for entry, seed in desired if (entry_id(entry), seed) not in completed]
    already_done = len(desired) - len(jobs)
    progress(
        f"RESUME target={len(desired)} already_done={already_done} "
        f"remaining={len(jobs)} workers={args.workers}"
    )
    lock = threading.Lock()
    finished = successes = errors = 0
    with output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, entry, seed, args) for entry, seed in jobs]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                finished += 1
                successes += int(record["reward"] > 0)
                errors += int(record["error"] is not None)
                progress(
                    f"DONE progress={already_done + finished}/{len(desired)} "
                    f"task={record['instance_id']} seed={record['seed']} "
                    f"reward={record['reward']:.1f} exit={record['exit_reason']} "
                    f"turns={record['model_turns']} calls={record['tool_calls']} "
                    f"seconds={record['elapsed_sec']:.1f} "
                    f"new_successes={successes} new_errors={errors}"
                )
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
    run_lock.close()


if __name__ == "__main__":
    main()
