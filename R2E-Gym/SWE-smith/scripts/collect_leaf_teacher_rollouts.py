#!/usr/bin/env python3
"""Generate fresh Leaf trajectories with an OpenAI-compatible teacher."""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

from r2egym.agenthub.leaf import LEAF_TOOLS, SYSTEM_PROMPT, LeafAction, LeafToolCall
from r2egym.agenthub.leaf.environment import LeafRepoEnv


SCHEMA_VERSION = "leaf-frognano-5038f915"
PROGRESS_LOCK = threading.Lock()


def progress(message):
    with PROGRESS_LOCK:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def generate_one(entry, seed, args):
    task_id = entry_id(entry)
    env = LeafRepoEnv.from_dict(
        {**entry, "backend": args.backend, "step_timeout": args.step_timeout, "reward_timeout": args.reward_timeout}
    )
    started = time.monotonic()
    messages = []
    exit_reason = "unknown"
    error = None
    patch = ""
    reward = 0.0
    test_output = ""
    try:
        progress(f"START task={task_id} seed={seed} phase=container")
        instruction, _ = env.reset()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}]
        client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.base_url, timeout=args.request_timeout)
        for turn in range(1, args.max_steps + 1):
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=LEAF_TOOLS,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_completion_tokens=args.max_tokens_per_turn,
                seed=seed,
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
                calls.append(LeafToolCall(call["id"], call["function"].get("name", ""), arguments))
            if args.log_turns:
                names = ",".join(call.name for call in calls) or "none"
                progress(
                    f"TURN task={task_id} seed={seed} turn={turn} "
                    f"finish={choice.finish_reason or 'none'} tools={names}"
                )
            action = LeafAction(tuple(calls), assistant.get("content") or "", assistant.get("reasoning_content") or "")
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
        scored = env.env.runtime._calculate_reward(get_test_output=True, timeout=args.reward_timeout)
        if isinstance(scored, tuple):
            reward, test_output = scored
        else:
            reward = scored
    except Exception as exc:
        exit_reason = "environment_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": entry_id(entry),
        "repo": entry.get("repo") or entry.get("repo_name"),
        "image": entry.get("docker_image") or entry.get("image_name"),
        "entry": entry,
        "teacher": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens_per_turn": args.max_tokens_per_turn,
        },
        "seed": seed,
        "messages": messages,
        "tools": LEAF_TOOLS,
        "output_patch": patch,
        "reward": float(reward),
        "test_output": str(test_output),
        "exit_reason": exit_reason,
        "error": error,
        "model_turns": sum(message.get("role") == "assistant" for message in messages),
        "tool_calls": sum(len(message.get("tool_calls", [])) for message in messages),
        "elapsed_sec": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("LEAF_TEACHER_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("LEAF_TEACHER_MODEL"))
    parser.add_argument("--api-key-env", default="LEAF_TEACHER_API_KEY")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-tokens-per-turn", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--request-timeout", type=int, default=7200)
    parser.add_argument("--step-timeout", type=int, default=180)
    parser.add_argument("--reward-timeout", type=int, default=1800)
    parser.add_argument("--backend", default="docker")
    parser.add_argument("--log-turns", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not args.model or not os.environ.get(args.api_key_env):
        parser.error("teacher base URL, model, and API-key environment variable are required")

    entries = load_entries(args.input)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            record = json.loads(line)
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


if __name__ == "__main__":
    main()
