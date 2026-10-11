#!/usr/bin/env python3
"""Validate, optionally replay, and split fresh Leaf teacher trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

from r2egym.agenthub.leaf import LEAF_TOOLS, LeafAction, LeafToolCall
from rllm.environments.swe.swe_leaf import SWELeafEnv


ALLOWED = {tool["function"]["name"] for tool in LEAF_TOOLS}
SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in LEAF_TOOLS}
FORBIDDEN = {"submit", "finish", "execute_bash", "str_replace_editor", "file_editor", "search"}


def argument_errors(name, arguments):
    schema = SCHEMAS.get(name)
    if schema is None:
        return []
    errors = []
    for required in schema.get("required", []):
        if required not in arguments:
            errors.append(f"{name} missing required argument {required}")
    python_types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for key, value in arguments.items():
        expected = python_types.get((schema.get("properties", {}).get(key) or {}).get("type"))
        if expected is not None and not isinstance(value, expected):
            errors.append(f"{name}.{key} has invalid type")
    return errors


def structural_errors(record):
    errors = []
    messages = record.get("messages") or []
    call_ids, result_ids = [], []
    for message in messages:
        text = str(message.get("content") or "")
        if "<function=" in text:
            errors.append("legacy XML function call")
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name")
                if name not in ALLOWED or str(name).lower() in FORBIDDEN:
                    errors.append(f"invalid tool {name}")
                if call.get("id") in call_ids:
                    errors.append(f"duplicate call id {call.get('id')}")
                call_ids.append(call.get("id"))
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        errors.append(f"non-object arguments {call.get('id')}")
                    else:
                        errors.extend(argument_errors(name, arguments))
                except (json.JSONDecodeError, TypeError):
                    errors.append(f"invalid arguments {call.get('id')}")
        elif message.get("role") == "tool":
            result_ids.append(message.get("tool_call_id"))
    if call_ids != result_ids:
        errors.append("tool result IDs do not match call order")
    if record.get("reward", 0) < 1:
        errors.append("unresolved")
    if not record.get("output_patch"):
        errors.append("empty patch")
    if record.get("exit_reason") != "agent":
        errors.append(f"bad exit {record.get('exit_reason')}")
    return errors


def replay(record, args):
    env = SWELeafEnv.from_dict(
        {**record["entry"], "backend": args.backend, "step_timeout": args.step_timeout, "reward_timeout": args.reward_timeout}
    )
    try:
        env.reset()
        for message in record["messages"]:
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                calls.append(LeafToolCall(call["id"], function["name"], json.loads(function["arguments"])))
            env.step(LeafAction(tuple(calls)))
        patch = env.output_patch()
        reward = env.compute_final_reward()
        return patch == record["output_patch"] and float(reward) >= 1
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--backend", default="docker")
    parser.add_argument("--step-timeout", type=int, default=180)
    parser.add_argument("--reward-timeout", type=int, default=1800)
    args = parser.parse_args()
    accepted, rejected = [], []
    total_records = 0
    for line in Path(args.input).read_text().splitlines():
        record = json.loads(line)
        total_records += 1
        errors = structural_errors(record)
        if not errors and args.replay and not replay(record, args):
            errors.append("replay mismatch")
        if errors:
            rejected.append({"instance_id": record.get("instance_id"), "seed": record.get("seed"), "errors": errors})
        else:
            accepted.append(record)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    streams = {
        "train": (output / "train.jsonl").open("w"),
        "validation": (output / "validation.jsonl").open("w"),
    }
    counts = {"train": 0, "validation": 0}
    try:
        for record in accepted:
            group = f"{record.get('repo')}::{record.get('instance_id')}"
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
            split = "validation" if bucket < args.validation_percent else "train"
            sample = {
                "input": record["messages"],
                "tools": LEAF_TOOLS,
                "instance_id": record["instance_id"],
                "repo": record.get("repo"),
                "seed": record["seed"],
                "schema_version": record["schema_version"],
            }
            streams[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for stream in streams.values():
            stream.close()
    (output / "rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected))
    turns = sum(record.get("model_turns", 0) for record in accepted)
    calls = sum(record.get("tool_calls", 0) for record in accepted)
    multi_tool_turns = sum(
        len(message.get("tool_calls") or []) > 1
        for record in accepted
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    )
    tool_errors = sum(
        str(message.get("content") or "").startswith(("ValueError:", "Unknown tool:", "Tool runner exited"))
        for record in accepted
        for message in record.get("messages", [])
        if message.get("role") == "tool"
    )
    summary = {
        "total": total_records,
        "accepted": len(accepted),
        "rejected": len(rejected),
        **counts,
        "success_rate": len(accepted) / total_records if total_records else 0,
        "average_model_turns": turns / len(accepted) if accepted else 0,
        "average_tool_calls": calls / len(accepted) if accepted else 0,
        "multi_tool_turns": multi_tool_turns,
        "tool_errors": tool_errors,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
