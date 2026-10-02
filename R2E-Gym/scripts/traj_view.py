#!/usr/bin/env python3
"""Small CLI for inspecting and comparing R2E-Gym trajectory JSONL files."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def instance_id(record: dict[str, Any]) -> str:
    return str(record.get("ds", {}).get("instance_id") or record.get("docker_image") or "")


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error


def find_record(path: Path, query: str) -> dict[str, Any]:
    exact = None
    partial = []
    for record in iter_records(path):
        current = instance_id(record)
        if current == query:
            exact = record
            break
        if query.lower() in current.lower():
            partial.append(record)
    if exact:
        return exact
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"No trajectory matching {query!r} in {path}")
    matches = "\n".join(f"  {instance_id(item)}" for item in partial)
    raise ValueError(f"Query {query!r} is ambiguous:\n{matches}")


def reward_value(record: dict[str, Any]) -> float:
    try:
        return float(record.get("reward") or 0)
    except (TypeError, ValueError):
        return 0.0


def shorten(value: Any, width: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def action_name(action: Any) -> str:
    text = str(action or "").strip()
    if "<function=" in text:
        return text.split("<function=", 1)[1].split(">", 1)[0]
    return shorten(text.splitlines()[0] if text else "", 50)


def command_summary(args: argparse.Namespace) -> None:
    records = list(iter_records(args.file))
    rewards = Counter(int(reward_value(item)) for item in records)
    exits = Counter(str(item.get("exit_reason")) for item in records)
    print(f"file: {args.file}")
    print(f"tasks: {len(records)}")
    print(f"reward=1: {rewards[1]}")
    print(f"reward=0: {rewards[0]}")
    if records:
        print(f"resolve_rate: {rewards[1] / len(records):.1%}")
    print("exit_reasons:")
    for name, count in exits.most_common():
        print(f"  {name}: {count}")


def command_list(args: argparse.Namespace) -> None:
    rows = []
    for record in iter_records(args.file):
        current = instance_id(record)
        reward = int(reward_value(record))
        if args.filter and args.filter.lower() not in current.lower():
            continue
        if args.reward is not None and reward != args.reward:
            continue
        rows.append(
            (
                current,
                reward,
                str(record.get("exit_reason")),
                len(record.get("trajectory_steps") or []),
                record.get("trajectory_completion_tokens", ""),
            )
        )
    print(f"{'INSTANCE':42} {'R':>1} {'EXIT':20} {'STEPS':>5} {'TOKENS':>8}")
    for current, reward, exit_reason, steps, tokens in rows:
        print(f"{current:42} {reward:>1} {exit_reason:20} {steps:>5} {str(tokens):>8}")
    print(f"\nmatched: {len(rows)}")


def print_step(step: dict[str, Any], full: bool, width: int) -> None:
    index = step.get("step_idx", step.get("step_count", "?"))
    action = step.get("action", "")
    observation = step.get("observation", "")
    thought = step.get("thought", "")
    print(f"\n{'=' * 24} STEP {index} {'=' * 24}")
    if full:
        print("THOUGHT:\n" + str(thought or ""))
        print("\nACTION:\n" + str(action or ""))
        print("\nOBSERVATION:\n" + str(observation or ""))
    else:
        print(f"action: {action_name(action)} | {shorten(action, width)}")
        print(f"obs:    {shorten(observation, width)}")


def command_show(args: argparse.Namespace) -> None:
    record = find_record(args.file, args.task)
    steps = record.get("trajectory_steps") or []
    print(f"instance: {instance_id(record)}")
    print(f"reward: {record.get('reward')}")
    print(f"exit_reason: {record.get('exit_reason')}")
    print(f"steps: {len(steps)}")
    print(f"completion_tokens: {record.get('trajectory_completion_tokens')}")
    print(f"patch_chars: {len(record.get('output_patch') or '')}")

    selected = set(args.step or [])
    needle = (args.grep or "").lower()
    for step in steps:
        index = int(step.get("step_idx", step.get("step_count", -1)))
        searchable = "\n".join(
            str(step.get(key, "")) for key in ("thought", "action", "observation")
        ).lower()
        if selected and index not in selected:
            continue
        if needle and needle not in searchable:
            continue
        print_step(step, args.full or bool(selected), args.width)

    if args.patch:
        print(f"\n{'=' * 24} OUTPUT PATCH {'=' * 24}\n")
        print(record.get("output_patch") or "<empty>")
    if args.tests:
        print(f"\n{'=' * 24} TEST OUTPUT {'=' * 24}\n")
        output = str(record.get("test_output") or "<empty>")
        print(output[-args.test_chars :])


def converter_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "app" / "json2html_for_view-zh.py"
    if not path.exists():
        raise ValueError(f"HTML converter not found: {path}")
    return path


def write_selected_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_html(source: Path, query: str, output: Path) -> Path:
    record = find_record(source, query)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = output.with_suffix(".jsonl")
    write_selected_jsonl(record, selected)
    subprocess.run(
        [sys.executable, str(converter_path()), str(selected), "-o", str(output)],
        check=True,
    )
    return output


def command_html(args: argparse.Namespace) -> None:
    record = find_record(args.file, args.task)
    task = instance_id(record)
    output = args.output or Path("results/trajectory_views") / f"{task}.html"
    print(f"generated: {generate_html(args.file, task, output).resolve()}")


def command_compare(args: argparse.Namespace) -> None:
    left_record = find_record(args.baseline, args.task)
    right_record = find_record(args.candidate, args.task)
    task = instance_id(left_record)
    output_dir = args.output_dir or Path("results/trajectory_views") / f"{task}-compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    left_html = generate_html(args.baseline, task, output_dir / "sft.html")
    right_html = generate_html(args.candidate, task, output_dir / "rl.html")
    index = output_dir / "index.html"
    title = html.escape(task)
    index.write_text(
        f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{title} comparison</title>
<style>html,body{{height:100%;margin:0;font-family:sans-serif}}header{{height:42px;display:flex;align-items:center;justify-content:space-around;background:#222;color:white}}main{{height:calc(100% - 42px);display:grid;grid-template-columns:1fr 1fr}}iframe{{width:100%;height:100%;border:0}}iframe:first-child{{border-right:2px solid #555}}</style>
</head><body><header><strong>SFT — {title}</strong><strong>RL — {title}</strong></header>
<main><iframe src=\"{left_html.name}\"></iframe><iframe src=\"{right_html.name}\"></iframe></main></body></html>""",
        encoding="utf-8",
    )
    print(f"generated: {index.resolve()}")
    print("Copy the whole comparison directory if you want to view it locally.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect R2E-Gym trajectory JSONL files")
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summary", help="Show aggregate result counts")
    summary.add_argument("file", type=Path)
    summary.set_defaults(func=command_summary)

    listing = commands.add_parser("list", help="List trajectories in a compact table")
    listing.add_argument("file", type=Path)
    listing.add_argument("--filter", help="Only instance IDs containing this text")
    listing.add_argument("--reward", type=int, choices=(0, 1))
    listing.set_defaults(func=command_list)

    show = commands.add_parser("show", help="Show a compact terminal timeline")
    show.add_argument("file", type=Path)
    show.add_argument("task", help="Full instance ID or a unique substring")
    show.add_argument("--step", type=int, action="append", help="Show a full step; repeatable")
    show.add_argument("--grep", help="Only steps containing this text")
    show.add_argument("--full", action="store_true", help="Print full thought/action/observation")
    show.add_argument("--width", type=int, default=260, help="Preview width")
    show.add_argument("--patch", action="store_true", help="Append the output patch")
    show.add_argument("--tests", action="store_true", help="Append the tail of test output")
    show.add_argument("--test-chars", type=int, default=12000)
    show.set_defaults(func=command_show)

    html_command = commands.add_parser("html", help="Generate a Chinese HTML viewer for one task")
    html_command.add_argument("file", type=Path)
    html_command.add_argument("task")
    html_command.add_argument("-o", "--output", type=Path)
    html_command.set_defaults(func=command_html)

    compare = commands.add_parser("compare", help="Generate side-by-side SFT/RL HTML")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("task")
    compare.add_argument("-o", "--output-dir", type=Path)
    compare.set_defaults(func=command_compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
