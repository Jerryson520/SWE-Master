#!/usr/bin/env python3
"""Build an offline, per-task browser for paired SWE-Master evaluation runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True, help="SFT result JSONL")
    parser.add_argument("--rl", type=Path, required=True, help="RL result JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Report output directory")
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            instance_id = row.get("instance_id") or row.get("ds", {}).get("instance_id")
            if not instance_id:
                raise ValueError(f"{path}:{line_number}: missing instance_id")
            rows[instance_id] = row
    return rows


def metric(row: dict | None) -> dict | None:
    if row is None:
        return None
    steps = row.get("trajectory_steps") or []
    return {
        "reward": row.get("reward"),
        "exit_reason": row.get("exit_reason"),
        "steps": len(steps),
        "completion_tokens": row.get("trajectory_completion_tokens"),
        "summary_tokens": row.get("summary_completion_tokens"),
        "patch_chars": len(row.get("output_patch") or ""),
        "test_output_chars": len(row.get("test_output") or ""),
    }


def category(sft: dict | None, rl: dict | None) -> str:
    s = bool(sft and sft.get("reward") == 1)
    r = bool(rl and rl.get("reward") == 1)
    if s and r:
        return "both_success"
    if not s and r:
        return "rl_gained"
    if s and not r:
        return "sft_only"
    return "both_failed"


def main() -> None:
    args = parse_args()
    sft_rows = read_jsonl(args.sft)
    rl_rows = read_jsonl(args.rl)
    output = args.output.resolve()
    data_dir = output / "data" / "tasks"
    data_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    counts = {key: 0 for key in ("both_success", "rl_gained", "sft_only", "both_failed")}
    for instance_id in sorted(set(sft_rows) | set(rl_rows)):
        sft = sft_rows.get(instance_id)
        rl = rl_rows.get(instance_id)
        source = sft or rl or {}
        problem = source.get("problem_statement") or source.get("ds", {}).get("problem_statement") or ""
        cat = category(sft, rl)
        counts[cat] += 1
        safe_id = quote(instance_id, safe="._-")
        task_dir = data_dir / safe_id
        task_dir.mkdir(parents=True, exist_ok=True)
        for label, row in (("sft", sft), ("rl", rl)):
            if row is not None:
                (task_dir / f"{label}.json").write_text(
                    json.dumps(row, ensure_ascii=False), encoding="utf-8"
                )
        tasks.append({
            "instance_id": instance_id,
            "safe_id": safe_id,
            "repo": source.get("ds", {}).get("repo"),
            "problem_statement": problem,
            "problem_preview": " ".join(problem.split())[:240],
            "category": cat,
            "sft": metric(sft),
            "rl": metric(rl),
        })

    summary = {
        "total": len(tasks),
        "counts": counts,
        "sources": {"sft": str(args.sft), "rl": str(args.rl)},
        "tasks": tasks,
    }
    (output / "data" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_dir = Path(__file__).resolve().parent / "static"
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy2(source_dir / name, output / name)
    print(f"Built {len(tasks)} tasks at {output}")
    print("Categories:", ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
