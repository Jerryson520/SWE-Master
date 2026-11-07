#!/usr/bin/env python3
"""Smoke-test the pinned ripgrep binary in prepared SWE-bench images."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from r2egym.agenthub.leaf import AgentAction, RepoToolEnvironment, ToolCall


DEFAULT_INSTANCES = ("django__django-17084", "matplotlib__matplotlib-13989")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-glob", required=True, help="Glob matching prepared SWE-bench JSONL files.")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(args.instance_ids or DEFAULT_INSTANCES)
    rows: dict[str, dict] = {}
    paths = sorted(glob.glob(args.dataset_glob))
    if not paths:
        raise SystemExit(f"No dataset files matched: {args.dataset_glob}")
    for filename in paths:
        with Path(filename).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["instance_id"] in wanted:
                    rows[row["instance_id"]] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise SystemExit(f"Instances absent from dataset: {', '.join(missing)}")

    for instance in sorted(wanted):
        row = dict(rows[instance])
        row["_swebench_rollout_only"] = True
        row.update(backend="docker", step_timeout=180, reward_timeout=1800, network_mode="none")
        env = RepoToolEnvironment.from_dict(row)
        started = time.time()
        try:
            env.reset()
            raw, status = env.env.runtime.run(
                "test -f /usr/local/bin/rg && test -x /usr/local/bin/rg "
                "&& test ! -L /usr/local/bin/rg "
                "&& /usr/local/bin/rg --version | head -1"
            )
            results, _, _, info = env.step(
                AgentAction(
                    (ToolCall("smoke", "Grep", {"pattern": "def ", "path": ".", "head_limit": 1}),),
                    "",
                    "",
                )
            )
            print(json.dumps({
                "instance_id": instance,
                "setup_status": str(status),
                "setup": str(raw).strip(),
                "grep": results[0].content[:300],
                "tool_success": info["tool_success"],
                "elapsed": round(time.time() - started, 2),
            }, ensure_ascii=False), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
