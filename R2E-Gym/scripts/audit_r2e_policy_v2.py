#!/usr/bin/env python3
"""Audit an in-progress R2E policy-v2 rollout without exposing private tests."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import subprocess
from pathlib import Path


EXPECTED_SCHEMA = "leaf-openai-tools-v2"
EXPECTED_PROMPT = {"version": "leaf-prompt-v2", "variant": "r2e-offline"}
SCORING_PARTS = (
    "/r2e_tests",
    "r2e_tests/",
    "/root/run_tests.sh",
    "/testbed/run_tests.sh",
    "expected_test_output.json",
    "gold.patch",
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return round(vals[lo], 3)
    return round(vals[lo] * (hi - pos) + vals[hi] * (pos - lo), 3)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3) if values else None,
    }


def canonical_call(call: dict) -> str:
    fn = call.get("function") or {}
    name = str(fn.get("name") or "")
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw)
    except Exception:
        return name + ":" + re.sub(r"\s+", " ", str(raw)).strip()
    if name == "Bash" and isinstance(args, dict) and isinstance(args.get("command"), str):
        args = dict(args)
        args["command"] = re.sub(r"\s+", " ", args["command"]).strip()
    return name + ":" + json.dumps(args, sort_keys=True, separators=(",", ":"))


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                paths.extend(match.groups())
        elif line.startswith("+++ ") or line.startswith("--- "):
            value = line[4:].strip()
            if value != "/dev/null":
                paths.append(re.sub(r"^[ab]/", "", value))
    return sorted(set(paths))


def counter_dict(counter: collections.Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--worker-limit", type=int, default=72)
    args = parser.parse_args()

    records: list[dict] = []
    parse_errors: list[dict] = []
    with open(args.input, errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                parse_errors.append({"line": line_no, "kind": "unterminated"})
            try:
                records.append(json.loads(line))
            except Exception as exc:
                parse_errors.append({"line": line_no, "kind": type(exc).__name__})

    duplicate_keys: list[str] = []
    key_counts: collections.Counter[str] = collections.Counter()
    reward_counter: collections.Counter[str] = collections.Counter()
    error_counter: collections.Counter[str] = collections.Counter()
    verifier_status: collections.Counter[str] = collections.Counter()
    verifier_codes: collections.Counter[str] = collections.Counter()
    patch_applied: collections.Counter[str] = collections.Counter()
    repo_reward: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seed_reward: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    repo_error: collections.Counter[str] = collections.Counter()
    seed_error: collections.Counter[str] = collections.Counter()
    exit_reasons: collections.Counter[str] = collections.Counter()
    schema_bad: list[str] = []
    prompt_bad: list[str] = []
    tools_bad: list[str] = []
    policy_bad: list[str] = []
    blocked_mapping_bad: list[str] = []
    isolation_bad: list[str] = []
    digest_bad: list[str] = []
    empty_patch: list[str] = []
    tests_modified: list[str] = []
    illegal_patch: list[str] = []
    scoring_patch: list[str] = []
    scoring_attempts: list[str] = []
    suspicious_attempts: list[str] = []
    exact_repeat_traj: list[str] = []
    exact_repeat_count = 0
    blocked_exact_retry_traj: list[str] = []
    broad_test_traj: list[str] = []
    max_step: list[str] = []
    timeout_records: list[str] = []
    all_paths: collections.Counter[str] = collections.Counter()
    turns: list[float] = []
    calls: list[float] = []
    elapsed: list[float] = []
    agent_elapsed: list[float] = []
    verifier_elapsed: list[float] = []
    blocks_total = 0
    blocked_trajectories = 0
    policy_ids: set[str] = set()

    for record in records:
        iid = str(record.get("instance_id") or "<missing>")
        seed = record.get("seed")
        key = f"{iid}:seed:{seed}"
        key_counts[key] += 1
        repo = str(record.get("repo") or iid.split("__")[1] if "__" in iid else "unknown")
        reward = str(record.get("reward"))
        reward_counter[reward] += 1
        repo_reward[repo][reward] += 1
        seed_reward[str(seed)][reward] += 1
        error = record.get("error")
        if error:
            safe_error = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(error))[:300]
            error_counter[safe_error] += 1
            repo_error[repo] += 1
            seed_error[str(seed)] += 1
        verifier = record.get("verifier") or {}
        verifier_status[str(verifier.get("status"))] += 1
        verifier_codes[str(verifier.get("error_code"))] += 1
        patch_applied[str(verifier.get("patch_applied"))] += 1
        exit_reasons[str(record.get("exit_reason"))] += 1
        turns.append(float(record.get("model_turns") or 0))
        calls.append(float(record.get("tool_calls") or 0))
        for field, bucket in (("elapsed_sec", elapsed), ("agent_elapsed_sec", agent_elapsed), ("verifier_elapsed_sec", verifier_elapsed)):
            value = record.get(field)
            if isinstance(value, (int, float)):
                bucket.append(float(value))

        if record.get("schema_version") != EXPECTED_SCHEMA:
            schema_bad.append(key)
        prompt = record.get("prompt") or {}
        if any(prompt.get(k) != v for k, v in EXPECTED_PROMPT.items()):
            prompt_bad.append(key)
        tools = record.get("tools") or []
        if not tools or any(
            tool.get("type") != "function"
            or not isinstance(tool.get("function"), dict)
            or not all(name in tool["function"] for name in ("name", "description", "parameters"))
            or (tool["function"].get("parameters") or {}).get("type") != "object"
            for tool in tools
        ):
            tools_bad.append(key)

        policy = record.get("policy") or {}
        blocked_ids = policy.get("blocked_tool_call_ids") or []
        block_count = int(policy.get("block_count") or 0)
        blocks_total += block_count
        blocked_trajectories += bool(block_count)
        policy_ids.update(map(str, blocked_ids))
        if block_count != len(blocked_ids) or len(blocked_ids) != len(set(blocked_ids)):
            policy_bad.append(key)

        calls_by_id: dict[str, tuple[int, str]] = {}
        tool_results: dict[str, str] = {}
        ordered_calls: list[tuple[int, str, str, str]] = []
        for msg_index, message in enumerate(record.get("messages") or []):
            for call in message.get("tool_calls") or []:
                call_id = str(call.get("id"))
                signature = canonical_call(call)
                fn = call.get("function") or {}
                raw_args = str(fn.get("arguments") or "")
                calls_by_id[call_id] = (msg_index, signature)
                ordered_calls.append((msg_index, call_id, signature, raw_args))
                lower = raw_args.lower()
                if any(part.lower() in lower for part in SCORING_PARTS):
                    scoring_attempts.append(key)
                if any(token in lower for token in ("git reflog", "git stash", "refs/remotes", "git fetch", "git pull", "curl ", "wget ", "http://", "https://")):
                    suspicious_attempts.append(key)
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {}
                command = str(parsed_args.get("command") or "") if isinstance(parsed_args, dict) else ""
                broad = re.search(r"(?:^|[;&|]\s*)(?:python\s+-m\s+pytest|pytest)(?:\s+(?:-\w+\s*)*)?$", command.strip()) or re.search(r"(?:^|[;&|]\s*)(?:tox|nox)(?:\s|$)", command)
                if broad:
                    broad_test_traj.append(key)
            if message.get("role") == "tool" and message.get("tool_call_id"):
                tool_results[str(message["tool_call_id"])] = str(message.get("content") or "")

        signatures = [item[2] for item in ordered_calls]
        repeats = len(signatures) - len(set(signatures))
        if repeats:
            exact_repeat_traj.append(key)
            exact_repeat_count += repeats
        for blocked_id in blocked_ids:
            if blocked_id not in calls_by_id or not tool_results.get(blocked_id, "").startswith("POLICY_BLOCKED"):
                blocked_mapping_bad.append(key)
                continue
            blocked_index, signature = calls_by_id[blocked_id]
            if any(index > blocked_index and later == signature for index, _, later, _ in ordered_calls):
                blocked_exact_retry_traj.append(key)

        env = record.get("environment") or {}
        if not (
            env.get("network_mode") == "none"
            and env.get("image_role") == "agent"
            and env.get("isolation_verified") is True
            and verifier.get("network_mode") == "none"
        ):
            isolation_bad.append(key)
        if not env.get("prepared_digest") or not verifier.get("digest"):
            digest_bad.append(key)

        patch = str(record.get("output_patch") or "")
        if not patch:
            empty_patch.append(key)
        paths = patch_paths(patch)
        all_paths.update(paths)
        if any("test" in Path(path).name.lower() or "/tests/" in "/" + path.lower() for path in paths):
            tests_modified.append(key)
        for path in paths:
            parts = Path(path).parts
            if path.startswith("/") or ".." in parts or ".git" in parts:
                illegal_patch.append(key)
            if any(part.lower().strip("/") in path.lower() for part in SCORING_PARTS):
                scoring_patch.append(key)
        if float(record.get("model_turns") or 0) >= 150 or record.get("exit_reason") == "max_steps":
            max_step.append(key)
        if "timeout" in str(error).lower() or record.get("exit_reason") == "timeout":
            timeout_records.append(key)

    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)

    sidecar_events: list[dict] = []
    sidecar_errors = 0
    if os.path.exists(args.sidecar):
        for line in open(args.sidecar, errors="replace"):
            try:
                sidecar_events.append(json.loads(line))
            except Exception:
                sidecar_errors += 1
    sidecar_ids = {str(event.get("tool_call_id")) for event in sidecar_events if event.get("tool_call_id")}
    pattern_counts = collections.Counter(str(event.get("pattern_id")) for event in sidecar_events)
    pattern_by_traj: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for event in sidecar_events:
        pattern_by_traj[str(event.get("trajectory_id"))][str(event.get("pattern_id"))] += 1
    same_pattern_retry = sorted(
        trajectory for trajectory, counts_ in pattern_by_traj.items() if any(count > 1 for count in counts_.values())
    )

    log_text = Path(args.log).read_text(errors="replace") if os.path.exists(args.log) else ""
    api_429 = len(re.findall(r"(?:\b429\b|rate.?limit)", log_text, re.I))
    api_retry = len(re.findall(r"\bretry(?:ing)?\b", log_text, re.I))

    docker_lines: list[str] = []
    try:
        output = subprocess.check_output(["docker", "ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Names}}"], text=True)
        docker_lines = [line for line in output.splitlines() if line.strip()]
    except Exception:
        pass
    agent_containers = sum("-agent" in line for line in docker_lines)
    verifier_containers = len(docker_lines) - agent_containers

    successes = [r for r in records if r.get("reward") == 1]
    failures = [r for r in records if r.get("reward") == 0 and not r.get("error")]
    infra = [r for r in records if r.get("error")]
    representative = {
        "success": [f"{r.get('instance_id')}:seed:{r.get('seed')}" for r in sorted(successes, key=lambda r: r.get("tool_calls") or 0, reverse=True)[:3]],
        "ordinary_failure": [f"{r.get('instance_id')}:seed:{r.get('seed')}" for r in sorted(failures, key=lambda r: r.get("tool_calls") or 0, reverse=True)[:3]],
        "infrastructure_failure": [f"{r.get('instance_id')}:seed:{r.get('seed')}" for r in infra[:5]],
    }

    def unique(values: list[str]) -> list[str]:
        return sorted(set(values))

    report = {
        "report_version": "r2e-policy-v2-audit-v1",
        "milestone": args.milestone,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "snapshot_records": len(records),
        "json_integrity": {"parse_or_truncation_errors": parse_errors},
        "identity": {"unique_instance_seed": len(key_counts), "duplicate_keys": duplicate_keys},
        "contracts": {
            "schema_bad": unique(schema_bad),
            "prompt_bad": unique(prompt_bad),
            "tools_bad": unique(tools_bad),
            "policy_summary_bad": unique(policy_bad),
            "blocked_call_mapping_bad": unique(blocked_mapping_bad),
        },
        "isolation": {"field_failures": unique(isolation_bad), "digest_missing": unique(digest_bad)},
        "verifier": {
            "status": counter_dict(verifier_status),
            "error_code": counter_dict(verifier_codes),
            "patch_applied": counter_dict(patch_applied),
        },
        "outcomes": {
            "reward": counter_dict(reward_counter),
            "errors": counter_dict(error_counter),
            "reward_by_repo": {repo: counter_dict(values) for repo, values in sorted(repo_reward.items())},
            "reward_by_seed": {seed: counter_dict(values) for seed, values in sorted(seed_reward.items())},
            "errors_by_repo": counter_dict(repo_error),
            "errors_by_seed": counter_dict(seed_error),
            "exit_reasons": counter_dict(exit_reasons),
        },
        "patches": {
            "empty": unique(empty_patch),
            "trajectories_modifying_tests": unique(tests_modified),
            "illegal_paths": unique(illegal_patch),
            "scoring_asset_paths": unique(scoring_patch),
            "top_modified_paths": all_paths.most_common(30),
        },
        "behavior": {
            "policy_blocks": blocks_total,
            "policy_blocked_trajectories": blocked_trajectories,
            "scoring_asset_attempts": unique(scoring_attempts),
            "suspicious_network_or_git_attempts": unique(suspicious_attempts),
            "exact_repeated_call_count": exact_repeat_count,
            "trajectories_with_exact_repeated_calls": unique(exact_repeat_traj),
            "blocked_exact_retry_trajectories": unique(blocked_exact_retry_traj),
            "sidecar_same_pattern_retry_trajectories": same_pattern_retry,
            "broad_test_trajectories": unique(broad_test_traj),
            "max_step_trajectories": unique(max_step),
            "timeout_trajectories": unique(timeout_records),
        },
        "distributions": {
            "model_turns": stats(turns),
            "tool_calls": stats(calls),
            "elapsed_sec": stats(elapsed),
            "agent_elapsed_sec": stats(agent_elapsed),
            "verifier_elapsed_sec": stats(verifier_elapsed),
        },
        "api": {"log_429_or_rate_limit_mentions": api_429, "log_retry_mentions": api_retry},
        "antihack_reconciliation": {
            "sidecar_events": len(sidecar_events),
            "sidecar_parse_errors": sidecar_errors,
            "sidecar_unique_call_ids": len(sidecar_ids),
            "policy_unique_call_ids": len(policy_ids),
            "missing_from_sidecar": sorted(policy_ids - sidecar_ids),
            "extra_sidecar_ids": sorted(sidecar_ids - policy_ids),
            "pattern_counts": counter_dict(pattern_counts),
        },
        "containers": {
            "running_total": len(docker_lines),
            "running_agent": agent_containers,
            "running_non_agent_or_verifier": verifier_containers,
            "worker_limit": args.worker_limit,
            "over_worker_limit": len(docker_lines) > args.worker_limit,
            "note": "Mid-run snapshot; final audit must confirm zero paired containers after completion.",
        },
        "representative_trajectories": representative,
    }

    critical = sum(
        len(report["contracts"][field])
        for field in ("schema_bad", "prompt_bad", "tools_bad", "policy_summary_bad", "blocked_call_mapping_bad")
    ) + len(report["json_integrity"]["parse_or_truncation_errors"])
    infrastructure = len(infra)
    recommendation = "continue"
    if critical or infrastructure > 3 or report["containers"]["over_worker_limit"]:
        recommendation = "pause_and_investigate"
    elif infrastructure:
        recommendation = "continue_with_infrastructure_fix_before_retries"
    report["assessment"] = {
        "critical_contract_failures": critical,
        "infrastructure_failures": infrastructure,
        "recommendation": recommendation,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    success_count = reward_counter.get("1.0", 0) + reward_counter.get("1", 0)
    failure_count = reward_counter.get("0.0", 0) + reward_counter.get("0", 0)
    md = f"""# R2E policy-v2 milestone {args.milestone} audit

- Snapshot records: **{len(records)}**
- Unique instance+seed: **{len(key_counts)}**; duplicates: **{len(duplicate_keys)}**
- Reward 1 / 0: **{success_count} / {failure_count}**
- Infrastructure errors: **{infrastructure}**
- Contract failures: **{critical}**
- Policy blocks: **{blocks_total}** across **{blocked_trajectories}** trajectories
- Empty patches: **{len(set(empty_patch))}**
- Test-modifying patches: **{len(set(tests_modified))}**
- Illegal/scoring patch paths: **{len(set(illegal_patch))} / {len(set(scoring_patch))}**
- Exact blocked-call retries: **{len(set(blocked_exact_retry_traj))}** trajectories
- Same anti-hack pattern repeated after blocking (heuristic): **{len(same_pattern_retry)}** trajectories
- Broad test runs: **{len(set(broad_test_traj))}** trajectories
- Running containers: **{len(docker_lines)}** (agent={agent_containers}, other/verifier={verifier_containers})
- Recommendation: **{recommendation}**

## Representative evidence

- Successes: {', '.join(representative['success']) or 'none'}
- Ordinary failures: {', '.join(representative['ordinary_failure']) or 'none'}
- Infrastructure failures: {', '.join(representative['infrastructure_failure']) or 'none'}

See the adjacent JSON report for full counts, distributions, repository/seed breakdowns, and identifier lists. Private verifier output and API credentials are intentionally excluded.
"""
    out_md.write_text(md)
    print(json.dumps(report["assessment"], ensure_ascii=False))
    print(json.dumps({
        "records": len(records),
        "reward": report["outcomes"]["reward"],
        "blocks": blocks_total,
        "blocked_trajectories": blocked_trajectories,
        "empty": len(set(empty_patch)),
        "test_modifications": len(set(tests_modified)),
        "repeats": len(set(exact_repeat_traj)),
        "blocked_retries": len(set(blocked_exact_retry_traj)),
        "broad_tests": len(set(broad_test_traj)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
