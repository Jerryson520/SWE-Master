#!/usr/bin/env python3
"""Build a small, fixed-context Base-vs-SFT stop-boundary diagnostic set.

The output contains the exact message prefix immediately before a selected SFT
assistant turn.  The same ``context_messages`` and ``tools`` can therefore be
fed to both the Base and Full-SFT models without rollout-history confounding.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SFT_GLOB = (
    "results/qwen3-4b-instruct-2507-v3-reasoning-fullsft-"
    "official-topp08-pp0-w72-shard[0-5].jsonl"
)
DEFAULT_BASE_GLOB = (
    "results/qwen3-4b-instruct-2507-base-official-topp08-pp0-"
    "w48-shard[0-2].jsonl"
)
DEFAULT_OUTPUT_DIR = "results/diagnostics/qwen3-4b-stop-boundary-panel"


def load_jsonl(pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(path for path in glob.glob(pattern) if ".antihack." not in path)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {pattern}")
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_source_file"] = path
                record["_source_line"] = line_number
                records.append(record)
    return records


def last_assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("Trajectory contains no assistant message")


def assistant_turn_number(messages: list[dict[str, Any]], index: int) -> int:
    return sum(message.get("role") == "assistant" for message in messages[: index + 1])


def assistant_text(message: dict[str, Any]) -> str:
    return "\n".join(
        value
        for value in (message.get("reasoning_content"), message.get("content"))
        if isinstance(value, str) and value
    )


def evenly_spaced(records: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select deterministic quantiles after sorting by assistant-turn depth."""
    ordered = sorted(
        records,
        key=lambda item: (
            item["_assistant_turn"],
            item["instance_id"],
            item.get("seed", -1),
        ),
    )
    if len(ordered) < count:
        raise ValueError(f"Need {count} cases, but candidate pool has only {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(position * (len(ordered) - 1) / (count - 1)) for position in range(count)]
    return [ordered[index] for index in indices]


def annotate(records: list[dict[str, Any]]) -> None:
    for record in records:
        index = last_assistant_index(record["messages"])
        record["_assistant_index"] = index
        record["_assistant_turn"] = assistant_turn_number(record["messages"], index)
        record["_literal_tool_call"] = "<tool_call>" in assistant_text(record["messages"][index])


def retarget(record: dict[str, Any], index: int, control_kind: str) -> dict[str, Any]:
    """Return a shallow trajectory copy targeting a particular assistant turn."""
    targeted = dict(record)
    targeted["_assistant_index"] = index
    targeted["_assistant_turn"] = assistant_turn_number(record["messages"], index)
    targeted["_literal_tool_call"] = "<tool_call>" in assistant_text(record["messages"][index])
    targeted["_control_kind"] = control_kind
    return targeted


def make_case(
    sft: dict[str, Any],
    base: dict[str, Any] | None,
    group: str,
    ordinal: int,
) -> dict[str, Any]:
    assistant_index = sft["_assistant_index"]
    reference = deepcopy(sft["messages"][assistant_index])
    return {
        "schema_version": "stop-boundary-diagnostic-v1",
        "case_id": f"{group}-{ordinal:02d}",
        "group": group,
        "control_kind": sft.get("_control_kind"),
        "instance_id": sft["instance_id"],
        "sft_seed": sft.get("seed"),
        "base_seed": base.get("seed") if base else None,
        "assistant_turn_1based": sft["_assistant_turn"],
        "assistant_message_index_0based": assistant_index,
        "sft_exit_reason": sft.get("exit_reason"),
        "base_exit_reason": base.get("exit_reason") if base else None,
        "literal_tool_call_in_reference": sft["_literal_tool_call"],
        "context_messages": deepcopy(sft["messages"][:assistant_index]),
        "reference_assistant": reference,
        "messages_after_reference": deepcopy(sft["messages"][assistant_index + 1 :]),
        "tools": deepcopy(sft.get("tools") or []),
        "source": {
            "sft_file": sft["_source_file"],
            "sft_line": sft["_source_line"],
            "base_file": base.get("_source_file") if base else None,
            "base_line": base.get("_source_line") if base else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-glob", default=DEFAULT_SFT_GLOB)
    parser.add_argument("--base-glob", default=DEFAULT_BASE_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-group", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_records = load_jsonl(args.sft_glob)
    base_records = load_jsonl(args.base_glob)
    annotate(sft_records)
    annotate(base_records)
    base_by_instance = {record["instance_id"]: record for record in base_records}

    no_literal_failures = [
        record
        for record in sft_records
        if record.get("exit_reason") == "max_tokens_per_turn"
        and not record["_literal_tool_call"]
    ]
    shared_hard_failures = [
        record
        for record in no_literal_failures
        if base_by_instance.get(record["instance_id"], {}).get("exit_reason") != "agent"
    ]
    group_one = evenly_spaced(shared_hard_failures, args.per_group)
    used_ids = {record["instance_id"] for record in group_one}

    paired_recoveries = [
        record
        for record in no_literal_failures
        if record["instance_id"] not in used_ids
        and record["instance_id"] in base_by_instance
        and base_by_instance[record["instance_id"]].get("exit_reason") == "agent"
    ]
    group_two = evenly_spaced(paired_recoveries, args.per_group)
    used_ids.update(record["instance_id"] for record in group_two)

    # A useful normal control must expose an unambiguous boundary.  Use half
    # clean final-answer turns (for EOS) and half parsed structured-tool turns
    # (for the tool-call boundary), rather than final turns containing an
    # unparsed literal <tool_call> marker.
    clean_final_controls = [
        retarget(record, record["_assistant_index"], "clean_final_answer")
        for record in sft_records
        if record.get("exit_reason") == "agent"
        and record["instance_id"] not in used_ids
        and not record["_literal_tool_call"]
        and not record["messages"][record["_assistant_index"]].get("tool_calls")
    ]
    clean_count = args.per_group // 2
    clean_controls = evenly_spaced(clean_final_controls, clean_count)
    control_ids = used_ids | {record["instance_id"] for record in clean_controls}

    structured_tool_controls = []
    for record in sft_records:
        if record.get("exit_reason") != "agent" or record["instance_id"] in control_ids:
            continue
        tool_indices = [
            index
            for index, message in enumerate(record["messages"])
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        if tool_indices:
            structured_tool_controls.append(
                retarget(record, tool_indices[-1], "structured_tool_call")
            )
    tool_count = args.per_group - clean_count
    tool_controls = evenly_spaced(structured_tool_controls, tool_count)
    group_three = sorted(
        clean_controls + tool_controls,
        key=lambda record: (record["_assistant_turn"], record["instance_id"]),
    )

    selections = [
        ("base_nonagent_sft_max", group_one),
        ("base_agent_sft_max", group_two),
        ("sft_agent_control", group_three),
    ]
    cases: list[dict[str, Any]] = []
    for group, records in selections:
        for ordinal, record in enumerate(records, start=1):
            cases.append(
                make_case(record, base_by_instance.get(record["instance_id"]), group, ordinal)
            )

    case_ids = [case["instance_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("Diagnostic groups must contain unique instance IDs")
    for case in cases:
        if case["context_messages"] + [case["reference_assistant"]] + case["messages_after_reference"] != next(
            record["messages"] for record in sft_records if record["instance_id"] == case["instance_id"]
        ):
            raise AssertionError(f"Message reconstruction failed for {case['case_id']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "diagnostic_cases.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    table_path = output_dir / "cases.tsv"
    columns = [
        "case_id",
        "group",
        "control_kind",
        "instance_id",
        "sft_seed",
        "base_seed",
        "assistant_turn_1based",
        "assistant_message_index_0based",
        "sft_exit_reason",
        "base_exit_reason",
        "literal_tool_call_in_reference",
        "sft_file",
        "sft_line",
        "base_file",
        "base_line",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    **{key: case.get(key) for key in columns},
                    **case["source"],
                }
            )

    manifest = {
        "schema_version": "stop-boundary-diagnostic-manifest-v1",
        "sft_glob": args.sft_glob,
        "base_glob": args.base_glob,
        "selection_method": "deterministic quantiles by SFT assistant-turn depth",
        "same_context_policy": (
            "For each case, feed the exact context_messages and tools fields to both models."
        ),
        "pool_counts": {
            "sft_total": len(sft_records),
            "base_total": len(base_records),
            "sft_exit_reasons": dict(Counter(record.get("exit_reason") for record in sft_records)),
            "sft_max_no_literal": len(no_literal_failures),
            "base_nonagent_sft_max_no_literal": len(shared_hard_failures),
            "base_agent_sft_max_no_literal": sum(
                base_by_instance.get(record["instance_id"], {}).get("exit_reason") == "agent"
                for record in no_literal_failures
            ),
            "sft_agent_controls": sum(
                record.get("exit_reason") == "agent" for record in sft_records
            ),
        },
        "selected_counts": dict(Counter(case["group"] for case in cases)),
        "selected_control_kinds": dict(
            Counter(case["control_kind"] for case in cases if case["control_kind"])
        ),
        "selected_unique_instances": len(set(case_ids)),
        "outputs": {
            "machine_readable": str(jsonl_path),
            "human_readable": str(table_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
