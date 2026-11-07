#!/usr/bin/env python3
"""Audit repetition and redundancy in Leaf teacher reasoning messages.

The accepted training population is selected by identity from pretokenized
train/validation JSONL files.  This intentionally refuses to infer the 694
examples from aggregate reports or from the older 775-example RSFT dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Sequence


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+|[^\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
LINE_NUMBER_RE = re.compile(r"(?m)^\s*\d+[\s:|]+")
SPACE_RE = re.compile(r"\s+")
CODE_HINT_RE = re.compile(
    r"(?m)(^\s*(?:def |class |from |import |if |for |while |return |@|\$ |>>>|diff --git|@@)|"
    r"[{}();]|\b(?:Traceback|File \"|pytest|Error:|Exception:)\b)"
)
LITERAL_TOOL_CALL_RE = re.compile(r"<\s*/?\s*tool_call\s*>", re.IGNORECASE)

NGRAM_SIZE = 12
MIN_MEANINGFUL_TOKENS = 24
SELF_LOOP_MIN_TOKENS = 128
SELF_LOOP_MIN_RATIO = 0.15
CROSS_TURN_MIN_TOKENS = 128
CROSS_TURN_MIN_SIMILARITY = 0.65
TOOL_COPY_MIN_TOKENS = 128
TOOL_COPY_MIN_RATIO = 0.35


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield line_number, value


def identity(record: dict[str, Any]) -> tuple[str, str, int, str]:
    fingerprint = record.get("trajectory_fingerprint")
    instance_id = record.get("instance_id")
    repo = record.get("repo")
    seed = record.get("seed")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("missing trajectory_fingerprint")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("missing instance_id")
    if not isinstance(repo, str) or not repo:
        raise ValueError("missing repo")
    if type(seed) is not int:
        raise ValueError("invalid seed")
    return repo, instance_id, seed, fingerprint


def load_accepted_identities(paths: Sequence[Path]) -> set[tuple[str, str, int, str]]:
    accepted: set[tuple[str, str, int, str]] = set()
    for path in paths:
        for line_number, record in read_jsonl(path):
            key = identity(record)
            if key in accepted:
                raise ValueError(f"duplicate accepted identity at {path}:{line_number}")
            accepted.add(key)
    return accepted


def load_filtered_records(
    paths: Sequence[Path], accepted: set[tuple[str, str, int, str]]
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in paths:
        split = path.stem
        for input_line, record in read_jsonl(path):
            key = identity(record)
            if key not in accepted:
                continue
            if key in selected:
                raise ValueError(f"duplicate filtered identity at {path}:{input_line}")
            messages = record.get("input")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{input_line}: input is not a message list")
            selected[key] = {
                **record,
                "analysis_split": split,
                "analysis_input_file": str(path.resolve()),
                "analysis_input_line": input_line,
            }
    missing = accepted - selected.keys()
    if missing:
        examples = sorted(missing)[:3]
        raise ValueError(f"{len(missing)} accepted identities missing from filtered inputs: {examples}")
    return list(selected.values())


def normalize_text(text: str) -> str:
    text = LINE_NUMBER_RE.sub("", text.lower())
    return SPACE_RE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def ngrams(items: Sequence[Any], size: int) -> list[tuple[Any, ...]]:
    if len(items) < size:
        return []
    return [tuple(items[index : index + size]) for index in range(len(items) - size + 1)]


def repeated_ngram_positions(tokens: Sequence[int], size: int = NGRAM_SIZE) -> set[int]:
    positions: dict[tuple[int, ...], list[int]] = {}
    for index, gram in enumerate(ngrams(tokens, size)):
        positions.setdefault(gram, []).append(index)
    repeated: set[int] = set()
    for occurrences in positions.values():
        for start in occurrences[1:]:
            repeated.update(range(start, min(start + size, len(tokens))))
    return repeated


def longest_repeated_run(positions: set[int]) -> int:
    best = current = 0
    previous = None
    for position in sorted(positions):
        current = current + 1 if previous is not None and position == previous + 1 else 1
        best = max(best, current)
        previous = position
    return best


def token_overlap_ratio(source: Sequence[int], target: Sequence[int], size: int = NGRAM_SIZE) -> tuple[int, float]:
    if len(source) < size or len(target) < size:
        return 0, 0.0
    target_grams = set(ngrams(target, size))
    matched: set[int] = set()
    for start, gram in enumerate(ngrams(source, size)):
        if gram in target_grams:
            matched.update(range(start, min(start + size, len(source))))
    return len(matched), len(matched) / len(source) if source else 0.0


def word_ngram_similarity(left: str, right: str, size: int = 5) -> float:
    left_set = set(ngrams(words(left), size))
    right_set = set(ngrams(words(right), size))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def consecutive_sentence_repetition(text: str) -> tuple[int, list[str]]:
    parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    normalized = [normalize_text(part) for part in parts]
    repeated: list[str] = []
    count = 0
    for index in range(1, len(parts)):
        if len(normalized[index].split()) >= 5 and normalized[index] == normalized[index - 1]:
            count += 1
            repeated.append(parts[index][:500])
    return count, repeated[:5]


def code_like_ratio(text: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return sum(bool(CODE_HINT_RE.search(line)) for line in lines) / len(lines)


def serialize_assistant_payload(message: dict[str, Any]) -> str:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        parts.append(json.dumps(calls, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


class TokenizerAdapter:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return list(encoded)


def analyze_records(records: Sequence[dict[str, Any]], tokenizer: TokenizerAdapter):
    turn_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for trajectory_index, record in enumerate(records, 1):
        messages = record["input"]
        previous_reasonings: list[tuple[int, str, list[int]]] = []
        prior_tool_texts: list[str] = []
        trajectory_turns: list[dict[str, Any]] = []
        assistant_turn = 0
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "tool":
                content = message.get("content")
                if isinstance(content, str) and content:
                    prior_tool_texts.append(content)
                continue
            if role != "assistant":
                continue
            assistant_turn += 1
            reasoning = message.get("reasoning_content")
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
            if not reasoning:
                continue
            reasoning_ids = tokenizer.encode(reasoning)
            repeated_positions = repeated_ngram_positions(reasoning_ids)
            repeated_tokens = len(repeated_positions)
            repeat_ratio = repeated_tokens / len(reasoning_ids) if reasoning_ids else 0.0
            consecutive_count, repeated_sentences = consecutive_sentence_repetition(reasoning)

            best_previous_turn = None
            best_previous_similarity = 0.0
            best_previous_overlap_tokens = 0
            for prior_turn, prior_text, prior_ids in previous_reasonings:
                similarity = word_ngram_similarity(reasoning, prior_text)
                overlap_tokens, _ = token_overlap_ratio(reasoning_ids, prior_ids)
                if similarity > best_previous_similarity:
                    best_previous_turn = prior_turn
                    best_previous_similarity = similarity
                    best_previous_overlap_tokens = overlap_tokens

            tool_context = "\n".join(prior_tool_texts[-3:])
            tool_overlap_tokens, tool_overlap_ratio = token_overlap_ratio(
                reasoning_ids, tokenizer.encode(tool_context) if tool_context else []
            )
            payload = serialize_assistant_payload(message)
            assistant_overlap_tokens, assistant_overlap_ratio = token_overlap_ratio(
                reasoning_ids, tokenizer.encode(payload) if payload else []
            )
            literal_tool_call = bool(LITERAL_TOOL_CALL_RE.search(reasoning))
            code_ratio = code_like_ratio(reasoning)
            self_loop = (
                repeated_tokens >= SELF_LOOP_MIN_TOKENS
                and repeat_ratio >= SELF_LOOP_MIN_RATIO
            ) or consecutive_count >= 2
            cross_turn = (
                best_previous_overlap_tokens >= CROSS_TURN_MIN_TOKENS
                and best_previous_similarity >= CROSS_TURN_MIN_SIMILARITY
            )
            tool_copy = (
                tool_overlap_tokens >= TOOL_COPY_MIN_TOKENS
                and tool_overlap_ratio >= TOOL_COPY_MIN_RATIO
            )
            legitimate_code = tool_copy and code_ratio >= 0.55 and not self_loop and not cross_turn
            labels: list[str] = []
            if self_loop:
                labels.append("reasoning_self_loop")
            if cross_turn:
                labels.append("cross_turn_repetition")
            if tool_copy:
                labels.append("tool_output_copying")
            if legitimate_code:
                labels.append("legitimate_code_reference")
            if literal_tool_call:
                labels.append("literal_tool_call")
            if not labels:
                labels.append("no_material_issue")
            # A span may be repeated within a turn and copied from an earlier
            # message simultaneously. Use the maximum to avoid double-counting.
            ineffective_candidates = [repeated_tokens, best_previous_overlap_tokens]
            if tool_copy and not legitimate_code:
                ineffective_candidates.append(tool_overlap_tokens)
            ineffective_tokens = min(len(reasoning_ids), max(ineffective_candidates, default=0))
            high_risk = bool(
                self_loop or cross_turn or (tool_copy and not legitimate_code) or literal_tool_call
            )
            row = {
                "trajectory_index": trajectory_index,
                "split": record["analysis_split"],
                "instance_id": record["instance_id"],
                "repo": record["repo"],
                "seed": record["seed"],
                "trajectory_fingerprint": record["trajectory_fingerprint"],
                "source_file": record.get("source_file"),
                "source_line": record.get("source_line"),
                "input_file": record["analysis_input_file"],
                "input_line": record["analysis_input_line"],
                "assistant_turn": assistant_turn,
                "message_index": message_index,
                "reasoning_tokens": len(reasoning_ids),
                "self_repeated_tokens": repeated_tokens,
                "self_repeat_ratio": round(repeat_ratio, 6),
                "longest_repeated_token_run": longest_repeated_run(repeated_positions),
                "consecutive_repeated_sentences": consecutive_count,
                "best_previous_assistant_turn": best_previous_turn,
                "cross_turn_similarity": round(best_previous_similarity, 6),
                "cross_turn_overlap_tokens": best_previous_overlap_tokens,
                "tool_overlap_tokens": tool_overlap_tokens,
                "tool_overlap_ratio": round(tool_overlap_ratio, 6),
                "assistant_payload_overlap_tokens": assistant_overlap_tokens,
                "assistant_payload_overlap_ratio": round(assistant_overlap_ratio, 6),
                "code_like_ratio": round(code_ratio, 6),
                "literal_tool_call": literal_tool_call,
                "estimated_ineffective_tokens": ineffective_tokens,
                "high_risk": high_risk,
                "labels": labels,
                "reasoning_excerpt": reasoning[:2000],
                "repeated_sentence_examples": repeated_sentences,
                "tool_context_excerpt": tool_context[-2000:],
            }
            turn_rows.append(row)
            trajectory_turns.append(row)
            previous_reasonings.append((assistant_turn, reasoning, reasoning_ids))

        reasoning_tokens = sum(row["reasoning_tokens"] for row in trajectory_turns)
        ineffective = sum(row["estimated_ineffective_tokens"] for row in trajectory_turns)
        labels = sorted({label for row in trajectory_turns for label in row["labels"]})
        high_risk_turns = sum(row["high_risk"] for row in trajectory_turns)
        trajectory_rows.append({
            "trajectory_index": trajectory_index,
            "split": record["analysis_split"],
            "instance_id": record["instance_id"],
            "repo": record["repo"],
            "seed": record["seed"],
            "trajectory_fingerprint": record["trajectory_fingerprint"],
            "source_file": record.get("source_file"),
            "source_line": record.get("source_line"),
            "input_file": record["analysis_input_file"],
            "input_line": record["analysis_input_line"],
            "reasoning_turns": len(trajectory_turns),
            "reasoning_tokens": reasoning_tokens,
            "estimated_ineffective_tokens": ineffective,
            "estimated_ineffective_ratio": round(ineffective / reasoning_tokens, 6) if reasoning_tokens else 0.0,
            "high_risk_turns": high_risk_turns,
            "high_risk": high_risk_turns > 0,
            "max_turn_reasoning_tokens": max((row["reasoning_tokens"] for row in trajectory_turns), default=0),
            "max_self_repeat_ratio": max((row["self_repeat_ratio"] for row in trajectory_turns), default=0.0),
            "max_cross_turn_similarity": max((row["cross_turn_similarity"] for row in trajectory_turns), default=0.0),
            "max_tool_overlap_ratio": max((row["tool_overlap_ratio"] for row in trajectory_turns), default=0.0),
            "literal_tool_call_turns": sum(row["literal_tool_call"] for row in trajectory_turns),
            "labels": labels,
        })
    return trajectory_rows, turn_rows


def build_summary(trajectory_rows: Sequence[dict[str, Any]], turn_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_reasoning = sum(row["reasoning_tokens"] for row in trajectory_rows)
    ineffective = sum(row["estimated_ineffective_tokens"] for row in trajectory_rows)
    high_risk = sum(row["high_risk"] for row in trajectory_rows)
    label_counts = Counter(label for row in trajectory_rows for label in row["labels"])
    lengths = [row["reasoning_tokens"] for row in trajectory_rows]
    repeat_ratios = [row["estimated_ineffective_ratio"] for row in trajectory_rows]
    return {
        "schema_version": 1,
        "population": {
            "trajectories": len(trajectory_rows),
            "reasoning_turns": len(turn_rows),
            "train": sum(row["split"] == "train" for row in trajectory_rows),
            "validation": sum(row["split"] == "validation" for row in trajectory_rows),
        },
        "reasoning": {
            "total_tokens": total_reasoning,
            "estimated_ineffective_tokens": ineffective,
            "estimated_ineffective_ratio": round(ineffective / total_reasoning, 6) if total_reasoning else 0.0,
            "trajectory_tokens": {
                "p50": round(percentile(lengths, 0.50)),
                "p90": round(percentile(lengths, 0.90)),
                "p95": round(percentile(lengths, 0.95)),
                "p99": round(percentile(lengths, 0.99)),
                "max": max(lengths, default=0),
            },
            "trajectory_ineffective_ratio": {
                "p50": round(percentile(repeat_ratios, 0.50), 6),
                "p90": round(percentile(repeat_ratios, 0.90), 6),
                "p95": round(percentile(repeat_ratios, 0.95), 6),
                "max": max(repeat_ratios, default=0.0),
            },
        },
        "risk": {
            "high_risk_trajectories": high_risk,
            "high_risk_ratio": round(high_risk / len(trajectory_rows), 6) if trajectory_rows else 0.0,
            "trajectory_label_counts": dict(sorted(label_counts.items())),
        },
        "thresholds": {
            "token_ngram_size": NGRAM_SIZE,
            "self_loop_min_tokens": SELF_LOOP_MIN_TOKENS,
            "self_loop_min_ratio": SELF_LOOP_MIN_RATIO,
            "cross_turn_min_tokens": CROSS_TURN_MIN_TOKENS,
            "cross_turn_min_similarity": CROSS_TURN_MIN_SIMILARITY,
            "tool_copy_min_tokens": TOOL_COPY_MIN_TOKENS,
            "tool_copy_min_ratio": TOOL_COPY_MIN_RATIO,
        },
        "interpretation": "Candidate detection only; labels do not authorize automatic data deletion.",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_top_cases(path: Path, trajectories: Sequence[dict[str, Any]]) -> None:
    ranked = sorted(
        trajectories,
        key=lambda row: (
            row["high_risk"], row["estimated_ineffective_tokens"], row["reasoning_tokens"]
        ),
        reverse=True,
    )[:100]
    fields = [
        "instance_id", "seed", "split", "source_file", "source_line", "reasoning_tokens",
        "estimated_ineffective_tokens", "estimated_ineffective_ratio", "high_risk_turns",
        "max_self_repeat_ratio", "max_cross_turn_similarity", "max_tool_overlap_ratio",
        "literal_tool_call_turns", "labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow({**row, "labels": ";".join(row["labels"])})


def write_manual_review_sample(path: Path, trajectories: Sequence[dict[str, Any]]) -> None:
    """Create the four disjoint, deterministic review strata from the plan."""
    selected: dict[int, tuple[str, dict[str, Any]]] = {}

    def add(group: str, rows: Iterable[dict[str, Any]], limit: int) -> None:
        count = 0
        for row in rows:
            key = row["trajectory_index"]
            if key in selected:
                continue
            selected[key] = (group, row)
            count += 1
            if count == limit:
                return

    add(
        "highest_repetition",
        sorted(trajectories, key=lambda row: (row["estimated_ineffective_ratio"], row["estimated_ineffective_tokens"]), reverse=True),
        20,
    )
    add(
        "long_low_repetition",
        sorted(
            (row for row in trajectories if row["estimated_ineffective_ratio"] < 0.10),
            key=lambda row: row["reasoning_tokens"],
            reverse=True,
        ),
        10,
    )
    add(
        "highest_tool_copying",
        sorted(trajectories, key=lambda row: row["max_tool_overlap_ratio"], reverse=True),
        10,
    )
    normal = [row for row in trajectories if not row["high_risk"]]
    normal.sort(key=lambda row: hashlib.sha256(row["trajectory_fingerprint"].encode()).hexdigest())
    add("deterministic_normal_sample", normal, 20)

    fields = [
        "review_group", "review_decision", "review_notes", "instance_id", "seed", "split",
        "source_file", "source_line", "reasoning_tokens", "estimated_ineffective_tokens",
        "estimated_ineffective_ratio", "max_self_repeat_ratio", "max_cross_turn_similarity",
        "max_tool_overlap_ratio", "literal_tool_call_turns", "labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for _, (group, row) in selected.items():
            writer.writerow({
                **row,
                "review_group": group,
                "review_decision": "",
                "review_notes": "",
                "labels": ";".join(row["labels"]),
            })


def _report_data(trajectories: Sequence[dict[str, Any]], turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    top_ids = {
        row["trajectory_index"]
        for row in sorted(
            trajectories,
            key=lambda item: (item["high_risk"], item["estimated_ineffective_tokens"], item["reasoning_tokens"]),
            reverse=True,
        )[:80]
    }
    compact_trajectories = [{
        key: row[key] for key in (
            "trajectory_index", "split", "instance_id", "seed", "reasoning_tokens",
            "estimated_ineffective_tokens", "estimated_ineffective_ratio", "high_risk",
            "labels", "source_file", "source_line",
        )
    } for row in trajectories]
    compact_turns = [{
        key: row[key] for key in (
            "trajectory_index", "assistant_turn", "message_index", "reasoning_tokens",
            "self_repeat_ratio", "cross_turn_similarity", "tool_overlap_ratio", "high_risk",
            "labels", "reasoning_excerpt", "tool_context_excerpt",
        )
    } for row in turns if row["trajectory_index"] in top_ids]
    return {"trajectories": compact_trajectories, "turns": compact_turns}


def write_report(path: Path, summary: dict[str, Any], trajectories, turns) -> None:
    data = json.dumps(_report_data(trajectories, turns), ensure_ascii=False).replace("</", "<\\/")
    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher reasoning repetition</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1b1d20;--muted:#667085;--card:#f5f6f8;--line:#d8dce3;--blue:#2878d0;--orange:#d56a17;--red:#c33;--green:#198754}
@media(prefers-color-scheme:dark){:root{--bg:#181818;--fg:#f4f4f4;--muted:#aeb4bf;--card:#24262a;--line:#3a3d43;--blue:#78b7ff;--orange:#ffad66;--red:#ff7777;--green:#71d69a}}
*{box-sizing:border-box}body{margin:0;padding:20px;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}main{max-width:1280px;margin:auto}h1,h2{font-weight:500}h1{margin:0 0 12px}h2{margin:24px 0 8px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.stat{background:var(--card);padding:12px;border-radius:8px}.stat b{display:block;font-size:24px;font-weight:500}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.controls label{display:flex;gap:6px;align-items:center}select,input{font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);padding:6px;border-radius:5px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart{min-height:280px}.chart svg{width:100%;height:280px;overflow:visible}.axis{stroke:var(--line)}.dot{fill:var(--blue);opacity:.55}.dot.risk{fill:var(--red);opacity:.75}.bar{fill:var(--orange)}.label{fill:var(--muted);font-size:11px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px;border-bottom:1px solid var(--line);vertical-align:top}button{font:inherit;color:var(--blue);background:none;border:0;padding:0;text-align:left}details{max-width:700px}pre{white-space:pre-wrap;max-height:320px;overflow:auto;background:var(--card);padding:10px}.badge{display:inline-block;padding:1px 6px;margin:1px;border-radius:999px;background:var(--card)}@media(max-width:760px){.charts{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Teacher reasoning 重复与冗余</h1>
<div class="stats" id="stats"></div>
<div class="controls"><label>Split <select id="split"><option value="all">全部</option><option>train</option><option>validation</option></select></label><label>风险 <select id="risk"><option value="all">全部</option><option value="true">高风险</option><option value="false">非高风险</option></select></label><label>标签 <select id="label"><option value="all">全部</option></select></label><label>最小 reasoning token <input id="minTokens" type="number" min="0" value="0"></label></div>
<div class="charts"><section class="chart"><h2>Reasoning token 与估计无效 token</h2><svg id="scatter" role="img" aria-label="reasoning token 与估计无效 token 散点图"></svg></section><section class="chart"><h2>无效 token 比例分布</h2><svg id="hist" role="img" aria-label="无效 token 比例直方图"></svg></section></div>
<h2>轨迹明细</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>Tokens</th><th>估计无效</th><th>标签</th><th>来源</th></tr></thead><tbody id="rows"></tbody></table></div>
</main><script>
const DATA=__DATA__, SUMMARY=__SUMMARY__; const $=id=>document.getElementById(id); const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels=[...new Set(DATA.trajectories.flatMap(x=>x.labels))].sort(); labels.forEach(x=>$('label').insertAdjacentHTML('beforeend',`<option>${esc(x)}</option>`));
function filtered(){const split=$('split').value,risk=$('risk').value,label=$('label').value,min=+$('minTokens').value||0;return DATA.trajectories.filter(x=>(split==='all'||x.split===split)&&(risk==='all'||String(x.high_risk)===risk)&&(label==='all'||x.labels.includes(label))&&x.reasoning_tokens>=min)}
function svgEl(tag,attrs={}){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}
function scatter(rows){const s=$('scatter');s.replaceChildren();const W=600,H=250,p=34,maxX=Math.max(1,...rows.map(x=>x.reasoning_tokens)),maxY=Math.max(1,...rows.map(x=>x.estimated_ineffective_tokens));s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.append(svgEl('line',{x1:p,y1:H-p,x2:W-8,y2:H-p,class:'axis'}),svgEl('line',{x1:p,y1:8,x2:p,y2:H-p,class:'axis'}));rows.forEach(x=>s.append(svgEl('circle',{cx:p+x.reasoning_tokens/maxX*(W-p-12),cy:H-p-x.estimated_ineffective_tokens/maxY*(H-p-12),r:3,class:x.high_risk?'dot risk':'dot'})));[['0',p,H-10],[String(maxX),W-45,H-10],[String(maxY),2,14]].forEach(([t,x,y])=>{const e=svgEl('text',{x,y,class:'label'});e.textContent=t;s.append(e)})}
function hist(rows){const s=$('hist');s.replaceChildren();const W=600,H=250,p=34,bins=20,c=Array(bins).fill(0);rows.forEach(x=>c[Math.min(bins-1,Math.floor(x.estimated_ineffective_ratio*bins))]++);const max=Math.max(1,...c);s.setAttribute('viewBox',`0 0 ${W} ${H}`);s.append(svgEl('line',{x1:p,y1:H-p,x2:W-8,y2:H-p,class:'axis'}));c.forEach((n,i)=>s.append(svgEl('rect',{x:p+i*(W-p-12)/bins+1,y:H-p-n/max*(H-p-12),width:(W-p-12)/bins-2,height:n/max*(H-p-12),class:'bar'})));[['0%',p,H-10],['100%',W-45,H-10],[String(max),2,14]].forEach(([t,x,y])=>{const e=svgEl('text',{x,y,class:'label'});e.textContent=t;s.append(e)})}
function turnDetails(id){return DATA.turns.filter(x=>x.trajectory_index===id).sort((a,b)=>b.reasoning_tokens-a.reasoning_tokens).slice(0,5).map(x=>`<details><summary>turn ${x.assistant_turn} · ${x.reasoning_tokens} tokens · ${x.labels.map(y=>`<span class="badge">${esc(y)}</span>`).join('')}</summary><h3>Reasoning</h3><pre>${esc(x.reasoning_excerpt)}</pre>${x.tool_context_excerpt?`<h3>前序 tool context</h3><pre>${esc(x.tool_context_excerpt)}</pre>`:''}</details>`).join('')}
function render(){const rows=filtered().sort((a,b)=>b.estimated_ineffective_tokens-a.estimated_ineffective_tokens);const total=rows.reduce((n,x)=>n+x.reasoning_tokens,0),bad=rows.reduce((n,x)=>n+x.estimated_ineffective_tokens,0),risk=rows.filter(x=>x.high_risk).length;$('stats').innerHTML=`<div class="stat"><span>轨迹</span><b>${rows.length}</b></div><div class="stat"><span>Reasoning tokens</span><b>${total.toLocaleString()}</b></div><div class="stat"><span>估计无效 tokens</span><b>${bad.toLocaleString()}</b><small>${total?(bad/total*100).toFixed(2):0}%</small></div><div class="stat"><span>高风险轨迹</span><b>${risk}</b></div>`;scatter(rows);hist(rows);$('rows').innerHTML=rows.slice(0,100).map(x=>`<tr><td><button type="button" onclick="this.nextElementSibling.open=!this.nextElementSibling.open">${esc(x.instance_id)} · seed ${x.seed}</button>${turnDetails(x.trajectory_index)}</td><td>${x.reasoning_tokens.toLocaleString()}</td><td>${x.estimated_ineffective_tokens.toLocaleString()} (${(x.estimated_ineffective_ratio*100).toFixed(1)}%)</td><td>${x.labels.map(y=>`<span class="badge">${esc(y)}</span>`).join('')}</td><td>${esc(x.source_file)}:${esc(x.source_line)}</td></tr>`).join('')}
['split','risk','label','minTokens'].forEach(id=>$(id).addEventListener('input',render));render();
</script></body></html>'''
    path.write_text(template.replace("__DATA__", data).replace("__SUMMARY__", summary_json), encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any], command: str) -> None:
    population = summary["population"]
    reasoning = summary["reasoning"]
    risk = summary["risk"]
    text = f"""# Teacher reasoning repetition audit

This directory contains a read-only candidate audit of the exact pretokenized
Full SFT population selected by trajectory identity.

## Population

- trajectories: {population['trajectories']} ({population['train']} train, {population['validation']} validation)
- reasoning turns: {population['reasoning_turns']}
- reasoning tokens: {reasoning['total_tokens']}
- candidate ineffective tokens: {reasoning['estimated_ineffective_tokens']} ({reasoning['estimated_ineffective_ratio']:.2%})
- high-risk trajectories: {risk['high_risk_trajectories']} ({risk['high_risk_ratio']:.2%})

Labels are candidate flags, not deletion decisions. `legitimate_code_reference`
is retained specifically to separate code/context quotation from model loops.

## Reproduce

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-train", type=Path, required=True)
    parser.add_argument("--filtered-validation", type=Path, required=True)
    parser.add_argument("--tokenized-train", type=Path, required=True)
    parser.add_argument("--tokenized-validation", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=694)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    accepted = load_accepted_identities([args.tokenized_train, args.tokenized_validation])
    if len(accepted) != args.expected_count:
        raise SystemExit(f"accepted identity count is {len(accepted)}, expected {args.expected_count}")
    records = load_filtered_records([args.filtered_train, args.filtered_validation], accepted)
    tokenizer = TokenizerAdapter(AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    ))
    trajectories, turns = analyze_records(records, tokenizer)
    summary = build_summary(trajectories, turns)
    if summary["population"]["trajectories"] != args.expected_count:
        raise SystemExit("analysis population changed unexpectedly")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "trajectory_repetition_metrics.jsonl", trajectories)
    write_jsonl(args.output_dir / "turn_repetition_metrics.jsonl", turns)
    write_top_cases(args.output_dir / "top_repetition_cases.csv", trajectories)
    write_manual_review_sample(args.output_dir / "manual_review_sample.csv", trajectories)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.html", summary, trajectories, turns)
    command = (
        "python R2E-Gym/scripts/analyze_teacher_reasoning_repetition.py "
        f"--filtered-train {args.filtered_train} --filtered-validation {args.filtered_validation} "
        f"--tokenized-train {args.tokenized_train} --tokenized-validation {args.tokenized_validation} "
        f"--tokenizer {args.tokenizer} --output-dir {args.output_dir}"
    )
    write_readme(args.output_dir / "README.md", summary, command)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
