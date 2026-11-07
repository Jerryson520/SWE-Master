#!/usr/bin/env python3
"""Score EOS/tool-boundary tokens for fixed Base-vs-SFT trajectory prefixes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


CHECKPOINTS = (0, 128, 512, 1024, 2048, 4096, 8191)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-endpoints", default="http://127.0.0.1:18000,http://127.0.0.1:18001")
    parser.add_argument("--sft-endpoints", default="http://127.0.0.1:18002,http://127.0.0.1:18003")
    parser.add_argument("--base-model", default="qwen3-base")
    parser.add_argument("--sft-model", default="qwen3-full-sft")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = {}
                function["arguments"] = parsed if isinstance(parsed, dict) else {}
    return normalized


def as_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def render_case(
    tokenizer: Any,
    case: dict[str, Any],
    im_end_id: int,
    assistant_end_ids: list[int],
) -> tuple[list[int], list[int]]:
    context = normalize_messages(case["context_messages"])
    reference = normalize_messages([case["reference_assistant"]])[0]
    tools = case.get("tools") or []
    context_ids = as_ids(
        tokenizer.apply_chat_template(
            context,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    full_ids = as_ids(
        tokenizer.apply_chat_template(
            context + [reference],
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full_ids[: len(context_ids)] != context_ids:
        raise ValueError(f"Chat-template prefix mismatch for {case['case_id']}")
    reference_ids = full_ids[len(context_ids) :]
    if assistant_end_ids and reference_ids[-len(assistant_end_ids) :] == assistant_end_ids:
        del reference_ids[-len(assistant_end_ids) :]
    elif reference_ids and reference_ids[-1] == im_end_id:
        reference_ids.pop()
    return context_ids, reference_ids


def positions_for(case: dict[str, Any], reference_length: int) -> list[int]:
    positions = [position for position in CHECKPOINTS if position < reference_length]
    if case["group"] == "sft_agent_control":
        positions.append(reference_length)
    return sorted(set(positions))


def ngram_metrics(token_ids: list[int]) -> dict[str, float | int]:
    def distinct(n: int) -> float:
        total = len(token_ids) - n + 1
        if total <= 0:
            return 1.0
        return len(set(zip(*(token_ids[offset:] for offset in range(n))))) / total

    total_eight = max(0, len(token_ids) - 7)
    unique_eight = 0 if total_eight == 0 else len(set(zip(*(token_ids[offset:] for offset in range(8)))))
    return {
        "distinct_4": distinct(4),
        "repeated_8gram_fraction": 0.0 if total_eight == 0 else 1.0 - unique_eight / total_eight,
        "unique_tokens": len(set(token_ids)),
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def token_score(top_logprobs: dict[str, float], token: str, token_id: int) -> float:
    if token in top_logprobs:
        return float(top_logprobs[token])
    token_key = f"token_id:{token_id}"
    if token_key in top_logprobs:
        return float(top_logprobs[token_key])
    raise KeyError(f"Requested token {token!r}/{token_id} missing from {top_logprobs}")


def score_endpoint(
    endpoint: str,
    model_name: str,
    variant: str,
    assignments: list[dict[str, Any]],
    eos_id: int,
    eos_token: str,
    tool_id: int,
    tool_token: str,
    timeout: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in assignments:
        case = item["case"]
        context_ids = item["context_ids"]
        reference_ids = item["reference_ids"]
        metrics = item["metrics"]
        for checkpoint in item["positions"]:
            prefix = context_ids + reference_ids[:checkpoint]
            requested_ids = [eos_id, tool_id]
            reference_next_id = None
            reference_next_token = None
            if checkpoint < len(reference_ids):
                reference_next_id = reference_ids[checkpoint]
                reference_next_token = item["tokenizer"].convert_ids_to_tokens(reference_next_id)
                requested_ids.append(reference_next_id)
            payload = {
                "model": model_name,
                "prompt": prefix,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 1,
                "logprob_token_ids": sorted(set(requested_ids)),
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            }
            response = post_json(endpoint, payload, timeout)
            choice = response["choices"][0]
            top = choice["logprobs"]["top_logprobs"][0]
            row = {
                "case_id": case["case_id"],
                "group": case["group"],
                "control_kind": case.get("control_kind"),
                "instance_id": case["instance_id"],
                "model_variant": variant,
                "model_name": model_name,
                "endpoint": endpoint,
                "checkpoint": checkpoint,
                "context_tokens": len(context_ids),
                "reference_tokens": len(reference_ids),
                "total_prefix_tokens": len(prefix),
                "eos_token_id": eos_id,
                "tool_call_token_id": tool_id,
                "eos_logprob": token_score(top, eos_token, eos_id),
                "tool_call_logprob": token_score(top, tool_token, tool_id),
                "reference_next_token_id": reference_next_id,
                "reference_next_logprob": (
                    token_score(top, reference_next_token, reference_next_id)
                    if reference_next_id is not None
                    else None
                ),
                "sampled_token_id": choice.get("token_ids", [None])[0],
                **metrics,
            }
            rows.append(row)
            print(
                f"{variant} {case['case_id']} checkpoint={checkpoint} "
                f"eos={row['eos_logprob']:.3f} tool={row['tool_call_logprob']:.3f}",
                flush=True,
            )
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["model_variant"], row["checkpoint"])].append(row)
    output = []
    for (group, variant, checkpoint), bucket in sorted(buckets.items()):
        result: dict[str, Any] = {
            "group": group,
            "model_variant": variant,
            "checkpoint": checkpoint,
            "n": len(bucket),
        }
        for field in ("eos_logprob", "tool_call_logprob", "reference_next_logprob"):
            values = [float(row[field]) for row in bucket if row[field] is not None]
            result[f"{field}_mean"] = statistics.fmean(values) if values else None
            result[f"{field}_median"] = statistics.median(values) if values else None
            result[f"{field}_p25"] = percentile(values, 0.25) if values else None
            result[f"{field}_p75"] = percentile(values, 0.75) if values else None
        output.append(result)
    return output


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[(row["case_id"], row["checkpoint"])][row["model_variant"]] = row
    deltas = []
    for (case_id, checkpoint), pair in sorted(pairs.items()):
        if set(pair) != {"base", "full_sft"}:
            continue
        base, sft = pair["base"], pair["full_sft"]
        deltas.append(
            {
                "case_id": case_id,
                "group": base["group"],
                "checkpoint": checkpoint,
                "sft_minus_base_eos_logprob": sft["eos_logprob"] - base["eos_logprob"],
                "sft_minus_base_tool_logprob": sft["tool_call_logprob"] - base["tool_call_logprob"],
            }
        )
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with open(args.input, encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    tool_ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    if len(eos_ids) != 1 or len(tool_ids) != 1:
        raise ValueError(f"Expected single-token boundaries, got EOS={eos_ids}, tool={tool_ids}")
    eos_id, tool_id = eos_ids[0], tool_ids[0]
    eos_token = tokenizer.convert_ids_to_tokens(eos_id)
    tool_token = tokenizer.convert_ids_to_tokens(tool_id)
    assistant_end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    prepared = []
    for case in cases:
        context_ids, reference_ids = render_case(
            tokenizer, case, eos_id, assistant_end_ids
        )
        positions = positions_for(case, len(reference_ids))
        if context_ids and positions and len(context_ids) + max(positions) >= 81920:
            raise ValueError(f"Prefix exceeds model limit for {case['case_id']}")
        prepared.append(
            {
                "case": case,
                "context_ids": context_ids,
                "reference_ids": reference_ids,
                "positions": positions,
                "metrics": ngram_metrics(reference_ids),
                "tokenizer": tokenizer,
            }
        )

    endpoint_specs = []
    for variant, model, endpoints_text in (
        ("base", args.base_model, args.base_endpoints),
        ("full_sft", args.sft_model, args.sft_endpoints),
    ):
        endpoints = [value.strip() for value in endpoints_text.split(",") if value.strip()]
        for endpoint_index, endpoint in enumerate(endpoints):
            assigned = [item for index, item in enumerate(prepared) if index % len(endpoints) == endpoint_index]
            endpoint_specs.append((endpoint, model, variant, assigned))

    rows = []
    with ThreadPoolExecutor(max_workers=len(endpoint_specs)) as executor:
        futures = [
            executor.submit(
                score_endpoint,
                endpoint,
                model,
                variant,
                assigned,
                eos_id,
                eos_token,
                tool_id,
                tool_token,
                args.timeout,
            )
            for endpoint, model, variant, assigned in endpoint_specs
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["group"], row["case_id"], row["checkpoint"], row["model_variant"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "boundary_logprobs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "boundary_logprobs.csv", rows)
    summaries = summarize(rows)
    write_csv(output_dir / "group_by_checkpoint.csv", summaries)
    deltas = paired_deltas(rows)
    write_csv(output_dir / "paired_deltas.csv", deltas)

    manifest = {
        "cases": len(cases),
        "groups": dict(Counter(case["group"] for case in cases)),
        "rows": len(rows),
        "eos_token": eos_token,
        "eos_token_id": eos_id,
        "tool_call_token": tool_token,
        "tool_call_token_id": tool_id,
        "checkpoints": list(CHECKPOINTS),
        "base_endpoints": args.base_endpoints,
        "sft_endpoints": args.sft_endpoints,
    }
    (output_dir / "probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
