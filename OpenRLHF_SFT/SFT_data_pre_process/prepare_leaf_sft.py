#!/usr/bin/env python3
"""Pretokenize Leaf conversations with exact assistant-token loss masks."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from rllm.agents.leaf_agent import _tool_prompt
from rllm.parser.chat_template.parser import QwenChatTemplateParser


def canonical_messages(messages):
    result = [dict(message) for message in messages]
    if not result or result[0].get("role") != "system":
        raise ValueError("Leaf conversation must start with a system message")
    result[0]["content"] = result[0]["content"] + "\n\n" + _tool_prompt()
    return result


def token_ids(tokenizer, parser, messages, add_generation_prompt=False):
    text = parser.parse(messages, add_generation_prompt=add_generation_prompt, is_first_msg=True)
    return tokenizer.encode(text, add_special_tokens=False)


def process(sample, tokenizer, max_length):
    messages = canonical_messages(sample["input"])
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=False)
    ids = token_ids(tokenizer, parser, messages)
    if len(ids) > max_length:
        return None
    loss_mask = [0] * len(ids)
    assistant_ranges = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before = token_ids(tokenizer, parser, messages[:index], add_generation_prompt=True)
        through = token_ids(tokenizer, parser, messages[: index + 1])
        start, end = len(before), len(through)
        if end < start or ids[:end] != through:
            raise ValueError(f"chat template is not prefix-stable at assistant message {index}")
        for position in range(start, end):
            loss_mask[position] = 1
        assistant_ranges.append([start, end])
    if not assistant_ranges or not any(loss_mask):
        return None
    return {
        **{key: sample.get(key) for key in ("instance_id", "repo", "seed", "schema_version")},
        "input_ids": ids,
        "loss_mask": loss_mask,
        "assistant_ranges": assistant_ranges,
        "num_tokens": len(ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=81920)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    accepted = rejected = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open() as source, output.open("w") as destination:
        for line_number, line in enumerate(source, 1):
            sample = json.loads(line)
            processed = process(sample, tokenizer, args.max_length)
            if processed is None:
                rejected += 1
                continue
            destination.write(json.dumps(processed, ensure_ascii=False) + "\n")
            accepted += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected, "output": str(output)}))


if __name__ == "__main__":
    main()
