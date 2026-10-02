#!/usr/bin/env python3
"""用 Qwen3 tokenizer 审计长度，并生成 SWE-Master SFT 可直接读取的数据。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm


_TOKENIZER = None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout:
            json.dump(value, fout, ensure_ascii=False, indent=2, sort_keys=True)
            fout.write("\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def pending_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return os.fdopen(fd, "w", encoding="utf-8"), Path(pending)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )


def token_length(text: str) -> int:
    return len(_TOKENIZER(text, add_special_tokens=False)["input_ids"])


def convert_record(item: tuple[int, str, int]) -> dict:
    line_number, line, max_length = item
    try:
        sample = json.loads(line)
        messages = sample["input"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("input messages missing")
        if messages[-1].get("role") != "assistant" or \
                "<function=submit>" not in messages[-1].get("content", ""):
            raise ValueError("final submit missing")

        # 与 SWE-Master 的 sft_data_pre_tokenize.py 保持相同语义：
        # prompt + response 是完整轨迹，response_ranges 标出每轮 assistant。
        prompt = _TOKENIZER.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = _TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if not full_text.startswith(prompt):
            raise ValueError("chat template prompt is not a prefix")
        response = full_text[len(prompt):]
        prompt_ids_len = token_length(prompt)

        response_ranges = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            assistant_prompt = _TOKENIZER.apply_chat_template(
                messages[:index], tokenize=False, add_generation_prompt=True
            )
            assistant_through = _TOKENIZER.apply_chat_template(
                messages[: index + 1], tokenize=False,
                add_generation_prompt=False,
            )
            if not assistant_through.startswith(assistant_prompt):
                raise ValueError("assistant response is not a prompt suffix")
            start = token_length(assistant_prompt)
            end = start + token_length(
                assistant_through[len(assistant_prompt):]
            ) - 1
            response_ranges.append([start, end])

        if not prompt or not response or not response_ranges:
            raise ValueError("empty prompt, response, or assistant ranges")

        # 与 SFTDataset.__getitem__ 的最终训练文本完全一致。
        train_text = (prompt + response).rstrip("\n")
        if not train_text.endswith(_TOKENIZER.eos_token):
            train_text += " " + _TOKENIZER.eos_token
        sequence_length = token_length(train_text)
        # SFTDataset 使用 loss_mask[start - 1:end]；end 是切片的排他上界。
        # 最后一轮因 rstrip + EOS 替换可能相差 1 个 token，按最终训练
        # 序列长度截断，保持与实际 __getitem__ 行为一致。
        response_ranges = [
            [start, min(end, sequence_length)]
            for start, end in response_ranges
        ]
        loss_tokens = sum(end - start + 1 for start, end in response_ranges)

        if sequence_length > max_length:
            return {
                "status": "over_max_length",
                "source_line": line_number,
                "instance_id": sample.get("instance_id"),
                "exp_name": sample.get("exp_name"),
                "qwen3_sequence_length": sequence_length,
                "max_length": max_length,
                "reason": "qwen3_sequence_over_80k",
            }

        sample.update({
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges,
            "qwen3_sequence_length": sequence_length,
            "qwen3_assistant_loss_tokens": loss_tokens,
        })
        return {"status": "kept", "sample": sample}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "invalid",
            "source_line": line_number,
            "reason": f"token_audit_error:{type(exc).__name__}",
            "detail": str(exc),
        }


def audit(input_path: Path, output_dir: Path, tokenizer_path: str,
          max_length: int = 81920, workers: int = 1) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_qwen3_80k.jsonl"
    rejected_path = output_dir / "token_length_rejected.jsonl"
    report_path = output_dir / "token_length_report.json"
    train_file, train_pending = pending_jsonl(train_path)
    rejected_file, rejected_pending = pending_jsonl(rejected_path)

    stats = Counter()
    length_buckets = Counter()
    total_lines = count_lines(input_path)

    def work_items():
        with input_path.open("r", encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, start=1):
                yield line_number, line, max_length

    progress = tqdm(total=total_lines, desc="Qwen3 SFT preprocess", unit="traj")
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(tokenizer_path,),
        ) as executor:
            results = executor.map(convert_record, work_items(), chunksize=1)
            for result in results:
                status = result.pop("status")
                stats["total"] += 1
                stats[status] += 1
                if status == "kept":
                    sample = result["sample"]
                    sequence_length = sample["qwen3_sequence_length"]
                    bucket_start = sequence_length // 8192 * 8192
                    length_buckets[
                        f"{bucket_start}-{bucket_start + 8191}"
                    ] += 1
                    train_file.write(
                        json.dumps(sample, ensure_ascii=False) + "\n"
                    )
                else:
                    rejected_file.write(
                        json.dumps(result, ensure_ascii=False) + "\n"
                    )
                progress.update(1)
                if stats["total"] % 10 == 0 or stats["total"] == total_lines:
                    progress.set_postfix(
                        kept=stats["kept"],
                        rejected=stats["over_max_length"] + stats["invalid"],
                    )

        for fout in (train_file, rejected_file):
            fout.flush()
            os.fsync(fout.fileno())
            fout.close()
        progress.close()
        os.replace(train_pending, train_path)
        os.replace(rejected_pending, rejected_path)
    except BaseException:
        progress.close()
        train_file.close()
        rejected_file.close()
        train_pending.unlink(missing_ok=True)
        rejected_pending.unlink(missing_ok=True)
        raise

    report = {
        "input": str(input_path.resolve()),
        "tokenizer": tokenizer_path,
        "max_length": max_length,
        "workers": workers,
        "counts": dict(stats),
        "sequence_length_histogram": dict(sorted(length_buckets.items())),
        "outputs": {
            "train": train_path.name,
            "rejected": rejected_path.name,
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=81920)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)),
        help="并行 tokenizer 进程数；8 核 32G 建议使用 6",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须大于等于 1")
    audit(
        args.input_path, args.output_dir, args.tokenizer,
        args.max_length, args.workers,
    )


if __name__ == "__main__":
    main()
