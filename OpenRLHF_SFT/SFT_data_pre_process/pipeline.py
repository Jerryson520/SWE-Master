from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm

FC_SP = '''## Function Definition
- You have access to the following functions:

---- BEGIN FUNCTION #1: execute_bash ----
**Description**: Execute a bash command in the terminal within a persistent shell session.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.

**Parameters**:
(1) command (string, required): The bash command to execute. For example: `python my_script.py`. If not provided, will show help. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process. Note: You can only execute one bash command at a time. If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: str_replace_editor ----
**Description**: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* The following binary file extensions can be viewed in Markdown format: [\".xlsx\", \".pptx\", \".wav\", \".mp3\", \".m4a\", \".flac\", \".pdf\", \".docx\"]. IT DOES NOT HANDLE IMAGES.\n* The create command cannot be used if the specified path already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique.
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
* This tool can be used for creating and editing files in plain-text format.
* Before using this tool:
- 1. Use the view tool to understand the file's contents and context
- 2. Verify the directory path is correct (only applicable when creating new files):
    - Use the view tool to verify the parent directory exists and is the correct location
    - When making edits:
    - Ensure the edit results in idiomatic, correct code
    - Do not leave the code in a broken state
    - Always use absolute file paths (starting with /)
* Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.

**Parameters**:
(1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
(2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
(3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
(4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
(5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
(6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
(7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #2 ----


---- BEGIN FUNCTION #3: submit ----
**Description**: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #3 ----

- If you choose to call a function, ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
</IMPORTANT>
'''

FC_HINT = (
    "You forgot to use a function call in your response. "
)

DEFAULT_MAX_STEPS = 100
DEFAULT_MAX_TOKEN_USAGE = 81920

NATURAL_EXIT_REASON = "agent"

ALLOWED_TOOLS = {"execute_bash", "str_replace_editor", "submit"}
ALLOWED_EDITOR_COMMANDS = {"view", "create", "str_replace", "insert"}

FUNCTION_PATTERN = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL
)
PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL
)

VALID_EMPTY_SUBMIT_FORMS = (
    "<function=submit>\n</function>",
    "<function=submit></function>",
)


def _parse_action(action: object) -> tuple[str | None, dict[str, str], list[str]]:
    """解析一条 XML 风格 action，并返回工具名、参数和格式错误。"""
    if not isinstance(action, str):
        return None, {}, ["action_not_string"]

    matches = list(FUNCTION_PATTERN.finditer(action))
    if not matches:
        if "<function=" in action or "</function>" in action:
            return None, {}, ["malformed_tool_call"]
        return None, {}, ["missing_tool_call"]
    if len(matches) != 1:
        return None, {}, ["multiple_tool_calls"]

    match = matches[0]
    if action[:match.start()].strip() or action[match.end():].strip():
        return None, {}, ["content_outside_tool_call"]

    tool_name = match.group(1).strip()
    body = match.group(2)
    parameter_matches = list(PARAMETER_PATTERN.finditer(body))
    remaining = PARAMETER_PATTERN.sub("", body)
    reasons: list[str] = []
    if remaining.strip():
        reasons.append("malformed_tool_parameters")

    parameters: dict[str, str] = {}
    for parameter in parameter_matches:
        name = parameter.group(1).strip()
        if not name or name in parameters:
            reasons.append("duplicate_or_empty_parameter")
            continue
        parameters[name] = parameter.group(2)

    return tool_name, parameters, reasons


def validate_trajectory_steps(trajectory_steps: object) -> list[str]:
    """严格检查整条轨迹中每一步的工具名、调用数量和必要参数。"""
    if not isinstance(trajectory_steps, list):
        return ["trajectory_steps_not_list"]
    if not trajectory_steps:
        return ["empty_trajectory"]

    reasons: set[str] = set()
    for step in trajectory_steps:
        if not isinstance(step, dict):
            reasons.add("step_not_object")
            continue

        tool_name, parameters, action_reasons = _parse_action(step.get("action"))
        reasons.update(action_reasons)
        if tool_name is None:
            continue
        if tool_name not in ALLOWED_TOOLS:
            reasons.add(f"unknown_tool:{tool_name}")
            continue

        if tool_name == "execute_bash":
            # command 可以是空字符串；框架用它轮询仍在运行的上一条命令。
            if "command" not in parameters:
                reasons.add("execute_bash_missing_command")
        elif tool_name == "str_replace_editor":
            command = parameters.get("command")
            if command not in ALLOWED_EDITOR_COMMANDS:
                reasons.add("invalid_editor_command")
            if not parameters.get("path", "").strip():
                reasons.add("editor_missing_path")
            if command == "create" and "file_text" not in parameters:
                reasons.add("create_missing_file_text")
            if command == "str_replace" and "old_str" not in parameters:
                reasons.add("str_replace_missing_old_str")
            if command == "insert":
                if "insert_line" not in parameters:
                    reasons.add("insert_missing_line")
                if "new_str" not in parameters:
                    reasons.add("insert_missing_new_str")
        elif parameters:
            reasons.add("submit_with_parameters")

    return sorted(reasons)

def convert_record(raw: dict) -> dict:
    """
    convert raw r2e trajectory to openrlhf trajectory
    """
    problem_statement = raw["problem_statement"]
    reward = raw["reward"]
    trajectory_steps = raw.get("trajectory_steps")
    agent_args = raw["agent_args"]
    ds = raw.get("ds")
    if not isinstance(ds, dict):
        raise TypeError("ds must be an object")
    instance_id = ds.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id is missing")
    docker_image = raw.get("docker_image")

    instance_prompt = agent_args["instance_prompt"]
    user_prompt = (
        instance_prompt
        .replace("{working_dir}", "/testbed")
        .replace("{problem_statement}", problem_statement)
    )

    messages = [
        {
            "role": "system",
            "content": agent_args["system_prompt"],
        },
        {
            "role": "system",
            "content": FC_SP,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    for step in trajectory_steps:
        assistant_content = step["thought"] + "\n\n" + step["action"]
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "tool", "content": step["observation"]})

    last_step = trajectory_steps[-1]
    step_count = last_step["step_count"]
    token_usage_total = last_step["token_usage_total"]

    return {
        "docker_image": docker_image,
        "instance_id": instance_id,
        "input": messages,
        "step_count": step_count,
        "token_usage_total": token_usage_total,
        "reward": reward,
        "exit_reason": raw.get("exit_reason"),
        "exp_name": raw.get("exp_name"),
    }

def filter_sample(sample: dict) -> tuple[dict | None, list[str]]:
    """过滤一条转换后的轨迹，返回 ``(保留的样本, 淘汰原因)``。

    通过筛选时会删除最后一条 tool observation，使训练数据以 assistant 的
    ``submit`` 调用结尾；未通过时返回 ``None`` 和全部已发现的原因。
    """
    reasons: list[str] = []

    # 先验证最基本的数据结构，避免后面的倒数索引和字段访问抛异常。
    raw_inputs = sample.get("input")
    if not isinstance(raw_inputs, list):
        return None, ["input_not_list"]
    if len(raw_inputs) < 2:
        return None, ["too_few_messages"]
    if not all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        for message in raw_inputs
    ):
        return None, ["invalid_message"]

    # 复制消息，避免筛选过程修改调用方传入的原始 sample。
    inputs = [message.copy() for message in raw_inputs]

    # 条件 1：只保留成功轨迹，失败轨迹不能作为正向 SFT 示范。
    if sample.get("reward") != 1:
        reasons.append("reward_not_one")

    # 条件 2：排除依赖环境提醒后才补 function call 的轨迹。
    # 注意这里必须检查 content；role 只会是 system/user/assistant/tool。
    all_content_before_final_tool = "".join(
        message["content"] for message in inputs[:-1]
    )
    if FC_HINT in all_content_before_final_tool:
        reasons.append("contains_function_call_hint")

    # 条件 3：只保留自然调用 submit 结束的轨迹。预算、上下文、超时或模型
    # 请求异常导致的终止即使最终 reward == 1，也不是干净的行为克隆示范。
    if sample.get("exit_reason") != NATURAL_EXIT_REASON:
        reasons.append("non_natural_exit")

    # 条件 4：SWE-Master 论文为降低长尾样本带来的 OOM 风险，只保留不超过
    # 100 turns 的成功轨迹。自然在第 100 步 submit 可以保留。
    step_count = sample.get("step_count")
    if not isinstance(step_count, int):
        reasons.append("invalid_step_count")
    elif step_count > DEFAULT_MAX_STEPS:
        reasons.append("too_many_steps")

    # 条件 5：teacher token_usage_total 是最后一步模型请求的上下文估算值，
    # 用于审计是否接近采集预算，但不等同于 Qwen3 chat template 展开后的
    # 实际训练长度。最终 80K 判断必须在训练机上用 Qwen3 tokenizer 重算。
    token_usage_total = sample.get("token_usage_total")
    if not isinstance(token_usage_total, (int, float)):
        reasons.append("invalid_token_usage_total")

    # 条件 6：最后两条必须是 assistant action + tool observation。
    final_assistant = inputs[-2]
    final_tool = inputs[-1]
    if final_assistant["role"] != "assistant":
        reasons.append("penultimate_role_not_assistant")
    if final_tool["role"] != "tool":
        reasons.append("final_role_not_tool")

    # 条件 7：最终 assistant action 必须调用无参数 submit。
    final_content = final_assistant["content"]
    if "<function=submit>" not in final_content:
        reasons.append("final_action_not_submit")
    elif not any(
        submit_form in final_content for submit_form in VALID_EMPTY_SUBMIT_FORMS
    ):
        reasons.append("invalid_submit_format")

    if reasons:
        return None, reasons

    # 最后的 tool observation 通常只是 Finish。训练样本应以模型生成的
    # assistant submit action 结束，因此保存时去掉该 observation。
    filtered = sample.copy()
    filtered["input"] = inputs[:-1]
    return filtered, []

def trajectory_fingerprint(raw: dict) -> str:
    """生成与批次位置无关的稳定指纹，用于去掉恢复/合并产生的重复轨迹。"""
    payload = {
        "instance_id": raw.get("ds", {}).get("instance_id")
        if isinstance(raw.get("ds"), dict) else None,
        "trajectory_steps": raw.get("trajectory_steps"),
        "output_patch": raw.get("output_patch"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
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


def _pending_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return os.fdopen(fd, "w", encoding="utf-8"), Path(pending)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def run_pipeline(input_dir: str, output_dir: str, strategy: str = "rsft"):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")
    if strategy != "rsft":
        raise ValueError(f"暂不支持的筛选策略: {strategy}")

    batch_files = sorted(input_dir.glob("batch-*.jsonl"))
    if not batch_files:
        raise FileNotFoundError(
            f"输入目录中没有 batch-*.jsonl: {input_dir}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "clean_success_all.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "report.json"
    clean_file, clean_pending = _pending_jsonl(clean_path)
    rejected_file, rejected_pending = _pending_jsonl(rejected_path)

    stats = Counter()
    repository_counts = Counter()
    kept_repository_counts = Counter()
    kept_per_task = Counter()
    step_histogram = Counter()
    teacher_context_histogram = Counter()
    seen_fingerprints: set[str] = set()
    total_lines = sum(_count_lines(batch_file) for batch_file in batch_files)
    progress = tqdm(total=total_lines, desc="RSFT filter", unit="traj")

    def reject(raw: object, source_file: Path, line_number: int,
               reasons: list[str]) -> None:
        reasons = list(dict.fromkeys(reasons))
        stats["rejected"] += 1
        for reason in reasons:
            stats[f"reason:{reason}"] += 1
        raw_dict = raw if isinstance(raw, dict) else {}
        ds = raw_dict.get("ds") if isinstance(raw_dict.get("ds"), dict) else {}
        record = {
            "source_file": source_file.name,
            "source_line": line_number,
            "instance_id": ds.get("instance_id"),
            "exp_name": raw_dict.get("exp_name"),
            "reward": raw_dict.get("reward"),
            "exit_reason": raw_dict.get("exit_reason"),
            "step_count": (
                raw_dict.get("trajectory_steps", [{}])[-1].get("step_count")
                if isinstance(raw_dict.get("trajectory_steps"), list)
                and raw_dict.get("trajectory_steps")
                and isinstance(raw_dict["trajectory_steps"][-1], dict)
                else None
            ),
            "reasons": reasons,
        }
        rejected_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        for batch_file in batch_files:
            with batch_file.open("r", encoding="utf-8") as fin:
                for line_number, line in enumerate(fin, start=1):
                    stats["total"] += 1
                    progress.update(1)
                    if stats["total"] % 25 == 0 or stats["total"] == total_lines:
                        progress.set_postfix(
                            kept=stats["kept"], rejected=stats["rejected"]
                        )

                    if not line.strip():
                        reject({}, batch_file, line_number, ["empty_line"])
                        continue

                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        reject({}, batch_file, line_number, ["invalid_json"])
                        continue

                    if not isinstance(raw, dict):
                        reject(raw, batch_file, line_number, ["record_not_object"])
                        continue

                    ds = raw.get("ds") if isinstance(raw.get("ds"), dict) else {}
                    repo_name = ds.get("repo_name") or ds.get("repo")
                    if isinstance(repo_name, str) and repo_name:
                        repository_counts[repo_name] += 1

                    fingerprint = trajectory_fingerprint(raw)
                    if fingerprint in seen_fingerprints:
                        reject(raw, batch_file, line_number,
                               ["duplicate_trajectory"])
                        continue
                    seen_fingerprints.add(fingerprint)

                    trajectory_steps = raw.get("trajectory_steps")
                    if not isinstance(trajectory_steps, list):
                        reject(raw, batch_file, line_number,
                               ["trajectory_steps_not_list"])
                        continue

                    if not trajectory_steps:
                        reject(raw, batch_file, line_number, ["empty_trajectory"])
                        continue

                    action_reasons = validate_trajectory_steps(trajectory_steps)

                    try:
                        converted = convert_record(raw)
                    except (KeyError, TypeError, ValueError) as exc:
                        reject(raw, batch_file, line_number,
                               [f"convert_error:{type(exc).__name__}"])
                        continue

                    stats["converted"] += 1

                    filtered, reasons = filter_sample(converted)
                    reasons = list(dict.fromkeys(action_reasons + reasons))

                    if reasons:
                        reject(raw, batch_file, line_number, reasons)
                        continue

                    filtered["source_file"] = batch_file.name
                    filtered["source_line"] = line_number
                    filtered["trajectory_fingerprint"] = fingerprint
                    clean_file.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
                    kept_per_task[filtered["instance_id"]] += 1
                    if isinstance(repo_name, str) and repo_name:
                        kept_repository_counts[repo_name] += 1
                    step_histogram[str(filtered["step_count"])] += 1
                    teacher_tokens = filtered["token_usage_total"]
                    if teacher_tokens >= DEFAULT_MAX_TOKEN_USAGE:
                        stats["teacher_context_at_or_over_80k"] += 1
                    bucket = f"{int(teacher_tokens) // 8192 * 8192}-{(int(teacher_tokens) // 8192 + 1) * 8192 - 1}"
                    teacher_context_histogram[bucket] += 1

        clean_file.flush()
        os.fsync(clean_file.fileno())
        rejected_file.flush()
        os.fsync(rejected_file.fileno())
        progress.set_postfix(kept=stats["kept"], rejected=stats["rejected"])
        clean_file.close()
        rejected_file.close()
        progress.close()
        os.replace(clean_pending, clean_path)
        os.replace(rejected_pending, rejected_path)
    except BaseException:
        progress.close()
        clean_file.close()
        rejected_file.close()
        clean_pending.unlink(missing_ok=True)
        rejected_pending.unlink(missing_ok=True)
        raise

    report = {
        "strategy": strategy,
        "input_dir": str(input_dir.resolve()),
        "batch_files": [path.name for path in batch_files],
        "outputs": {
            "clean_success_all": clean_path.name,
            "rejected": rejected_path.name,
        },
        "limits": {
            "max_steps": DEFAULT_MAX_STEPS,
            "qwen3_max_sequence_length": DEFAULT_MAX_TOKEN_USAGE,
            "qwen3_length_audited": False,
        },
        "counts": {
            "total": stats["total"],
            "converted": stats["converted"],
            "kept": stats["kept"],
            "rejected": stats["rejected"],
            "unique_tasks_kept": len(kept_per_task),
            "teacher_context_at_or_over_80k": stats[
                "teacher_context_at_or_over_80k"
            ],
        },
        "rejection_reasons": {
            name.removeprefix("reason:"): count
            for name, count in sorted(stats.items())
            if name.startswith("reason:")
        },
        "repository_counts": dict(sorted(repository_counts.items())),
        "kept_repository_counts": dict(sorted(kept_repository_counts.items())),
        "successful_rollouts_per_task": dict(sorted(Counter(
            kept_per_task.values()
        ).items())),
        "step_count_histogram": dict(sorted(
            step_histogram.items(), key=lambda item: int(item[0])
        )),
        "teacher_context_histogram": dict(sorted(teacher_context_histogram.items())),
    }
    _atomic_json(report_path, report)

    print(f"输入目录：{input_dir}")
    print(f"批次文件：{len(batch_files)}")
    print(f"输出目录：{output_dir}")
    print(f"原始样本：{stats['total']}")
    print(f"成功转换：{stats['converted']}")
    print(f"筛选保留：{stats['kept']}")
    print(f"筛选淘汰：{stats['rejected']}")

    print("\n淘汰原因：")
    for name, count in sorted(stats.items()):
        if name.startswith("reason:"):
            print(f"  {name.removeprefix('reason:')}: {count}")

    print("\n注意：clean_success_all.jsonl 尚未完成 Qwen3 80K 长度审计。")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="将 R2E-Gym teacher 轨迹转换为可审计的 RSFT 候选数据"
    )
    parser.add_argument(
        "input_dir",
        help="包含 batch-*.jsonl 的原始 trajectory 批次目录",
    )
    parser.add_argument("output_dir", help="输出 clean/rejected/report 的目录")
    parser.add_argument(
        "--strategy", choices=("rsft",), default="rsft",
        help="本轮固定使用成功轨迹 rejection sampling",
    )
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, strategy=args.strategy)

if __name__ == "__main__":
    main()
