import json
from collections import Counter

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

DEFAULT_STEP_LIMITS = {100, 120, 150}
DEFAULT_MAX_TOKEN_USAGE = 81920

VALID_EMPTY_SUBMIT_FORMS = (
    "<function=submit>\n</function>",
    "<function=submit></function>",
)

def convert_record(raw: dict) -> dict:
    """
    convert raw r2e trajectory to openrlhf trajectory
    """
    problem_statement = raw["problem_statement"]
    reward = raw["reward"]
    trajectory_steps = raw.get("trajectory_steps")
    agent_args = raw["agent_args"]
    ds = raw.get("ds")
    instance_id = ds.get("instance_id")
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

    # 复制消息，下面的脏数据修复不会修改调用方传入的原始 sample。
    inputs = [message.copy() for message in raw_inputs]

    # 修复官方数据中一种已知的空函数名终止 action。修复后丢弃它后面的
    # 多余一轮消息，让该 action 与紧随其后的 tool observation 成为结尾。
    if len(inputs) >= 4:
        compact_content = "".join(inputs[-4]["content"].split())
        if "<function=></function>" in compact_content:
            inputs[-4]["content"] = inputs[-4]["content"].replace(
                "<function=>", "<function=submit>"
            )
            inputs = inputs[:-2]

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

    # 条件 3：恰好在历史最大步数结束通常表示轨迹被预算强制截断。
    if sample.get("step_count") in DEFAULT_STEP_LIMITS:
        reasons.append("hit_step_limit")

    # 条件 4：排除累计 token 使用量达到历史 80K 上限的轨迹。
    token_usage_total = sample.get("token_usage_total")
    if not isinstance(token_usage_total, (int, float)):
        reasons.append("invalid_token_usage_total")
    elif token_usage_total >= DEFAULT_MAX_TOKEN_USAGE:
        reasons.append("hit_token_usage_limit")

    # 条件 5：最后两条必须是 assistant action + tool observation。
    final_assistant = inputs[-2]
    final_tool = inputs[-1]
    if final_assistant["role"] != "assistant":
        reasons.append("penultimate_role_not_assistant")
    if final_tool["role"] != "tool":
        reasons.append("final_role_not_tool")

    # 条件 6：最终 assistant action 必须调用无参数 submit。
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

def run_pipeline(input_path: str, output_path: str):
    stats = Counter()
    with open(input_path, "r", encoding="utf-8") as fin, \
        open(output_path, "w", encoding="utf-8") as fout:

        for line_number, line in enumerate(fin, start=1):
            stats["total"] += 1

            if not line.strip():
                stats["empty_line"] += 1
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                continue

            trajectory_steps = raw["trajectory_steps"]
            if not isinstance(trajectory_steps, list):
                stats["trajectory_steps_not_list"] += 1
                continue

            if not trajectory_steps:
                stats["empty_trajectories"] += 1
                continue

            try:
                converted = convert_record(raw)
            except (KeyError, TypeError) as exc:
                stats[f"convert_error:{type(exc).__name__}"] += 1
                continue

            stats["converted"] += 1

            filtered, reasons = filter_sample(converted)

            if filtered is None:
                stats["rejected"] += 1

                for reason in reasons:
                    stats[f"reason: {reason}"] += 1

                continue

            fout.write(json.dumps(filtered, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    # 7. 输出本次处理报告
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"原始样本：{stats['total']}")
    print(f"成功转换：{stats['converted']}")
    print(f"筛选保留：{stats['kept']}")
    print(f"筛选淘汰：{stats['rejected']}")

    print("\n淘汰原因：")
    for name, count in sorted(stats.items()):
        if name.startswith("reason:"):
            print(f"  {name.removeprefix('reason:')}: {count}")


def main():
    # with open("data_examples/swe-master-4b-50tasks-128k-150step.jsonl", "r") as f:
    #     for line in f:
    #         raw = json.loads(line)
    #         break

    # converted = convert_record(raw)

    # print(converted.keys)
    # print([message["role"] for message in converted["input"]])

    input_path = "data_examples/r2e-gym-inference-traj/glm46_0_used_swe_rebench_1_demo.jsonl"
    output_path = "data_examples/sft_data/glm46_0_used_swe_rebench_1_filtered.jsonl"
    run_pipeline(input_path, output_path)

if __name__ == "__main__":
    main()
