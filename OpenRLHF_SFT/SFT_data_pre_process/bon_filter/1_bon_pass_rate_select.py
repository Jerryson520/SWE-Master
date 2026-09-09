import json

def filter_by_reward_range(input_jsonl, output_jsonl, min_acc, max_acc):
    """选择经验通过率位于 [min_acc, max_acc) 的任务。

    这是任务级过滤：每个输入行汇总同一任务的多次 rollout。常用区间近似为
    0 < x < 1，用来排除 teacher 始终失败或始终成功的任务。
    """
    with open(input_jsonl, "r", encoding="utf-8") as f, \
         open(output_jsonl, "w", encoding="utf-8") as out:
        num_a = 0
        for line in f:
            data = json.loads(line)

            # 条件 1：缺少任务级难度统计 reward_mean 的记录无法分类，直接忽略。
            if "reward_mean" not in data:
                continue

            acc = data["reward_mean"]

            # 条件 2：使用左闭右开的 [min_acc, max_acc) 区间。下面最终生效的
            # 参数用微小偏移近似 0 < x < 1：排除全失败任务（x == 0）和
            # 全成功任务（x == 1），保留同时出现成功和失败 rollout 的任务。
            if acc >= min_acc and acc < max_acc:
                # 条件 3：至少观测到一条 rollout。注意 count >= 1 只是合法性
                # 检查；要可靠估计任务难度，仍需对同一任务执行多次 rollout。
                if data["count"] >=1:
                    num_a+=1
                    out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Finished: {output_jsonl}")
    print(f"After Filtering, the num: {num_a}")

def filter_by_count(input_jsonl, output_jsonl, count):
    """可选：只保留恰好执行了 `count` 次 rollout 的任务。"""
    num=0
    with open(input_jsonl, "r", encoding="utf-8") as f, \
         open(output_jsonl, "w", encoding="utf-8") as out:

        for line in f:
            data = json.loads(line)

            count_rollout = data["count"]

            # 当不同任务的采样次数不一致时，该条件可以让任务之间更可比。
            # 此函数仅供可选使用，文件底部的默认流程并没有调用它。
            if count_rollout==count:
                num+=1
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Finished: {output_jsonl}")
    print(f"After Filtering, the num: {num}")


input_jsonl = "./R2E-Gym/results/0_bon_filter_resultes/0121_swe_bon/bon_stats.jsonl"

minacc = 0.0000001
maxacc = 0.7999999

minacc = 0.7999999
maxacc = 0.9999999

minacc = 0.9999999
maxacc = 1.0000001

minacc = 0.0000001
maxacc = 0.9999999

minacc = 0
maxacc = 0.0000001

minacc = 0.0000001
maxacc = 0.6000001

minacc = 0.0000001
maxacc = 0.4000001

minacc = 0.4000001
maxacc = 0.6000001

minacc = 0.3999999
maxacc = 0.9999999

# 在调用 filter_by_reward_range() 之前，只有最后一次 minacc/maxacc 赋值生效；
# 前面的多组赋值只是上游保留的实验痕迹。下面的实际区间近似为
# 0 < reward_mean < 1，即保留同时包含成功和失败 rollout 的任务。
minacc = 0.0000001
maxacc = 0.9999999

# minacc = 0.9999999
# maxacc = 1.0000001

# minacc = 0.600000
# maxacc = 0.999999

file_name = input_jsonl.split(".jsonl")[0]
output_jsonl = f"{file_name}_{minacc}_{maxacc}.jsonl"
filter_by_reward_range(
    input_jsonl=input_jsonl,
    output_jsonl=output_jsonl,
    min_acc= minacc,
    max_acc= maxacc
)
