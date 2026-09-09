import json
import random
from collections import defaultdict

def extract_by_docker_image_upsample(
    all_jsonl, candicate_jsonl, out_jsonl,
    limit_min_num=-1, limit_max_turn=-1
):
    """将干净的 SFT 成功轨迹与通过难度筛选的任务集合取交集。

    `candicate_jsonl` 保存前两个 BON 脚本选出的任务级难度结果；
    `all_jsonl` 应保存经过轨迹级过滤的干净成功 SFT conversation。因此，
    失败 rollout 只参与任务难度判断，不会被复制进最终 SFT 数据集。
    """

    # 筛选条件 1：载入经验通过率落在目标难度区间内的任务 problem_statement。
    with open(candicate_jsonl, "r", encoding="utf-8") as f:
        wanted = {json.loads(line)["problem_statement"].strip() for line in f}
    print(len(wanted ))
    # Cluster by problem_statement (ps)
    clusters = defaultdict(list)
    count_all = 0
    with open(all_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            count_all+=1
            data = json.loads(line)
            ps = data["input"][2]["content"].split("</issue_description>")[0].split("<issue_description>")[-1].strip()
            # 筛选条件 2：只有轨迹所属任务位于难度候选集合中，才保留这条干净
            # SFT 轨迹。按完整问题文本精确匹配比较脆弱；如果两个文件都保留
            # instance_id，使用 instance_id 匹配会更稳健。
            if ps in wanted:
                clusters[ps].append(data)

    print(f"Data entries in the original file: {count_all}")

    total_upsampled = 0
    total_downsampled = 0
    upsample_stats = defaultdict(int)    # added count : number of ps
    downsample_stats = defaultdict(int)  # deleted count : number of ps
    all_write_num = 0
    # Write refined data
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for ps, items in clusters.items():
            count = len(items)

            # 筛选条件 3（可选下采样）：限制每个任务最多保留多少条轨迹。
            # 当前按 len(input) 排序，它表示消息数量而不是 tokenizer token
            # 长度，并优先保留消息轮次更少的 conversation。
            # =============== Limit maximum number of turns (priority processing) ==================
            if limit_max_turn > -1 and count > limit_max_turn:
                # Sort by turns in ascending order
                items.sort(key=lambda x: len(x["input"]))
                deleted = count - limit_max_turn
                items = items[:limit_max_turn]
                clusters[ps] = items  # Update
                total_downsampled += deleted
                downsample_stats[deleted] += 1

            # 筛选条件 4（可选上采样）：如果某个候选任务的干净轨迹少于
            # limit_min_num，就从已有轨迹中有放回抽样，直到达到下限。这会
            # 改变任务权重，但不会产生新的轨迹信息。
            # ================== Upsample insufficient parts ==================
            count = len(items)
            if limit_min_num > -1 and count < limit_min_num:
                need = limit_min_num - count
                sampled = random.choices(items, k=need)
                items.extend(sampled)
                total_upsampled += need
                upsample_stats[need] += 1

            # Write to file
            for d in items:
                all_write_num+=1
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("======== Processing Statistics ========")
    print("Number of filtered PS:", len(clusters))
    print("Upsampling statistics:", dict(upsample_stats))
    print("Total upsampled count:", total_upsampled)
    print("Downsampling statistics:", dict(downsample_stats))
    print("Total deleted trajectory count:", total_downsampled)
    print("======== Final Results ========")
    print("Final trajectory count remaining:", all_write_num)

    return clusters, upsample_stats, downsample_stats, total_upsampled, total_downsampled


all_jsonl = "xx.jsonl"

candicate_jsonl = "./R2E-Gym/results/0_bon_filter_resultes/0121_swe_bon/bon_stats_1e-07_0.9999999.jsonl"

out_jsonl =  "yy.jsonl"


extract_by_docker_image_upsample(
    all_jsonl,
    candicate_jsonl,
    out_jsonl,
    # 负数会关闭上述两种可选数量过滤，因此默认调用只执行候选任务取交集。
    limit_min_num=-2,  
    limit_max_turn=-2
)
