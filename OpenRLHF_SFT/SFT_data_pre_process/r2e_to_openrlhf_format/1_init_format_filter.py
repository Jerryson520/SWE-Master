import json
from collections import defaultdict
hint_content = "You forgot to use a function call in your response. "
in_file = "./R2E-Gym/results/swe_all_w_sp_obs.jsonl"
out_file = "./R2E-Gym/results/swe_all_w_sp_obs_filter.jsonl"
failed_num = defaultdict(int)
success_num = 0

# 本文件执行“轨迹级质量过滤”，回答的是：这一条 rollout 是否是一条干净、
# 成功、适合模仿的 SFT 示范。任务级难度过滤（例如保留
# 0 < reward_mean < 1 的任务）由 ../bon_filter/ 中的脚本另行完成。

with open(in_file, "r", encoding="utf-8") as fin, \
     open(out_file, "w", encoding="utf-8") as fout:

    for line in fin:
        skip_flag=False
        sample = json.loads(line)
        # 条件 1：只保留成功轨迹作为正向 SFT 示范。失败 rollout 仍可交给
        # bon_filter 估计任务难度，但不会进入这个 SFT 成功轨迹池。
        if sample["reward"]!=1: # 1. 只保留reward==1的轨迹
            continue
        inputs = sample["input"]
        
        # 针对一种已知的错误终止 action 做启发式修复：如果模型在接近结尾处
        # 输出了空函数名，就将它解释为 submit，并删除其后的两条消息。这是
        # 针对现有脏数据的清洗规则，不是通用的 function-call 解析规则。
        # 注意：这里在后面的短轨迹检查之前就访问了 inputs[-4]。
        if "<function=></function>" in inputs[-4]["content"].replace("\n","").replace(" ",""):
            # print(inputs[-4]["content"][-100:])
            inputs[-4]["content"] = inputs[-4]["content"].replace("<function=>","<function=submit>")
            inputs = inputs[:-2]

        all_str = ""
        for mess in inputs[:-1]:
            all_str += mess["content"]
        step_count = sample["step_count"]
        token_usage_total =sample["token_usage_total"]
        
        # print(inputs[-2]["content"][-100:])
        # print("=="*50)
        # 条件 2：排除依赖环境提示才纠正无效/缺失 function call 的轨迹。
        # 即使最终 reward == 1，这种轨迹也会让模型学到“先输出坏 action，
        # 再依靠环境纠正”的错误行为模式。
        if (hint_content in all_str) : 
            # print(inputs[-2]["content"][-100:])
            # print("=="*50)
            # if hint_content not in inputs[-4]["content"]
            failed_num["have_hint_content"]+=1
            skip_flag=True
        # print(all_str)
        # exit()

        # 条件 3：排除恰好在历史最大步数处结束的轨迹。此类轨迹被视为因预算
        # 截断，而不是自然完成的干净示范。如果 rollout 使用其他最大步数，
        # 这里的硬编码数值也必须同步更新。
        if step_count == 100 or step_count == 120 or step_count == 150:
            failed_num["max_step_limit"]+=1
            skip_flag=True
        # if token_usage_total >=98304:
        # if token_usage_total >=65536:
        # if token_usage_total >=40960:
        # if token_usage_total >=32768:
        # 条件 4：排除累计 rollout token 使用量达到历史 80K 阈值的轨迹。
        # 这里统计的是 rollout 累计用量，不是 chat template 展开后的最终训练
        # 序列长度；后者仍需使用训练 tokenizer 重新统计。
        if token_usage_total >=81920 :
            failed_num["max_token_limit"]+=1
            skip_flag=True
        # 条件 5：排除结构不完整的 conversation。过滤器至少要求存在一个终止
        # assistant action 及其对应的 tool observation，因为后续会访问
        # inputs[-2] 和 inputs[-1]。
        if len(inputs) < 2:
            failed_num["so_less_turn"]+=1
            skip_flag=True

        # 条件 6：倒数第二条消息必须显式调用 submit。仅仅停止、异常退出，
        # 或以其他工具结束的轨迹，不属于协议完整的干净 SFT 示范。
        if "<function=submit>" not in inputs[-2]["content"]:
            failed_num["final_rool_not_submit"]+=1
            skip_flag=True
            # print(inputs[-2]["content"])
            # print(len(inputs))
            if "<function=>" in inputs[-2]["content"]:
                pass
                # print(inputs[-1]["content"])
                # print(len(inputs))
                # print("=="*50)
            else:
                pass
                # failed_num["final_rool_not_submit"]+=1
                # skip_flag=True

        final_mess = inputs[-2]
        # 条件 7：submit 是无参数工具。这里只接受下面两种空调用写法；如果
        # submit 携带参数，就将它判定为格式错误的终止 action。
        if ("<function=submit>\n</function>" not in final_mess["content"]) and  ("<function=submit></function>" not in final_mess["content"]) :
            failed_num["submit_with_paras"]+=1
            skip_flag=True
            
        
        # 条件 8：裁剪前的最后一条消息必须是与 submit 配对的 tool observation，
        # 以确认最后的 assistant action 确实被环境执行过。
        if inputs[-1]["role"]!= "tool":
            failed_num["final_role_not_tool"]+=1
            skip_flag=True

        if skip_flag:
            continue

        # if "Finish" not in inputs[-1]["content"]:
        #     print(inputs)
        #     kill
        # 最后一条 tool observation 通常只是 "Finish" 一类执行确认。将其删除，
        # 使训练 conversation 以 assistant 的正确 submit action 结束，并且不再
        # 暗示模型还需要继续回复。
        sample["input"] = inputs[:-1]
        # print(inputs[-1])
        success_num+=1
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("Finished →", out_file)
print(f"Count of failed/excluded entries: {failed_num}")
print(f"Count of successful/kept entries: {success_num}")
