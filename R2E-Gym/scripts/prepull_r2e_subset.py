#!/usr/bin/env python3
"""预拉取 2E-Gym-Subset 指定任务范围的 Docker 镜像。

这个脚本只做以下事情：

1. 从原始 Parquet 目录或准备后的 JSON 文件加载任务；
2. 根据 start_idx 和 task_count 选择任务；
3. 对 docker_image 去重；
4. 调用 edit.py 中已有的 prepull_docker_images()；
5. 不启动 Agent，不连接模型服务。

支持两种输入：

原始 Hugging Face 数据目录：
/home/ecs-user/swe_datasets/R2E-Gym-Subset

准备后的任务 JSON：
/home/ecs-user/swe_datasets/r2e_teacher_v1/train_tasks.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from r2egym.agenthub.run.edit import prepull_docker_images

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-pull Docker images for R2E-Gym-Subset tasks."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="R2E-Gym-Subset 根目录、包含 Parquet 的 data 目录，或者准备好的 JSON 任务文件。"
    )

    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="起始任务下标，默认为0"
    )

    parser.add_argument(
        "--task-count",
        type=int,
        default=3,
        help="选择多少条任务，默认为 3。"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行拉取镜像的线程数，默认为 1。",
    )

    parser.add_argument(
        "--docker-ip",
        default="",
        help="远程 Docker daemon 的 IP。留空表示使用当前机器的本地 Docker socket。"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印任务和镜像，不执行 docker pull。",
    )

    return parser.parse_args()



def load_parquet_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_dir = Path(args.dataset) / "data"
    parquet_files = sorted(data_dir.glob("train-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"没有找到数据文件：{data_dir}/train-*.parquet")

    dataset = load_dataset("parquet", data_files={"train": [str(path) for path in parquet_files]}, split="train")

    stop_idx = args.start_idx + args.task_count
    
    if stop_idx > len(dataset):
        raise ValueError(f"任务范围 [{args.start_idx}, {stop_idx}) 超过数据集长度 {len(dataset)}")

    if args.start_idx < 0:
        raise ValueError("--start-idx 不能小于 0")

    if args.task_count < 1:
        raise ValueError("--task-count 必须至少为 1")
    
    selected = [dataset[idx] for idx in range(args.start_idx, stop_idx)]

    return selected

def main():
    args = parse_args()

    selected = load_parquet_tasks(args)

    print(f"本次任务数：{len(selected)}")

    stop_idx = args.start_idx + args.task_count
    print(f"选中任务：[{args.start_idx}, {stop_idx})")

    for idx, task in enumerate(selected, start=args.start_idx):
        print(idx, task["repo_name"], task["commit_hash"], task["docker_image"])

    if args.dry_run:
        print("dry-run：没有拉取镜像")
        return
    
    prepull_docker_images(
        ds_selected=selected,
        max_workers=args.workers,
        ip=args.docker_ip,
    )
        
if __name__ == "__main__":
    main()

        
