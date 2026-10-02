#!/usr/bin/env python3
"""Pre-pull and validate Docker images for a SWE-bench task slice."""

from __future__ import annotations

import argparse

from datasets import load_dataset, load_from_disk

from r2egym.agenthub.run.edit import prepull_docker_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-pull R2E-Gym Docker images without starting an Agent or "
            "contacting the model server."
        )
    )
    parser.add_argument(
        "--dataset",
        default="R2E-Gym/SWE-Bench-Verified",
        help="Hugging Face dataset name or a dataset saved on disk.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=1,
        help="First dataset index. The default skips the previous one-task run.",
    )
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected tasks without pulling images.",
    )
    return parser.parse_args()


def load_task_dataset(dataset: str, split: str):
    try:
        return load_dataset(dataset, split=split)
    except Exception as load_error:
        try:
            return load_from_disk(dataset)
        except Exception:
            raise load_error


def main() -> None:
    args = parse_args()
    if args.start_idx < 0:
        raise ValueError("--start-idx must be non-negative")
    if args.task_count < 1:
        raise ValueError("--task-count must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    dataset = load_task_dataset(args.dataset, args.split)
    stop_idx = args.start_idx + args.task_count
    if stop_idx > len(dataset):
        raise ValueError(
            f"Requested indices [{args.start_idx}, {stop_idx}), but the "
            f"dataset contains only {len(dataset)} tasks."
        )

    selected = [dataset[index] for index in range(args.start_idx, stop_idx)]
    unique_images = {task["docker_image"] for task in selected}

    print(
        f"Selected {len(selected)} tasks ({len(unique_images)} unique images) "
        f"from indices {args.start_idx} through {stop_idx - 1}."
    )
    print(f"First task: {selected[0]['instance_id']}")
    print(f"Last task:  {selected[-1]['instance_id']}")

    if args.dry_run:
        for index, task in enumerate(selected, start=args.start_idx):
            print(f"{index:03d}  {task['instance_id']}  {task['docker_image']}")
        return

    prepull_docker_images(selected, max_workers=args.workers)
    print("All selected images are present locally and passed validation.")


if __name__ == "__main__":
    main()
