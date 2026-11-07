#!/usr/bin/env python3
"""Create and audit a local-image copy of SWE-Bench Verified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import docker
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--smoke-output")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    run_id = (download_dir / "latest").read_text(encoding="utf-8").strip()
    status_path = download_dir / f"status_{run_id}.tsv"
    source_images = []
    for line in status_path.read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if len(columns) >= 2 and columns[0] == "ok":
            source_images.append(columns[1])

    dataset = load_dataset("R2E-Gym/SWE-Bench-Verified", split="test")
    client = docker.from_env()
    local_tags = {
        tag
        for image in client.images.list()
        for tag in image.tags
    }

    rows = []
    missing = []
    ambiguous = []
    absent_local = []
    for item in dataset:
        row = dict(item)
        # This field is metadata only and Hugging Face's JSON loader may infer it
        # as a datetime, which is not part of the Leaf task contract.
        row.pop("created_at", None)
        instance_id = str(row["instance_id"])
        owner, remainder = instance_id.split("__", 1)
        needle = f".{owner}_1776_{remainder}:latest"
        matches = [image for image in source_images if needle in image]
        if not matches:
            missing.append(instance_id)
            continue
        if len(matches) != 1:
            ambiguous.append({"instance_id": instance_id, "matches": matches})
            continue
        image = matches[0]
        if image not in local_tags:
            absent_local.append({"instance_id": instance_id, "image": image})
            continue
        row["source_docker_image"] = row["docker_image"]
        row["docker_image"] = image
        rows.append(row)

    instance_ids = [row["instance_id"] for row in rows]
    duplicate_instance_ids = sorted(
        {instance_id for instance_id in instance_ids if instance_ids.count(instance_id) > 1}
    )
    mapped_images = [row["docker_image"] for row in rows]
    duplicate_images = sorted(
        {image for image in mapped_images if mapped_images.count(image) > 1}
    )
    audit = {
        "schema_version": "swebench-verified-local-map-v1",
        "dataset": "R2E-Gym/SWE-Bench-Verified",
        "split": "test",
        "download_status": str(status_path),
        "dataset_rows": len(dataset),
        "status_images": len(source_images),
        "mapped_rows": len(rows),
        "missing": missing,
        "ambiguous": ambiguous,
        "absent_local": absent_local,
        "duplicate_instance_ids": duplicate_instance_ids,
        "duplicate_images": duplicate_images,
        "passed": (
            len(dataset) == 500
            and len(source_images) == 500
            and len(rows) == 500
            and not missing
            and not ambiguous
            and not absent_local
            and not duplicate_instance_ids
            and not duplicate_images
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.smoke_output:
        smoke_path = Path(args.smoke_output)
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(
            json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8"
        )
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        raise SystemExit("local image mapping audit failed")


if __name__ == "__main__":
    main()
