#!/usr/bin/env python3
"""Prepare SWE-bench Verified image pairs concurrently with atomic checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import tempfile

from r2egym.imageprep.manifest import write_manifest_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pilot-script", required=True)
    parser.add_argument("--reuse-manifest", action="append", default=[])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    reusable: dict[str, dict] = {}
    for manifest_path in args.reuse_manifest:
        path = Path(manifest_path)
        if path.exists():
            reusable.update(json.loads(path.read_text()).get("images", {}))

    records = {
        str(row["docker_image"]): reusable[str(row["docker_image"])]
        for row in rows
        if str(row["docker_image"]) in reusable
        and reusable[str(row["docker_image"])].get("instance_ids") == [str(row["instance_id"])]
    }

    def save() -> None:
        write_manifest_atomic(args.manifest, {
            "schema_version": "swebench-verified-paired-image-v1",
            "dataset_type": "swebench-verified",
            "prepare_version": "leaf-offline-v1",
            "registry": args.repository,
            "selection": {
                "task_count": len(rows),
                "instance_ids": [str(row["instance_id"]) for row in rows],
                "prepared_count": len(records),
            },
            "images": records,
        })

    save()
    pending = [row for row in rows if str(row["docker_image"]) not in records]
    print(f"resume prepared={len(records)} pending={len(pending)} workers={args.workers}", flush=True)

    with tempfile.TemporaryDirectory(prefix="swebench-verified-parallel-") as directory:
        root = Path(directory)

        def prepare(index_row: tuple[int, dict]) -> tuple[str, dict]:
            index, row = index_row
            one_input = root / f"task-{index}.jsonl"
            one_manifest = root / f"manifest-{index}.json"
            one_input.write_text(json.dumps(row) + "\n")
            instance_id = str(row["instance_id"])
            print(f"[{index}/{len(rows)}] prepare {instance_id}", flush=True)
            subprocess.run([
                str(Path(".venv/bin/python")), args.pilot_script,
                "--input", str(one_input),
                "--manifest", str(one_manifest),
                "--repository", args.repository,
            ], check=True)
            data = json.loads(one_manifest.read_text())["images"]
            return instance_id, data

        indexed = [(i, row) for i, row in enumerate(rows, 1) if str(row["docker_image"]) not in records]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(prepare, item): item for item in indexed}
            for future in as_completed(futures):
                instance_id, data = future.result()
                records.update(data)
                save()
                print(f"DONE prepared={len(records)}/{len(rows)} instance={instance_id}", flush=True)

    print(json.dumps({"manifest": args.manifest, "task_count": len(rows), "prepared": len(records)}))


if __name__ == "__main__":
    main()
