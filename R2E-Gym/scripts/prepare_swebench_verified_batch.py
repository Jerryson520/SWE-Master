#!/usr/bin/env python3
"""Prepare several SWE-bench Verified image pairs by reusing the audited pilot builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from r2egym.imageprep.manifest import write_manifest_atomic


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pilot-script", required=True)
    parser.add_argument("--reuse-manifest", action="append", default=[])
    parser.add_argument("--instance-id", action="append", default=[])
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [row for row in rows if str(row["instance_id"]) in wanted]
        found = {str(row["instance_id"]) for row in rows}
        missing = wanted - found
        if missing:
            raise SystemExit(f"instance ids not found: {sorted(missing)}")
    if not rows:
        raise SystemExit("batch input is empty")

    reusable: dict[str, dict] = {}
    for manifest_path in args.reuse_manifest:
        path = Path(manifest_path)
        if path.exists():
            reusable.update(json.loads(path.read_text()).get("images", {}))

    records: dict[str, dict] = {}

    def save() -> None:
        manifest = {
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
        }
        write_manifest_atomic(args.manifest, manifest)
    with tempfile.TemporaryDirectory(prefix="swebench-verified-batch-") as directory:
        root = Path(directory)
        for index, row in enumerate(rows, 1):
            source = str(row["docker_image"])
            instance_id = str(row["instance_id"])
            cached = reusable.get(source)
            if cached and cached.get("instance_ids") == [instance_id]:
                print(f"[{index}/{len(rows)}] reuse {instance_id}", flush=True)
                records[source] = cached
                save()
                continue

            one_input = root / f"task-{index}.jsonl"
            one_manifest = root / f"manifest-{index}.json"
            one_input.write_text(json.dumps(row) + "\n")
            print(f"[{index}/{len(rows)}] prepare {instance_id}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    args.pilot_script,
                    "--input", str(one_input),
                    "--manifest", str(one_manifest),
                    "--repository", args.repository,
                ],
                check=True,
            )
            records.update(json.loads(one_manifest.read_text())["images"])
            save()

    save()
    print(json.dumps({"manifest": args.manifest, "task_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
