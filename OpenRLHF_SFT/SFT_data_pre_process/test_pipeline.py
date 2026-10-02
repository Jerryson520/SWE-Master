from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pipeline


def raw_record(instance_id: str, *, reward: int = 1, step_count: int = 20,
               exit_reason: str = "agent", action: str | None = None) -> dict:
    return {
        "problem_statement": "Fix the bug.",
        "reward": reward,
        "exit_reason": exit_reason,
        "exp_name": f"test/{instance_id}",
        "docker_image": "example/image:latest",
        "agent_args": {
            "instance_prompt": "Work in {working_dir}: {problem_statement}",
            "system_prompt": "You are a coding agent.",
        },
        "ds": {"instance_id": instance_id, "repo_name": "example"},
        "trajectory_steps": [{
            "thought": "The fix is complete.",
            "action": action or "<function=submit>\n</function>",
            "observation": "Finished.",
            "step_count": step_count,
            "token_usage_total": 90000,
        }],
        "output_patch": "diff --git a/a.py b/a.py",
    }


class PipelineTest(unittest.TestCase):
    def test_rsft_filters_and_audits_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batches = root / "batches"
            batches.mkdir()
            records = [
                raw_record("keep-at-100", step_count=100),
                raw_record("too-long", step_count=101),
                raw_record("budget-stop", exit_reason="max_step_limit"),
                raw_record("failed", reward=0),
                raw_record("invalid-action", action="plain text"),
                raw_record("keep-at-100", step_count=100),
            ]
            with (batches / "batch-0000.jsonl").open("w", encoding="utf-8") as fout:
                for record in records:
                    fout.write(json.dumps(record) + "\n")

            output = root / "output"
            report = pipeline.run_pipeline(batches, output)

            kept = [json.loads(line) for line in
                    (output / "clean_success_all.jsonl").read_text().splitlines()]
            rejected = [json.loads(line) for line in
                        (output / "rejected.jsonl").read_text().splitlines()]
            self.assertEqual([item["instance_id"] for item in kept], ["keep-at-100"])
            self.assertEqual(report["counts"]["kept"], 1)
            self.assertEqual(report["counts"]["rejected"], 5)
            self.assertEqual(report["counts"]["teacher_context_at_or_over_80k"], 1)
            reasons = {reason for item in rejected for reason in item["reasons"]}
            self.assertIn("too_many_steps", reasons)
            self.assertIn("non_natural_exit", reasons)
            self.assertIn("reward_not_one", reasons)
            self.assertIn("missing_tool_call", reasons)
            self.assertIn("duplicate_trajectory", reasons)


if __name__ == "__main__":
    unittest.main()
