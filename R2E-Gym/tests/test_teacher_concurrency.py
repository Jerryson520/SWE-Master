"""仅 mock rollout；覆盖并发、断点和失败保护，不访问模型或 Docker。"""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_teacher_rollouts as c


def fake_run(ds, exp_name, **kwargs):
    time.sleep(0.03)
    return json.dumps({"ds": ds, "exp_name": exp_name, "reward": 1,
                       "trajectory_steps": [{"pid": os.getpid()}], "test_output": "passed"})


class Images:
    def __init__(self, limit=None):
        self.cleaned = []
        self.limit = limit

    def prepare(self, tasks):
        return tasks[:self.limit] if self.limit else tasks

    def cleanup(self, tasks):
        self.cleaned.append([t["instance_id"] for t in tasks])


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.tasks = [{"instance_id": f"task-{i}", "docker_image": f"image:{i}",
                       "repo_name": "test-repo"}
                      for i in range(31)]
        self.config = {"batch_size": 30, "rollouts": 3, "agent": {}}

    def collect(self, tasks, images, run=fake_run, workers=2, real_process=False):
        kwargs = {} if real_process else {
            "executor_factory": lambda: ThreadPoolExecutor(max_workers=workers)}
        with contextlib.redirect_stdout(io.StringIO()):
            c.collect(tasks, self.out, self.config, images, run, "test",
                      max_workers=workers, **kwargs)

    def test_two_batches_bounded_parallel_and_resume(self):
        lock = threading.Lock()
        active = peak = calls = 0
        def run(**kwargs):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
            try:
                return fake_run(**kwargs)
            finally:
                with lock:
                    active -= 1
        images = Images()
        self.collect(self.tasks, images, run)
        self.assertEqual((calls, peak), (93, 2))
        self.assertEqual([len(b) for b in images.cleaned], [1] * 31)
        self.assertEqual(len(list((self.out / "batches").glob("*.jsonl"))), 2)
        rows = [json.loads(s) for s in (self.out / "batches/batch-0000.jsonl").read_text().splitlines()]
        self.assertEqual([r["ds"]["instance_id"] for r in rows],
                         [t["instance_id"] for t in self.tasks[:30] for _ in range(3)])
        self.collect(self.tasks, images, run, workers=4)
        self.assertEqual(calls, 93)  # 改并发不重复生成已有结果

    def test_error_drains_inflight_no_new_slots_no_cleanup(self):
        started = threading.Barrier(2)
        calls = []
        def run(**kwargs):
            exp = kwargs["exp_name"]
            calls.append(exp)
            started.wait(timeout=5)
            if exp.endswith("rollout-0"):
                raise RuntimeError("mock failure")
            time.sleep(0.1)
            return fake_run(**kwargs)
        images = Images()
        with self.assertRaisesRegex(RuntimeError, "mock failure"):
            self.collect(self.tasks[:2], images, run)
        self.assertEqual(len(calls), 2)
        self.assertEqual(images.cleaned, [])
        self.assertFalse((self.out / "batches/batch-0000.jsonl").exists())
        self.assertTrue(c.result_path(self.out, self.tasks[0], 1).exists())
        self.assertTrue((self.out / "errors/task-0/rollout-0.json").exists())
        self.collect(self.tasks[:2], images)
        self.assertEqual(len(list(self.out.glob("task-*/rollout-*.json"))), 6)

    def test_space_reduces_batch_and_serial_option(self):
        images = Images(limit=1)
        self.collect(self.tasks[:3], images, workers=1)
        self.assertEqual([len(b) for b in images.cleaned], [1, 1, 1])

    def test_none_return_is_error(self):
        images = Images()
        with self.assertRaisesRegex(RuntimeError, "None"):
            self.collect(self.tasks[:1], images, lambda **kwargs: None, workers=1)
        self.assertFalse(images.cleaned)
        self.assertFalse(list(self.out.glob("task-*/rollout-*.json")))

    def test_cleanup_before_next_task_and_after_all_rollouts(self):
        images = Images()
        def run(**kwargs):
            task = kwargs["ds"]["instance_id"]
            if task == "task-1":
                self.assertEqual(images.cleaned, [["task-0"]])
                self.assertEqual(len(list((self.out / "task-0").glob("rollout-*.json"))), 3)
            else:
                self.assertFalse(images.cleaned)
            return fake_run(**kwargs)
        self.collect(self.tasks[:2], images, run, workers=1)
        self.assertEqual(images.cleaned, [["task-0"], ["task-1"]])

    def test_shared_image_waits_for_all_tasks(self):
        tasks = [dict(t, docker_image="shared:tag") for t in self.tasks[:2]]
        images = Images()
        def run(**kwargs):
            self.assertFalse(images.cleaned)
            return fake_run(**kwargs)
        self.collect(tasks, images, run)
        self.assertEqual(images.cleaned, [["task-0", "task-1"]])

    def test_resume_cleanup_precedes_space_check(self):
        c.atomic_write(self.out / "batches/batch-0000.json", [t["instance_id"] for t in self.tasks[:2]])
        for i in range(3):
            c.save_attempt(self.tasks[0], i, self.out, self.config, fake_run, "test")
        images = Images()
        def prepare(tasks):
            self.assertIn(["task-0"], images.cleaned)
            return tasks
        images.prepare = prepare
        self.collect(self.tasks[:2], images)
        self.assertEqual(images.cleaned, [["task-0"], ["task-1"]])

    def test_low_space_waits_for_inflight_cleanup(self):
        self.config["rollouts"] = 1
        c.atomic_write(self.out / "batches/batch-0000.json", [t["instance_id"] for t in self.tasks[:2]])
        images = Images()
        blocked = []
        def prepare(tasks):
            if tasks[0]["instance_id"] == "task-1" and not images.cleaned:
                blocked.append(True)
                raise c.InsufficientSpace("mock low disk")
            return tasks
        images.prepare = prepare
        self.collect(self.tasks[:2], images)
        self.assertTrue(blocked)
        self.assertEqual(images.cleaned, [["task-0"], ["task-1"]])

    def test_real_spawn_workers(self):
        images = Images()
        self.collect(self.tasks[:1], images, real_process=True)
        rows = [json.loads(p.read_text()) for p in self.out.glob("task-*/rollout-*.json")]
        self.assertEqual(len(rows), 3)
        pids = {r["trajectory_steps"][0]["pid"] for r in rows}
        self.assertEqual(len(pids), 2)
        self.assertNotIn(os.getpid(), pids)

    def test_existing_manifest_allows_changing_workers(self):
        client = SimpleNamespace(
            info=lambda: {"DockerRootDir": str(self.out), "ID": "test"},
            images=SimpleNamespace(list=lambda **kwargs: []), close=lambda: None)
        fake_edit = SimpleNamespace(runagent=Mock(), configure_loopback_llm_no_proxy=Mock())
        argv = ["--source-dir", str(self.out), "--output-dir", str(self.out / "run"),
                "--llm-name", "openai/test", "--context-window", "131072",
                "--max-output-tokens", "8192", "--context-safety-margin", "4096",
                "--max-trajectory-output-tokens", "131072"]
        with patch.dict(os.environ, {"OPENAI_API_BASE": "http://127.0.0.1:18008/v1",
                                     "OPENAI_API_KEY": "EMPTY", "DOCKER_HOST": ""}), \
             patch.dict(sys.modules, {"r2egym.agenthub.run.edit": fake_edit,
                                      "docker": SimpleNamespace(from_env=lambda **kwargs: client)}), \
             patch.object(c, "load_tasks", return_value=(self.tasks[:1], {"file": "hash"})), \
             patch.object(c, "collect") as collect, patch.object(c.os, "chdir"):
            c.main(argv + ["--max-workers", "1"])
            original = (self.out / "run/manifest.json").read_bytes()
            self.assertNotIn("max_workers", json.loads(original)["config"])
            c.main(argv + ["--max-workers", "2"])
            self.assertEqual((self.out / "run/manifest.json").read_bytes(), original)
            self.assertEqual(collect.call_args.kwargs["max_workers"], 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            c.parse_args(argv + ["--max-workers", "0"])


if __name__ == "__main__":
    unittest.main()
