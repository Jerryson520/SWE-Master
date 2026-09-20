"""覆盖 main 初始化路径；禁止拉镜像、调用教师或删除镜像。"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_teacher_rollouts.py"
spec = importlib.util.spec_from_file_location("teacher_collector_startup", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class TeacherStartupTests(unittest.TestCase):
    def check_startup(self, fail=False, real_docker=False):
        import docker
        with tempfile.TemporaryDirectory() as directory:
            if real_docker:
                client = docker.from_env(timeout=10)
                original_close = client.close
                client.close = Mock(side_effect=original_close)
            else:
                # 必须不是 MagicMock，否则其 __enter__ 会掩盖真实 SDK 不支持 with 的问题。
                client = SimpleNamespace(
                    info=Mock(return_value={"DockerRootDir": directory, "ID": "test"}),
                    images=SimpleNamespace(list=Mock(return_value=[])), close=Mock())
            self.assertFalse(hasattr(client, "__enter__"))
            fake_edit = SimpleNamespace(runagent=Mock(), configure_loopback_llm_no_proxy=Mock())
            collect = Mock(side_effect=RuntimeError("mock stop") if fail else None)
            argv = ["--source-dir", directory, "--output-dir", directory + "/out",
                    "--llm-name", "openai/test", "--context-window", "131072",
                    "--max-output-tokens", "8192", "--context-safety-margin", "4096",
                    "--max-trajectory-output-tokens", "131072"]
            with patch.dict(os.environ, {"OPENAI_API_BASE": "http://127.0.0.1:18008/v1",
                                         "OPENAI_API_KEY": "EMPTY", "DOCKER_HOST": ""}), \
                 patch.dict(sys.modules, {"r2egym.agenthub.run.edit": fake_edit}), \
                 patch.object(collector, "load_tasks", return_value=([
                     {"instance_id": "test", "docker_image": "test:image",
                      "repo_name": "test-repo"}], {"x": "hash"})), \
                 patch.object(collector, "collect", collect), \
                 patch.object(docker, "from_env", return_value=client), \
                 patch.object(collector.os, "chdir"):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "mock stop"):
                        collector.main(argv)
                else:
                    collector.main(argv)
            collect.assert_called_once()
            fake_edit.runagent.assert_not_called()
            client.close.assert_called_once()

    def test_client_without_context_manager(self):
        self.check_startup()

    def test_client_closed_on_failure(self):
        self.check_startup(fail=True)

    @unittest.skipUnless(os.environ.get("TEACHER_TEST_REAL_DOCKER") == "1", "opt-in read-only Docker check")
    def test_real_sdk_startup_no_rollout(self):
        self.check_startup(real_docker=True)


if __name__ == "__main__":
    unittest.main()
