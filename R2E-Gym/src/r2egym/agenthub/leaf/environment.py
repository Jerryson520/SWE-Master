"""Leaf environment adapter over the existing R2E Docker runtime."""

import base64
import io
import inspect
import json
import shlex
import tarfile
import uuid

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.leaf import LeafAction, LeafToolResult
from r2egym.agenthub.leaf import tool_runner


_CONTAINER_RUNNER = "/tmp/r2egym_leaf_tool_runner.py"


class LeafRepoEnv:
    def __init__(
        self,
        entry=None,
        step_timeout=180,
        reward_timeout=300,
        backend="docker",
        delete_image=False,
        verbose=False,
        tool_output_chars=20000,
        bash_output_chars=30000,
        glob_limit=1000,
        bash_timeout_sec=120,
        **_: object,
    ):
        self.entry = entry or {}
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.backend = backend
        self.delete_image = delete_image
        self.verbose = verbose
        self.env = None
        self.has_container = False
        self.total_steps = 0
        self.tool_calls = 0
        self.leaf_env = {
            "LEAF_TOOL_OUTPUT_CHARS": str(tool_output_chars),
            "LEAF_BASH_OUTPUT_CHARS": str(bash_output_chars),
            "LEAF_GLOB_LIMIT": str(glob_limit),
            "LEAF_BASH_TIMEOUT_SEC": str(bash_timeout_sec),
        }

    def reset(self):
        if self.env is None:
            self.env = RepoEnv(
                EnvArgs(ds=self.entry), backend=self.backend, step_timeout=self.step_timeout,
                reward_timeout=self.reward_timeout, verbose=self.verbose,
            )
        else:
            self.env.reset()
        self.has_container = bool(self.env.runtime.container)
        if self.has_container:
            self.env.runtime.copy_to_container(inspect.getfile(tool_runner), _CONTAINER_RUNNER)
        self.total_steps = 0
        self.tool_calls = 0
        return self.env.get_task_instruction(), {"leaf_tools": True}

    def _execute(self, call):
        payload = base64.b64encode(
            json.dumps({"tool": call.name, "args": call.arguments, "workdir": self.env.runtime.repo_path}).encode()
        ).decode()
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.leaf_env.items())
        output_path = f"/tmp/r2egym_leaf_output_{uuid.uuid4().hex}.txt"
        command = (
            f"printf %s {shlex.quote(payload)} | base64 -d | {env_prefix} "
            f"python3 {shlex.quote(_CONTAINER_RUNNER)} > {shlex.quote(output_path)}"
        )
        timeout = self.step_timeout
        if call.name == "Bash":
            raw = call.arguments.get("timeout", self.leaf_env["LEAF_BASH_TIMEOUT_SEC"])
            timeout = max(timeout, int(float(raw) / 1000 if float(raw) > 1000 else float(raw)) + 60)
        command_output, exit_code = self.env.runtime.run(command, timeout=timeout, workdir="/")
        output = self._read_output(output_path)
        self.env.runtime.run(f"rm -f -- {shlex.quote(output_path)}", timeout=30, workdir="/")
        if str(exit_code) not in {"0", "None"}:
            output = f"Tool runner exited with code {exit_code}.\n{output or command_output}"
        return LeafToolResult(tool_call_id=call.id, name=call.name, content=str(output))

    def _read_output(self, path):
        runtime = self.env.runtime
        if runtime.backend != "docker":
            output, _ = runtime.run(f"cat -- {shlex.quote(path)}", timeout=30, workdir="/")
            return output
        chunks, _ = runtime.container.get_archive(path)
        archive = io.BytesIO(b"".join(chunks))
        with tarfile.open(fileobj=archive, mode="r:") as bundle:
            member = bundle.next()
            stream = bundle.extractfile(member)
            return stream.read().decode("utf-8", errors="replace") if stream else ""

    def step(self, action):
        if not isinstance(action, LeafAction):
            raise TypeError(f"LeafRepoEnv expects LeafAction, got {type(action).__name__}")
        if action.is_terminal:
            return [], 0.0, True, {"exit_reason": "agent", "tool_calls": 0}
        results = [self._execute(call) for call in action.tool_calls]
        self.total_steps += 1
        self.tool_calls += len(results)
        return results, 0.0, False, {"tool_calls": len(results), "total_tool_calls": self.tool_calls}

    def compute_final_reward(self):
        return self.env.compute_reward()

    def output_patch(self):
        return self.env.runtime.get_patch()

    def close(self):
        if self.env is not None:
            self.env.close()

    @staticmethod
    def from_dict(extra_info):
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        parameters = inspect.signature(LeafRepoEnv.__init__).parameters
        kwargs = {key: value for key, value in extra_info.items() if key in parameters and key != "entry"}
        return LeafRepoEnv(entry=extra_info, **kwargs)
