"""Function-calling environment adapter over the existing repository runtime."""

import inspect
import json

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.leaf.protocol import AgentAction, ToolResult
from r2egym.agenthub.leaf.tools import ToolContext, ToolRegistry, ToolRuntimeConfig, create_default_registry


class RepoToolEnvironment:
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
        tool_registry: ToolRegistry | None = None,
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
        self.tool_registry = tool_registry or create_default_registry(
            ToolRuntimeConfig(
                tool_output_chars=tool_output_chars,
                bash_output_chars=bash_output_chars,
                glob_limit=glob_limit,
                bash_timeout_sec=bash_timeout_sec,
            )
        )

    def reset(self):
        if self.env is None:
            self.env = RepoEnv(
                EnvArgs(ds=self.entry), backend=self.backend, step_timeout=self.step_timeout,
                reward_timeout=self.reward_timeout, verbose=self.verbose,
            )
        else:
            self.env.reset()
        self.has_container = bool(self.env.runtime.container)
        self.total_steps = 0
        self.tool_calls = 0
        return self.env.get_task_instruction(), {"leaf_tools": True}

    def _execute(self, call):
        context = ToolContext(
            runtime=self.env.runtime,
            workdir=self.env.runtime.repo_path,
            timeout=self.step_timeout,
        )
        output = self.tool_registry.execute(call.name, call.arguments, context)
        return ToolResult(tool_call_id=call.id, name=call.name, content=output.content), output

    def step(self, action):
        if not isinstance(action, AgentAction):
            raise TypeError(f"RepoToolEnvironment expects AgentAction, got {type(action).__name__}")
        if action.is_terminal:
            return [], 0.0, True, {"exit_reason": "agent", "tool_calls": 0}
        executed = [self._execute(call) for call in action.tool_calls]
        results = [result for result, _ in executed]
        outputs = [output for _, output in executed]
        self.total_steps += 1
        self.tool_calls += len(results)
        return results, 0.0, False, {
            "tool_calls": len(results),
            "total_tool_calls": self.tool_calls,
            "tool_success": [output.success for output in outputs],
            "tool_metadata": [output.metadata for output in outputs],
        }

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
        parameters = inspect.signature(RepoToolEnvironment.__init__).parameters
        kwargs = {key: value for key, value in extra_info.items() if key in parameters and key != "entry"}
        return RepoToolEnvironment(entry=extra_info, **kwargs)
