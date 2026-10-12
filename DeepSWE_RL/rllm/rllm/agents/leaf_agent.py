"""Leaf agent using Qwen's native tool-call token format."""

import json
import re

from r2egym.agenthub.leaf import TOOL_DEFINITIONS, SYSTEM_PROMPT, AgentAction, ToolCall, ToolResult
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _tool_prompt() -> str:
    schemas = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in TOOL_DEFINITIONS)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{schemas}\n</tools>\n"
        "\nFor each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def parse_tool_response(response: str, step: int) -> AgentAction:
    calls = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(response)):
        try:
            value = json.loads(match.group(1))
            name = str(value.get("name") or "")
            arguments = value.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
        except (json.JSONDecodeError, TypeError, AttributeError):
            name, arguments = "", {}
        calls.append(ToolCall(id=f"call_{step}_{index}", name=name, arguments=arguments))
    content = _TOOL_CALL_RE.sub("", response).strip()
    return AgentAction(tool_calls=tuple(calls), content=content, reasoning_content=content)


class FunctionCallingAgent(BaseAgent):
    def __init__(self, parallel_tool_calls: bool = True, **_: object):
        self.parallel_tool_calls = parallel_tool_calls
        self._trajectory = Trajectory()
        self.reset()

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _tool_prompt()},
        ]
        self.step = 0
        self.cur_step = None

    def update_from_env(self, observation, reward, done, info, **kwargs):
        if not self._trajectory.steps:
            self.messages.append({"role": "user", "content": str(observation)})
            self.cur_step = Step(observation=str(observation))
            return

        results = observation if isinstance(observation, list) else []
        for result in results:
            if not isinstance(result, ToolResult):
                raise TypeError(f"Expected ToolResult, got {type(result).__name__}")
            self.messages.append(
                {"role": "tool", "tool_call_id": result.tool_call_id, "name": result.name, "content": result.content}
            )
        prior = self._trajectory.steps[-1]
        prior.next_observation = [result.content for result in results]
        prior.reward = reward
        prior.done = done
        prior.info.update(info)
        self.cur_step = Step(observation=[result.content for result in results])

    def update_from_model(self, response: str, **kwargs):
        if self.cur_step is None:
            raise RuntimeError("FunctionCallingAgent must receive an environment observation before generation")
        action = parse_tool_response(response, self.step)
        assistant = {"role": "assistant", "content": action.content or None}
        if action.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in action.tool_calls
            ]
        if action.reasoning_content:
            assistant["reasoning_content"] = action.reasoning_content
        self.messages.append(assistant)
        self.cur_step.thought = action.reasoning_content
        self.cur_step.action = action
        self.cur_step.model_response = response
        self._trajectory.steps.append(self.cur_step)
        self.step += 1
        return Action(action=action)

    def get_current_state(self):
        return self._trajectory.steps[-1] if self._trajectory.steps else self.cur_step

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def chat_completions(self):
        return self.messages
