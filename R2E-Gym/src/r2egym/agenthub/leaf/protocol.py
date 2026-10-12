"""Protocol objects shared by function-calling agents and environments."""

from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT = (
    "You are a software-engineering agent working in a checked-out repository.\n"
    "Use the available tools to inspect files, make focused edits, and run commands.\n"
    "Use Grep to search file contents and Glob to search for file names.\n"
    "Solve the task, validate when practical, then give a brief final answer."
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True)
class AgentAction:
    tool_calls: tuple[ToolCall, ...] = ()
    content: str = ""
    reasoning_content: str = ""

    @property
    def is_terminal(self) -> bool:
        return not self.tool_calls
