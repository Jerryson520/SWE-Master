"""Protocol objects shared by function-calling agents and environments."""

from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT = (
    "You are a software-engineering agent working in a checked-out repository.\n"
    "Use the available tools to inspect files, make focused edits, and run commands.\n"
    "Use Grep to search file contents and Glob to search for file names.\n"
    "Solve the task, validate when practical, then give a brief final answer."
)

PROMPT_VERSION = "leaf-prompt-v2"

R2E_OFFLINE_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + (
    "The current environment is an offline, sanitized repository workspace.\n\n"
    "Use only the tools and capabilities provided in this environment. The complete "
    "tool names, descriptions, and argument schemas are provided through the "
    "function-calling interface.\n\n"
    "You may inspect and modify repository files, run visible repository tests, and "
    "use sanitized local Git history such as git status, git diff, git blame, git log "
    "on HEAD ancestry, and git show on HEAD ancestors.\n\n"
    "Network access, remote Git references, reflog, stash, other branches, external "
    "solution sources, and private evaluation assets are unavailable.\n\n"
    "If a tool returns POLICY_BLOCKED, do not retry an equivalent operation or attempt "
    "to bypass the restriction. Continue using available repository files, visible "
    "tests, and allowed local history.\n\n"
    "Solve the task with focused edits, validate when practical, and finish with a "
    "brief summary."
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
