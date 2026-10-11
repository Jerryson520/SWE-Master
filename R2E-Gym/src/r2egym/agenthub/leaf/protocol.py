"""Pinned Leaf protocol from microsoft/FrogNano.

Upstream commit: 5038f91560c635acb50daa452eb62c12daa737ad (2026-10-11).
The executor remains sequential even when one assistant turn contains several calls.
"""

from dataclasses import dataclass, field
from typing import Any


SYSTEM_PROMPT = (
    "You are a software-engineering agent working in a checked-out repository.\n"
    "Use the available tools to inspect files, make focused edits, and run commands.\n"
    "Solve the task, validate when practical, then give a brief final answer."
)


@dataclass(frozen=True)
class LeafToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LeafToolResult:
    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True)
class LeafAction:
    tool_calls: tuple[LeafToolCall, ...] = ()
    content: str = ""
    reasoning_content: str = ""

    @property
    def is_terminal(self) -> bool:
        return not self.tool_calls


LEAF_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a text file. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read."},
                    "offset": {"type": "integer", "description": "Optional 1-based starting line."},
                    "limit": {"type": "integer", "description": "Optional maximum number of lines."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a complete file, creating parent directories when needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to write."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace an exact string in a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to edit."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py."},
                    "path": {"type": "string", "description": "Optional directory to search from."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command in the repository workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run."},
                    "description": {"type": "string", "description": "Short description of the command."},
                    "timeout": {"type": "number", "description": "Optional timeout in seconds."},
                },
                "required": ["command"],
            },
        },
    },
]
