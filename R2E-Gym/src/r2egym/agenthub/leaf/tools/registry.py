"""Registry for generic repository tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import BaseTool, ToolCallError, ToolContext, ToolExecutionError, ToolInfrastructureError, ToolOutput
from .bash import BashTool
from .edit import EditTool
from .glob import GlobTool
from .read import ReadTool
from .write import WriteTool


@dataclass(frozen=True)
class ToolRuntimeConfig:
    tool_output_chars: int = 20000
    bash_output_chars: int = 30000
    glob_limit: int = 1000
    bash_timeout_sec: float = 120


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()):
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolCallError(f"Unknown tool: {name}") from exc

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.get_definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        try:
            return self.get(name).execute(arguments, context)
        except ToolInfrastructureError:
            raise
        except ToolExecutionError as exc:
            return ToolOutput(str(exc), success=False, metadata={"error_type": "ToolExecutionError"})
        except ToolCallError as exc:
            prefix = "" if str(exc).startswith("Unknown tool:") else "ValueError: "
            return ToolOutput(f"{prefix}{exc}", success=False, metadata={"error_type": "ToolCallError"})
        except Exception as exc:
            return ToolOutput(f"{type(exc).__name__}: {exc}", success=False, metadata={"error_type": type(exc).__name__})


def create_default_registry(config: ToolRuntimeConfig | None = None) -> ToolRegistry:
    config = config or ToolRuntimeConfig()
    return ToolRegistry(
        [
            ReadTool(output_chars=config.tool_output_chars),
            WriteTool(),
            EditTool(),
            GlobTool(limit=config.glob_limit),
            BashTool(output_chars=config.bash_output_chars, default_timeout=config.bash_timeout_sec),
        ]
    )
