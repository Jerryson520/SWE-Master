"""Generic repository tool framework."""

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolExecutionError, ToolInfrastructureError, ToolOutput
from .registry import ToolRegistry, ToolRuntimeConfig, create_default_registry
from .grep import GrepTool

DEFAULT_TOOL_REGISTRY = create_default_registry()
TOOL_DEFINITIONS = DEFAULT_TOOL_REGISTRY.get_definitions()

__all__ = [
    "BaseTool",
    "DEFAULT_TOOL_REGISTRY",
    "GrepTool",
    "ParameterSpec",
    "TOOL_DEFINITIONS",
    "ToolCallError",
    "ToolContext",
    "ToolExecutionError",
    "ToolInfrastructureError",
    "ToolOutput",
    "ToolRegistry",
    "ToolRuntimeConfig",
    "create_default_registry",
]
