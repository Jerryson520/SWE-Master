"""Generic repository tool framework."""

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolExecutionError, ToolInfrastructureError, ToolOutput
from .registry import ToolRegistry, ToolRuntimeConfig, create_default_registry

DEFAULT_TOOL_REGISTRY = create_default_registry()
TOOL_DEFINITIONS = DEFAULT_TOOL_REGISTRY.get_definitions()

__all__ = [
    "BaseTool",
    "DEFAULT_TOOL_REGISTRY",
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
