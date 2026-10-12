"""Generic tool contracts shared by repository tool environments."""

from __future__ import annotations

import base64
import json
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


MISSING = object()


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    json_type: str
    description: str
    required: bool = False
    default: Any = MISSING
    enum: tuple[Any, ...] | None = None
    normalize: Callable[[Any], Any] | None = field(default=None, repr=False, compare=False)

    def schema(self) -> dict[str, Any]:
        schema = {"type": self.json_type, "description": self.description}
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        return schema

    def parse(self, raw: dict[str, Any]) -> Any:
        if self.name not in raw:
            if self.required:
                raise ToolCallError(f"{self.name} is required")
            if self.default is MISSING:
                return MISSING
            return self.default
        value = raw[self.name]
        if self.normalize is not None:
            try:
                return self.normalize(value)
            except (TypeError, ValueError) as exc:
                raise ToolCallError(f"{self.name} has invalid type") from exc
        expected = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }.get(self.json_type)
        if expected is not None and not isinstance(value, expected):
            raise ToolCallError(f"{self.name} has invalid type")
        if self.enum is not None and value not in self.enum:
            choices = ", ".join(map(str, self.enum))
            raise ToolCallError(f"{self.name} must be one of: {choices}")
        return value


@dataclass
class ToolOutput:
    content: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    runtime: Any
    workdir: str
    timeout: float


class ToolCallError(Exception):
    """The model supplied an unusable tool call."""


class ToolInfrastructureError(RuntimeError):
    """The tool could not communicate with its execution environment."""


class ToolExecutionError(RuntimeError):
    """A well-formed operation failed inside the repository environment."""


class BaseTool(ABC):
    name: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()

    def get_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {parameter.name: parameter.schema() for parameter in self.parameters},
                    "required": [parameter.name for parameter in self.parameters if parameter.required],
                },
            },
        }

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ToolCallError("arguments must be an object")
        parsed: dict[str, Any] = {}
        for parameter in self.parameters:
            value = parameter.parse(raw)
            if value is not MISSING:
                parsed[parameter.name] = value
        return parsed

    @abstractmethod
    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        raise NotImplementedError


def run_python(
    context: ToolContext,
    source: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> str:
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    command = "python3 -c {} {}".format(shlex.quote(source), shlex.quote(encoded))
    output, exit_code = context.runtime.run(
        command,
        timeout=max(1, int(timeout if timeout is not None else context.timeout)),
        workdir=context.workdir,
    )
    if str(exit_code) == "-1":
        raise ToolInfrastructureError(output)
    if str(exit_code) != "0":
        lines = [line.strip() for line in str(output).splitlines() if line.strip()]
        raise ToolExecutionError(lines[-1] if lines else str(output))
    return str(output)


def decode_prelude() -> str:
    return "import base64,json,sys; a=json.loads(base64.b64decode(sys.argv[1])); "


def truncate(value: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(value) <= limit:
        return value, False
    return f"{value[:limit]}\n[truncated {len(value) - limit} chars]", True
