"""Bash tool implementation."""

import re
import shlex
from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolInfrastructureError, ToolOutput, truncate


class BashTool(BaseTool):
    name = "Bash"
    description = "Run a shell command in the repository workdir."
    parameters = (
        ParameterSpec("command", "string", "Command to run.", required=True),
        ParameterSpec("description", "string", "Short description of the command."),
        ParameterSpec("timeout", "number", "Optional timeout in seconds."),
    )

    def __init__(self, output_chars: int = 30000, default_timeout: float = 120):
        self.output_chars = output_chars
        self.default_timeout = default_timeout

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(raw) if isinstance(raw, dict) else {}
        command = str(parsed.get("command") or "")
        if not command:
            raise ToolCallError("command is required")
        raw_timeout = parsed.get("timeout_ms") if parsed.get("timeout_ms") is not None else parsed.get("timeout")
        if raw_timeout is None:
            timeout = self.default_timeout
        else:
            try:
                value = float(raw_timeout)
            except (TypeError, ValueError) as exc:
                raise ToolCallError("timeout has invalid type") from exc
            timeout = max(0.1, value / 1000.0 if value > 1000 else value)
        return {"command": command, "timeout": timeout}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        timeout = args["timeout"]
        stdout, stderr, exit_status = context.runtime.demux_run(
            f"/bin/bash -lc {shlex.quote(args['command'])}",
            timeout=max(1, int(timeout)),
            workdir=context.workdir,
        )
        match = re.search(r"Exit code (\d+)", str(exit_status))
        returncode = int(match.group(1)) if match else (0 if str(exit_status) == "0" else -1)
        combined = f"{stdout}\n{stderr}"
        if returncode == 124 or (returncode == -1 and "too long" in combined.lower()):
            content = f"Timed out after {timeout:.1f}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            content, clipped = truncate(content, self.output_chars)
            return ToolOutput(content, success=False, metadata={"timeout": True, "returncode": returncode, "truncated": clipped})
        if returncode == -1:
            raise ToolInfrastructureError(combined.strip() or "Bash transport failed")
        content = f"Exit code: {returncode}\nSTDOUT:\n{stdout or ''}\nSTDERR:\n{stderr or ''}"
        content, clipped = truncate(content, self.output_chars)
        return ToolOutput(content, success=returncode == 0, metadata={"returncode": returncode, "truncated": clipped})
