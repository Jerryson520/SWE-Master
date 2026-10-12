"""Ripgrep-backed content search tool."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolInfrastructureError, ToolOutput, truncate
from .ripgrep import RG_VERSION, RipgrepUnavailable, resolve_host_rg


_CONTAINER_RG = "/usr/local/bin/rg"


class GrepTool(BaseTool):
    name = "Grep"
    description = (
        "Search file contents using ripgrep. Supports regular expressions, file filters, context lines, "
        "and content, file-list, or count output modes."
    )
    parameters = (
        ParameterSpec("pattern", "string", "Regular expression to search for.", required=True),
        ParameterSpec("path", "string", "File or directory to search. Defaults to the repository root."),
        ParameterSpec("glob", "string", "Glob used to include or exclude files, passed to rg --glob."),
        ParameterSpec("type", "string", "File type passed to rg --type, such as py, js, rust, or go."),
        ParameterSpec(
            "output_mode", "string", "Output mode. Defaults to files_with_matches.",
            enum=("content", "files_with_matches", "count"),
        ),
        ParameterSpec("-i", "boolean", "Perform a case-insensitive search."),
        ParameterSpec("-n", "boolean", "Show line numbers in content mode. Defaults to true."),
        ParameterSpec("-A", "number", "Show this many lines after each match in content mode."),
        ParameterSpec("-B", "number", "Show this many lines before each match in content mode."),
        ParameterSpec("-C", "number", "Show this many lines before and after each match in content mode."),
        ParameterSpec("context", "number", "Alias for -C; ignored when -C is supplied."),
        ParameterSpec("head_limit", "number", "Maximum output entries. Defaults to 250; 0 means unlimited."),
        ParameterSpec("offset", "number", "Number of output entries to skip. Defaults to 0."),
        ParameterSpec("multiline", "boolean", "Allow matches to span lines and make dot match newlines."),
    )

    def __init__(self, output_chars: int = 30000, default_head_limit: int = 250, default_timeout: float = 30):
        self.output_chars = output_chars
        self.default_head_limit = default_head_limit
        self.default_timeout = default_timeout
        self._rg_path: str | None = None

    @staticmethod
    def _non_negative_integer(value: Any, name: str, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ToolCallError(f"{name} has invalid type")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolCallError(f"{name} has invalid type") from exc
        if parsed < 0:
            raise ToolCallError(f"{name} must be non-negative")
        return parsed

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = super().parse_arguments(raw)
        pattern = parsed.get("pattern")
        if not pattern:
            raise ToolCallError("pattern is required")
        output_mode = parsed.get("output_mode", "files_with_matches")
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise ToolCallError("output_mode must be one of: content, files_with_matches, count")
        return {
            "pattern": pattern,
            "path": parsed.get("path", "."),
            "glob": parsed.get("glob"),
            "type": parsed.get("type"),
            "output_mode": output_mode,
            "ignore_case": parsed.get("-i", False),
            "line_numbers": parsed.get("-n", True),
            "after": self._non_negative_integer(parsed.get("-A"), "-A", 0),
            "before": self._non_negative_integer(parsed.get("-B"), "-B", 0),
            "context": self._non_negative_integer(parsed.get("-C", parsed.get("context")), "-C", 0),
            "head_limit": self._non_negative_integer(parsed.get("head_limit"), "head_limit", self.default_head_limit),
            "offset": self._non_negative_integer(parsed.get("offset"), "offset", 0),
            "multiline": parsed.get("multiline", False),
        }

    def _ensure_rg(self, context: ToolContext) -> str:
        if self._rg_path is not None:
            return self._rg_path
        output, status = context.runtime.run("sh -lc 'command -v rg'", timeout=10, workdir=context.workdir)
        if str(status) == "0" and str(output).strip():
            self._rg_path = str(output).strip().splitlines()[-1]
            return self._rg_path
        try:
            host_rg = resolve_host_rg()
        except RipgrepUnavailable as exc:
            raise ToolCallError(f"ripgrep is unavailable: {exc}") from exc
        created, create_status = context.runtime.run(
            "mkdir -p /usr/local/bin", timeout=10, workdir=context.workdir,
        )
        if str(create_status) == "-1":
            raise ToolInfrastructureError(str(created))
        if str(create_status) != "0":
            raise ToolCallError(f"could not prepare ripgrep directory: {created}")
        try:
            context.runtime.copy_to_container(str(host_rg), _CONTAINER_RG)
        except Exception as exc:
            raise ToolInfrastructureError(f"could not copy ripgrep into the container: {exc}") from exc
        checked, check_status = context.runtime.run(
            f"chmod 755 {_CONTAINER_RG} && test \"$(command -v rg)\" = {_CONTAINER_RG} && rg --version",
            timeout=10,
            workdir=context.workdir,
        )
        if str(check_status) == "-1":
            raise ToolInfrastructureError(str(checked))
        if str(check_status) != "0" or f"ripgrep {RG_VERSION}" not in str(checked):
            raise ToolCallError(f"staged ripgrep is not runnable: {str(checked).strip()[:300]}")
        self._rg_path = _CONTAINER_RG
        return self._rg_path

    @staticmethod
    def _returncode(status: Any) -> int:
        text = str(status)
        if text == "0":
            return 0
        match = re.search(r"Exit code (\d+)", text)
        return int(match.group(1)) if match else -1

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        rg = self._ensure_rg(context)
        argv = [rg, "--color=never"]
        mode = args["output_mode"]
        if mode == "files_with_matches":
            argv.append("--files-with-matches")
        elif mode == "count":
            argv.append("--count")
        elif args["line_numbers"]:
            argv.append("--line-number")
        if args["ignore_case"]:
            argv.append("--ignore-case")
        if args["multiline"]:
            argv.extend(["--multiline", "--multiline-dotall"])
        if mode == "content":
            for flag, value in (("--after-context", args["after"]), ("--before-context", args["before"]), ("--context", args["context"])):
                if value:
                    argv.extend([flag, str(value)])
        if args["glob"]:
            argv.extend(["--glob", args["glob"]])
        if args["type"]:
            argv.extend(["--type", args["type"]])
        argv.extend(["--regexp", args["pattern"], "--", args["path"]])
        stdout, stderr, status = context.runtime.demux_run(
            " ".join(shlex.quote(item) for item in argv),
            timeout=max(1, int(min(context.timeout, self.default_timeout))),
            workdir=context.workdir,
        )
        returncode = self._returncode(status)
        if returncode == -1:
            raise ToolInfrastructureError(str(stderr or stdout or "ripgrep transport failed"))
        if returncode not in {0, 1}:
            message = str(stderr or stdout).strip()
            return ToolOutput(
                f"Error: {message or f'ripgrep exited with code {returncode}'}",
                success=False,
                metadata={"returncode": returncode, "output_mode": mode},
            )
        lines = [line[2:] if line.startswith("./") else line for line in str(stdout).splitlines()]
        if mode in {"files_with_matches", "count"}:
            lines.sort()
        total = len(lines)
        start = min(args["offset"], total)
        stop = total if args["head_limit"] == 0 else min(total, start + args["head_limit"])
        selected = lines[start:stop]
        content = "\n".join(selected) if selected else "No matches found."
        content, char_truncated = truncate(content, self.output_chars)
        return ToolOutput(
            content,
            metadata={
                "returncode": returncode,
                "output_mode": mode,
                "entries": total,
                "offset": args["offset"],
                "head_limit": args["head_limit"],
                "truncated": stop < total or char_truncated,
            },
        )
