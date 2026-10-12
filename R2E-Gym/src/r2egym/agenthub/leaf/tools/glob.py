"""Glob tool implementation."""

import json
from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolOutput, decode_prelude, run_python


class GlobTool(BaseTool):
    name = "Glob"
    description = "Find files by glob pattern."
    parameters = (
        ParameterSpec("pattern", "string", "Glob pattern, e.g. **/*.py.", required=True),
        ParameterSpec("path", "string", "Optional directory to search from."),
    )

    def __init__(self, limit: int = 1000):
        self.limit = limit

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(raw) if isinstance(raw, dict) else {}
        pattern = str(parsed.get("pattern") or "")
        if not pattern:
            raise ToolCallError("pattern is required")
        return {"pattern": pattern, "path": str(parsed.get("path") or ".")}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        source = decode_prelude() + (
            "import glob,json,os; from pathlib import Path; root=Path(a['path']).expanduser(); "
            "root=root if root.is_absolute() else Path.cwd()/root; cwd=Path.cwd().resolve(); out=[]; "
            "[(out.append(os.path.relpath(str(p.resolve()),str(cwd)) if os.path.commonpath([str(p.resolve()),str(cwd)])==str(cwd) else str(p))) for p in map(Path,glob.glob(str(root/a['pattern']),recursive=True)) if p.is_file()]; "
            "print(json.dumps(sorted(out)),end='')"
        )
        matches = json.loads(run_python(context, source, args))
        shown = matches[: max(1, self.limit)]
        suffix = "" if len(matches) <= self.limit else f"\n... {len(matches) - self.limit} more files"
        content = "\n".join(shown) + suffix if shown else "No files matched."
        return ToolOutput(content, metadata={"matches": len(matches), "truncated": len(matches) > self.limit})
