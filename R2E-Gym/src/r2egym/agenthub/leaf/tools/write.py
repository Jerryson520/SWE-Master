"""Write tool implementation."""

from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolOutput, decode_prelude, run_python


class WriteTool(BaseTool):
    name = "Write"
    description = "Write a complete file, creating parent directories when needed."
    parameters = (
        ParameterSpec("file_path", "string", "Path to write.", required=True),
        ParameterSpec("content", "string", "Full file contents.", required=True),
    )

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(raw) if isinstance(raw, dict) else {}
        path = parsed.get("file_path") or parsed.get("path")
        if not path:
            raise ToolCallError("file_path is required")
        return {"file_path": str(path), "content": str(parsed.get("content", ""))}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        source = decode_prelude() + (
            "import os; from pathlib import Path; p=Path(a['file_path']).expanduser(); "
            "p=p if p.is_absolute() else Path.cwd()/p; p.parent.mkdir(parents=True,exist_ok=True); "
            "p.write_text(a['content'],encoding='utf-8'); "
            "pr=str(p.resolve()); cwd=str(Path.cwd().resolve()); disp=(os.path.relpath(pr,cwd) if os.path.commonpath([pr,cwd])==cwd else str(p)); "
            "print(f\"Wrote {disp} ({len(a['content'])} bytes).\",end='')"
        )
        return ToolOutput(run_python(context, source, args))
