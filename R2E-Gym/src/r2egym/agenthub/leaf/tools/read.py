"""Read tool implementation."""

from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolOutput, decode_prelude, run_python, truncate


class ReadTool(BaseTool):
    name = "Read"
    description = "Read a text file. Returns numbered lines."
    parameters = (
        ParameterSpec("file_path", "string", "Path to the file to read.", required=True),
        ParameterSpec("offset", "integer", "Optional 1-based starting line."),
        ParameterSpec("limit", "integer", "Optional maximum number of lines."),
    )

    def __init__(self, output_chars: int = 20000):
        self.output_chars = output_chars

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(raw) if isinstance(raw, dict) else {}
        path = parsed.get("file_path") or parsed.get("path")
        if not path:
            raise ToolCallError("file_path is required")
        try:
            offset = max(1, int(parsed.get("offset") or 1))
            limit = int(parsed.get("limit") or 0)
        except (TypeError, ValueError) as exc:
            raise ToolCallError("offset or limit has invalid type") from exc
        return {"file_path": str(path), "offset": offset, "limit": limit}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        source = decode_prelude() + (
            "import os; from pathlib import Path; p=Path(a['file_path']).expanduser(); "
            "p=p if p.is_absolute() else Path.cwd()/p; "
            "lines=p.read_text(encoding='utf-8',errors='replace').splitlines(); "
            "o=a['offset']; l=a['limit']; e=len(lines) if l<=0 else min(len(lines),o-1+l); "
            "pr=str(p.resolve()); cwd=str(Path.cwd().resolve()); disp=(os.path.relpath(pr,cwd) if os.path.commonpath([pr,cwd])==cwd else str(p)); "
            "print(disp+'\\n'+'\\n'.join(f'{n}. {line}' for n,line in enumerate(lines[o-1:e],start=o)),end='')"
        )
        output = run_python(context, source, args)
        content, clipped = truncate(output, self.output_chars)
        return ToolOutput(content, metadata={"truncated": clipped})
