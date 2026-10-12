"""Exact-string edit tool implementation."""

from typing import Any

from .base import BaseTool, ParameterSpec, ToolCallError, ToolContext, ToolExecutionError, ToolOutput, decode_prelude, run_python


class EditTool(BaseTool):
    name = "Edit"
    description = "Replace an exact string in a text file."
    parameters = (
        ParameterSpec("file_path", "string", "Path to edit.", required=True),
        ParameterSpec("old_string", "string", "Exact text to replace.", required=True),
        ParameterSpec("new_string", "string", "Replacement text.", required=True),
        ParameterSpec("replace_all", "boolean", "Replace every occurrence."),
    )

    def parse_arguments(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(raw) if isinstance(raw, dict) else {}
        path = parsed.get("file_path") or parsed.get("path")
        if not path:
            raise ToolCallError("file_path is required")
        old = str(parsed.get("old_string") or "")
        if not old:
            raise ToolCallError("old_string must be non-empty")
        value = parsed.get("replace_all", False)
        replace_all = value if isinstance(value, bool) else str(value or "").strip().lower() in {"1", "true", "yes", "on"}
        return {"file_path": str(path), "old_string": old, "new_string": str(parsed.get("new_string") or ""), "replace_all": replace_all}

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutput:
        args = self.parse_arguments(arguments)
        source = decode_prelude() + (
            "import os; from pathlib import Path; p=Path(a['file_path']).expanduser(); "
            "p=p if p.is_absolute() else Path.cwd()/p; t=p.read_text(encoding='utf-8',errors='replace'); "
            "c=t.count(a['old_string']); "
            "assert c, 'old_string was not found'; "
            "assert c==1 or a['replace_all'], f\"old_string matched {c} times; set replace_all=true or use a more specific old_string\"; "
            "p.write_text(t.replace(a['old_string'],a['new_string']) if a['replace_all'] else t.replace(a['old_string'],a['new_string'],1),encoding='utf-8'); "
            "changed=c if a['replace_all'] else 1; "
            "pr=str(p.resolve()); cwd=str(Path.cwd().resolve()); disp=(os.path.relpath(pr,cwd) if os.path.commonpath([pr,cwd])==cwd else str(p)); "
            "print(f\"Edited {disp} ({changed} replacement{'s' if changed != 1 else ''}).\",end='')"
        )
        try:
            return ToolOutput(run_python(context, source, args))
        except ToolExecutionError as exc:
            message = str(exc)
            if "AssertionError: " in message:
                message = message.rsplit("AssertionError: ", 1)[1].strip()
                return ToolOutput(f"ValueError: {message}", success=False, metadata={"error_type": "ValueError"})
            raise
