import subprocess
import shutil
from pathlib import Path

import pytest

from r2egym.agenthub.leaf import TOOL_DEFINITIONS, AgentAction, ToolCall, ToolResult
from r2egym.agenthub.leaf.tools import ToolContext, ToolRegistry, create_default_registry
from r2egym.agenthub.leaf.tools.base import BaseTool, ToolOutput
from r2egym.agenthub.leaf.tools.ripgrep import resolve_host_rg


class LocalRuntime:
    def run(self, command, timeout, workdir):
        try:
            result = subprocess.run(
                command, shell=True, cwd=workdir, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        return result.stdout, str(result.returncode) if result.returncode == 0 else f"Error: Exit code {result.returncode}"

    def demux_run(self, command, timeout, workdir):
        try:
            result = subprocess.run(
                command, shell=True, cwd=workdir, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return exc.stdout or "The command took too long to execute", exc.stderr or "", "Error: Exit code 124"
        status = "0" if result.returncode == 0 else f"Error: Exit code {result.returncode}"
        return result.stdout, result.stderr, status

    def copy_to_container(self, src_path, dest_path):
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)


@pytest.fixture
def tools(tmp_path):
    registry = create_default_registry()
    registry.get("Grep")._rg_path = str(resolve_host_rg(download=False))
    return registry, ToolContext(LocalRuntime(), str(tmp_path), 5)


def execute(tools, name, arguments):
    registry, context = tools
    return registry.execute(name, arguments, context)


def test_read_write_edit_glob_and_bash(tools):
    assert execute(tools, "Write", {"file_path": "a.txt", "content": "one\ntwo\n"}).content.startswith("Wrote a.txt")
    assert execute(tools, "Read", {"file_path": "a.txt", "offset": 2, "limit": 1}).content == "a.txt\n2. two"
    assert execute(tools, "Edit", {"file_path": "a.txt", "old_string": "two", "new_string": "three"}).content.startswith("Edited a.txt")
    assert execute(tools, "Glob", {"pattern": "*.txt"}).content == "a.txt"
    result = execute(tools, "Bash", {"command": "printf ok"})
    assert result.success
    assert result.content == "Exit code: 0\nSTDOUT:\nok\nSTDERR:\n"


def test_grep_modes_filters_unicode_and_limits(tools):
    _, context = tools
    Path(context.workdir, "a.py").write_text("Alpha\nneedle one\n中文 needle\n")
    Path(context.workdir, "b.txt").write_text("needle two\n")
    files = execute(tools, "Grep", {"pattern": "needle", "glob": "*.py"})
    assert files.success and files.content == "a.py"
    content = execute(tools, "Grep", {"pattern": "中文|Alpha", "path": "a.py", "output_mode": "content"})
    assert content.success
    assert "1:Alpha" in content.content and "3:中文 needle" in content.content
    count = execute(tools, "Grep", {"pattern": "needle", "output_mode": "count"})
    assert count.success and count.content.splitlines() == ["a.py:2", "b.txt:1"]
    limited = execute(tools, "Grep", {"pattern": "needle", "output_mode": "content", "offset": 1, "head_limit": 1})
    assert limited.success and len(limited.content.splitlines()) == 1
    assert limited.metadata["truncated"]


def test_grep_no_match_and_invalid_regex_are_observable(tools):
    _, context = tools
    Path(context.workdir, "a.txt").write_text("hello\n")
    missing = execute(tools, "Grep", {"pattern": "absent"})
    assert missing.success and missing.content == "No matches found."
    invalid = execute(tools, "Grep", {"pattern": "[", "output_mode": "content"})
    assert not invalid.success
    assert invalid.metadata["returncode"] == 2
    assert "Error:" in invalid.content


def test_grep_argument_validation(tools):
    assert execute(tools, "Grep", {}).content == "ValueError: pattern is required"
    invalid_mode = execute(tools, "Grep", {"pattern": "x", "output_mode": "wrong"})
    assert not invalid_mode.success and "output_mode must be one of" in invalid_mode.content
    negative = execute(tools, "Grep", {"pattern": "x", "head_limit": -1})
    assert not negative.success and "must be non-negative" in negative.content


def test_edit_requires_unique_match(tools):
    _, context = tools
    Path(context.workdir, "a.txt").write_text("x\nx\n")
    result = execute(tools, "Edit", {"file_path": "a.txt", "old_string": "x", "new_string": "y"})
    assert not result.success
    assert "matched 2 times" in result.content
    assert Path(context.workdir, "a.txt").read_text() == "x\nx\n"


def test_errors_are_observations(tools):
    assert execute(tools, "Unknown", {}).content == "Unknown tool: Unknown"
    assert execute(tools, "Read", {}).content == "ValueError: file_path is required"


def test_protocol_types_are_generic():
    call = ToolCall("id", "Read", {"file_path": "a"})
    action = AgentAction((call,))
    assert not action.is_terminal
    assert AgentAction().is_terminal
    assert ToolResult("id", "Read", "ok").tool_call_id == "id"


def test_tool_definitions_match_existing_wire_contract():
    assert [item["function"]["name"] for item in TOOL_DEFINITIONS] == ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
    assert TOOL_DEFINITIONS[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "offset": {"type": "integer", "description": "Optional 1-based starting line."},
            "limit": {"type": "integer", "description": "Optional maximum number of lines."},
        },
        "required": ["file_path"],
    }
    grep_schema = TOOL_DEFINITIONS[4]["function"]["parameters"]
    assert grep_schema["required"] == ["pattern"]
    assert grep_schema["properties"]["output_mode"]["enum"] == ["content", "files_with_matches", "count"]


def test_registry_does_not_hide_tool_key_errors(tmp_path):
    class BrokenTool(BaseTool):
        name = "Broken"
        description = "broken"

        def execute(self, arguments, context):
            raise KeyError("inside")

    registry = ToolRegistry([BrokenTool()])
    output = registry.execute("Broken", {}, ToolContext(LocalRuntime(), str(tmp_path), 5))
    assert output.content == "KeyError: 'inside'"
    assert output.content != "Unknown tool: Broken"


def test_duplicate_registration_is_rejected():
    registry = create_default_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("Read"))
