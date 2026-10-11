from pathlib import Path

from r2egym.agenthub.leaf.tool_runner import run_tool


def test_read_write_edit_glob_and_bash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_tool("Write", {"file_path": "a.txt", "content": "one\ntwo\n"}).startswith("Wrote a.txt")
    assert run_tool("Read", {"file_path": "a.txt", "offset": 2, "limit": 1}) == "a.txt\n2. two"
    assert run_tool("Edit", {"file_path": "a.txt", "old_string": "two", "new_string": "three"}).startswith("Edited a.txt")
    assert run_tool("Glob", {"pattern": "*.txt"}) == "a.txt"
    assert "Exit code: 0" in run_tool("Bash", {"command": "printf ok"})


def test_edit_requires_unique_match(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("a.txt").write_text("x\nx\n")
    assert "matched 2 times" in run_tool("Edit", {"file_path": "a.txt", "old_string": "x", "new_string": "y"})
    assert Path("a.txt").read_text() == "x\nx\n"


def test_errors_are_observations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_tool("Unknown", {}) == "Unknown tool: Unknown"
    assert run_tool("Read", {}) == "ValueError: file_path is required"

