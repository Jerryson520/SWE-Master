"""Dependency-free Leaf executor copied into each task container."""

import glob as globlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def run_tool(name: str, arguments: Dict[str, Any]) -> str:
    try:
        return {
            "Read": _read,
            "Write": _write,
            "Edit": _edit,
            "Glob": _glob,
            "Bash": _bash,
        }[name](arguments)
    except KeyError:
        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _read(arguments: Dict[str, Any]) -> str:
    path = _path(arguments, "file_path")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    offset = max(1, _optional_int(arguments.get("offset"), 1))
    limit = _optional_int(arguments.get("limit"), 0)
    end = len(lines) if limit <= 0 else min(len(lines), offset - 1 + limit)
    selected = "\n".join(
        f"{number}. {line}" for number, line in enumerate(lines[offset - 1 : end], start=offset)
    )
    return _truncate(f"{_display(path)}\n{selected}", _int_env("LEAF_TOOL_OUTPUT_CHARS", 20000))


def _write(arguments: Dict[str, Any]) -> str:
    path = _path(arguments, "file_path")
    content = str(arguments.get("content", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {_display(path)} ({len(content)} bytes)."


def _edit(arguments: Dict[str, Any]) -> str:
    path = _path(arguments, "file_path")
    old = str(arguments.get("old_string") or "")
    new = str(arguments.get("new_string") or "")
    if not old:
        raise ValueError("old_string must be non-empty")
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    replace_all = _truthy(arguments.get("replace_all"))
    if count == 0:
        raise ValueError("old_string was not found")
    if count != 1 and not replace_all:
        raise ValueError(f"old_string matched {count} times; set replace_all=true or use a more specific old_string")
    path.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
    changed = count if replace_all else 1
    return f"Edited {_display(path)} ({changed} replacement{'s' if changed != 1 else ''})."


def _glob(arguments: Dict[str, Any]) -> str:
    pattern = str(arguments.get("pattern") or "")
    if not pattern:
        raise ValueError("pattern is required")
    root = _resolve(arguments.get("path") or ".")
    matches = sorted(
        _display(Path(match)) for match in globlib.glob(str(root / pattern), recursive=True) if Path(match).is_file()
    )
    limit = max(1, _int_env("LEAF_GLOB_LIMIT", 1000))
    shown = matches[:limit]
    suffix = "" if len(matches) <= limit else f"\n... {len(matches) - limit} more files"
    return "\n".join(shown) + suffix if shown else "No files matched."


def _bash(arguments: Dict[str, Any]) -> str:
    command = str(arguments.get("command") or "")
    if not command:
        raise ValueError("command is required")
    timeout = _timeout_seconds(arguments)
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=max(0.1, timeout), check=False,
        )
        return _truncate(
            f"Exit code: {completed.returncode}\nSTDOUT:\n{completed.stdout or ''}\nSTDERR:\n{completed.stderr or ''}",
            _int_env("LEAF_BASH_OUTPUT_CHARS", 30000),
        )
    except subprocess.TimeoutExpired as exc:
        return _truncate(
            f"Timed out after {timeout:.1f}s\nSTDOUT:\n{_output(exc.stdout)}\nSTDERR:\n{_output(exc.stderr)}",
            _int_env("LEAF_BASH_OUTPUT_CHARS", 30000),
        )


def _timeout_seconds(arguments: Dict[str, Any]) -> float:
    if arguments.get("timeout_ms") is not None:
        return max(0.1, float(arguments["timeout_ms"]) / 1000.0)
    raw = arguments.get("timeout")
    if raw is None:
        return float(_int_env("LEAF_BASH_TIMEOUT_SEC", 120))
    value = float(raw)
    return max(0.1, value / 1000.0 if value > 1000 else value)


def _path(arguments: Dict[str, Any], key: str) -> Path:
    value = arguments.get(key) or arguments.get("path")
    if not value:
        raise ValueError(f"{key} is required")
    return _resolve(value)


def _resolve(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _optional_int(value: Any, default: int) -> int:
    return default if value in (None, "") else int(value)


def _truthy(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _truncate(value: str, limit: int) -> str:
    return value if limit <= 0 or len(value) <= limit else f"{value[:limit]}\n[truncated {len(value) - limit} chars]"


def _output(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        os.chdir(str(payload.get("workdir") or "/testbed"))
        result = run_tool(str(payload.get("tool") or ""), payload.get("args") or {})
    except Exception as exc:
        result = f"leaf tool_runner error: {type(exc).__name__}: {exc}"
    print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
