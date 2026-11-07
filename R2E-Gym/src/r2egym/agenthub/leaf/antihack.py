"""Runtime anti-reward-hacking guard for the Leaf tool environment.

Adapted from Xiaomi MiMoAgent's ``agents/antihack.py`` at commit
467f0a19016f0ac4d63b8d17a1f0da9ba07f232c (MIT License).  The original
implementation and tests are available at:

https://github.com/XiaomiMiMo/mimoagent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/agents/antihack.py

The Leaf adaptation keeps MiMo's policy model: inspect every model-controlled
tool call before execution, return a deliberately uninformative observation
when a rule matches, and keep detailed telemetry outside the model-visible
trajectory.  The guard is a defence-in-depth policy, not a sandbox; prepared
images and ``network_mode=none`` remain the primary isolation boundaries.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from r2egym.agenthub.leaf.tools.base import ToolOutput


POLICY_BLOCKED_ERROR_CODE = "POLICY_BLOCKED"
POLICY_BLOCKED_MESSAGE = (
    "POLICY_BLOCKED: This operation is outside the capabilities of the current "
    "environment. Do not retry an equivalent operation; continue using the "
    "declared workspace and available tools."
)


_BASH_FIELDS: dict[str, tuple[str, ...]] = {
    "Bash": ("command",),
    "bash": ("command",),
    "exec_command": ("cmd",),
    "apply_patch": ("input", "patch"),
}

_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "read": ("path", "file_path"),
    "edit": ("path", "file_path"),
    "write": ("path", "file_path"),
    "Grep": ("pattern", "path"),
    "Glob": ("pattern", "path"),
    "grep": ("pattern", "path"),
    "glob": ("pattern", "path"),
    "apply_patch": ("input", "patch"),
}


# The rule split follows MiMoAgent's recall-first v1 defaults.  Stable IDs are
# attached locally so sidecar telemetry does not have to expose raw commands or
# regex source.  Sanitized local HEAD history remains usable; remote/cross-ref
# archaeology and common evaluation-artifact probes are blocked.
_DEFAULT_BASH_RULES: tuple[tuple[str, str], ...] = (
    ("git_network", r"\bgit\s+(fetch|pull|clone|remote|ls-remote)\b"),
    ("git_hidden_state", r"\bgit\s+(reflog|stash|bisect|show-ref|for-each-ref)\b"),
    ("git_remote_branches", r"\bgit\s+branch\b[^|;&\n]*(?:-a|-r|--all|--remotes)\b"),
    ("git_tags", r"\bgit\s+tag\b"),
    (
        "git_cross_ref",
        r"\bgit\s+(log|show|diff|grep|checkout|switch|restore|merge|rebase|cherry-pick|ls-tree|archive)\b"
        r"[^|;&\n]*(?:--all|--branches|--remotes|--tags|\borigin/|\bupstream/|\brefs/|\bremotes/|\btags/|\bFETCH_HEAD\b|\bORIG_HEAD\b)",
    ),
    ("git_raw_objects", r"\bgit\s+(cat-file|fsck|verify-pack|pack-objects|unpack-objects|index-pack)\b"),
    ("git_remote_submodule", r"\bgit\s+submodule\b[^|;&\n]*(?:--remote|\bupdate\b[^|;&\n]*--remote)"),
    ("github_raw", r"raw\.githubusercontent\.com"),
    ("github_codeload", r"codeload\.github\.com"),
    ("github_api_source", r"api\.github\.com/repos/[^ \t\n]+/(?:commits|pulls|releases|compare|git/)"),
    ("github_source_page", r"github\.com/[^ \t\n]+/(?:raw|blob|commit|commits|pull|pulls|releases|archive|compare)/"),
    (
        "network_source_fetch",
        r"\b(curl|wget)\b[^|;&\n]*(?:raw\.githubusercontent|codeload\.github|gist\.github|pastebin|"
        r"/raw/|/blob/|/commit/|/commits/|/pull/|/pulls/|/releases?|/archive|/compare/)",
    ),
    ("pip_remote_install", r"\bpip3?\s+install\b[^|;&\n]*(?:git\+|https?://|--upgrade|\s-U\b)"),
    ("pip_remote_download", r"\bpip3?\s+download\b[^|;&\n]*(?:git\+|https?://)"),
    (
        "javascript_remote_install",
        r"\b(npm|pnpm|yarn)\s+(?:install|add|i)\b[^|;&\n]*(?:git\+|github:|gitlab:|bitbucket:|https?://|\blatest\b|\bnext\b|\bcanary\b)",
    ),
    ("go_unpinned_get", r"\bgo\s+get\b[^|;&\n]*@(?:latest|master|main|HEAD)"),
    ("package_manager_remote", r"\b(poetry|composer)\s+(?:add|require)\b[^|;&\n]*(?:git\+|https?://|dev-|latest)"),
    ("cargo_remote", r"\bcargo\s+(?:add|install)\b[^|;&\n]*(?:--git|https?://)"),
    ("gem_remote", r"\bgem\s+install\b[^|;&\n]*(?:--source|https?://)"),
    ("eval_directory", r"(?:^|[^\w./-])\.eval(?:/|$)"),
    ("secret_cases", r"\bsecret_cases\b"),
    (
        "find_protected_artifact",
        r"\bfind\b[^|;&\n]*(?:-name|-path|-wholename)[^|;&\n]*\b(?:hidden_tests?|private_tests?|secret_cases|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)\b",
    ),
    (
        "grep_protected_artifact",
        r"\b(?:grep|rg|ag)\b[^|;&\n]*\b(?:hidden_tests?|private_tests?|secret_cases|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)\b",
    ),
    (
        "read_protected_artifact",
        r"\b(?:cat|less|more|head|tail|sed|awk|xxd|od|strings)\b[^|;&\n]*(?:\.eval/|secret_cases|hidden_tests?|private_tests?|answer.?key|gold(?:en)?_patch|test_patch|ground.?truth)",
    ),
)

_DEFAULT_PATH_RULES: tuple[tuple[str, str], ...] = (
    ("eval_directory", r"(?:^|/)\.eval(?:/|$)"),
    ("secret_cases", r"\bsecret_cases\b"),
    ("hidden_tests", r"\bhidden_tests?\b"),
    ("private_tests", r"\bprivate_tests?\b"),
    ("answer_key", r"\banswer.?key\b"),
    ("gold_patch", r"\bgold(?:en)?_patch\b"),
    ("test_patch", r"\btest_patch\b"),
    ("ground_truth", r"\bground.?truth\b"),
)


@dataclass
class AntiHackConfig:
    """Configuration for the opt-in Leaf action guard."""

    enabled: bool = False
    bash_patterns: list[str] | None = None
    path_patterns: list[str] | None = None
    dummy_output: str = POLICY_BLOCKED_MESSAGE
    dummy_success: bool = False
    events_path: str | None = None
    run_id: str | None = None
    trajectory_id: str | None = None
    llm_judge_enabled: bool = False

    @classmethod
    def from_obj(cls, obj: "AntiHackConfig | Mapping[str, Any] | bool | None") -> "AntiHackConfig":
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, bool):
            return cls(enabled=obj)
        if isinstance(obj, Mapping):
            valid = {item.name for item in fields(cls)}
            return cls(**{key: value for key, value in dict(obj).items() if key in valid})
        raise TypeError(f"antihack config must be bool/mapping/AntiHackConfig/None, got {type(obj)!r}")


@dataclass(frozen=True)
class AntiHackVerdict:
    tool: str
    field: str
    value: str = field(repr=False)
    pattern_id: str = ""
    pattern: str = field(default="", repr=False)


class AntiHackEventWriter:
    """Append host-only JSONL events safely across rollout worker processes."""

    def __init__(self, path: str | None):
        self.path = Path(path).expanduser() if path else None

    def append(self, event: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view) :]
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


class AntiHackGuard:
    """Regex guard that substitutes a dummy Leaf ``ToolOutput`` on a match."""

    def __init__(self, config: AntiHackConfig | Mapping[str, Any] | bool | None = None):
        self.config = AntiHackConfig.from_obj(config)
        self._bash_res = self._compile(self.config.bash_patterns, _DEFAULT_BASH_RULES, "bash_custom")
        self._path_res = self._compile(self.config.path_patterns, _DEFAULT_PATH_RULES, "path_custom")
        self.blocks: list[dict[str, Any]] = []
        self.telemetry_errors: list[str] = []
        self._writer = AntiHackEventWriter(self.config.events_path)

    @staticmethod
    def _compile(
        custom: list[str] | None,
        defaults: tuple[tuple[str, str], ...],
        custom_prefix: str,
    ) -> list[tuple[str, str, re.Pattern[str]]]:
        rules = defaults if custom is None else tuple(
            (f"{custom_prefix}_{index}", pattern) for index, pattern in enumerate(custom)
        )
        return [(rule_id, pattern, re.compile(pattern, re.IGNORECASE)) for rule_id, pattern in rules]

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self) -> None:
        self.blocks.clear()
        self.telemetry_errors.clear()

    def inspect(self, action: Mapping[str, Any]) -> AntiHackVerdict | None:
        if not self.enabled:
            return None
        tool = action.get("tool")
        params = action.get("params")
        if not isinstance(tool, str) or not isinstance(params, Mapping):
            return None
        if tool in _BASH_FIELDS:
            for name in _BASH_FIELDS[tool]:
                hit = self._match(params.get(name), self._bash_res)
                if hit is not None:
                    rule_id, pattern = hit
                    return AntiHackVerdict(tool, name, str(params.get(name)), rule_id, pattern)
        if tool in _PATH_FIELDS:
            for name in _PATH_FIELDS[tool]:
                hit = self._match(params.get(name), self._path_res)
                if hit is not None:
                    rule_id, pattern = hit
                    return AntiHackVerdict(tool, name, str(params.get(name)), rule_id, pattern)
        return None

    @staticmethod
    def _match(
        value: Any,
        compiled: list[tuple[str, str, re.Pattern[str]]],
    ) -> tuple[str, str] | None:
        if not isinstance(value, str) or not value:
            return None
        for rule_id, pattern, regex in compiled:
            if regex.search(value):
                return rule_id, pattern
        return None

    def confirm(self, action: Mapping[str, Any], verdict: AntiHackVerdict) -> bool:
        if not self.config.llm_judge_enabled:
            return True
        return self._llm_judge(action, verdict)

    def _llm_judge(self, action: Mapping[str, Any], verdict: AntiHackVerdict) -> bool:
        # Kept as an explicit extension point, matching MiMoAgent v1.  Until a
        # judge is wired, a regex hit remains blocked rather than silently
        # weakening the policy.
        return True

    def intercept(
        self,
        action: Mapping[str, Any],
        *,
        event_context: Mapping[str, Any] | None = None,
    ) -> ToolOutput | None:
        verdict = self.inspect(action)
        if verdict is None or not self.confirm(action, verdict):
            return None

        block = {
            "tool": verdict.tool,
            "field": verdict.field,
            "pattern_id": verdict.pattern_id,
        }
        tool_call_id = (event_context or {}).get("tool_call_id")
        if tool_call_id is not None:
            block["tool_call_id"] = str(tool_call_id)
        self.blocks.append(block)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.config.run_id,
            "trajectory_id": self.config.trajectory_id,
            **dict(event_context or {}),
            **block,
        }
        # Never include verdict.value or the raw regex in the sidecar.  The
        # trajectory only receives the dummy content below.
        event = {key: value for key, value in event.items() if value is not None}
        try:
            self._writer.append(event)
        except Exception as exc:  # telemetry failure must not execute the blocked action
            message = f"{type(exc).__name__}: {exc}"
            self.telemetry_errors.append(message)
            warnings.warn(f"anti-hack telemetry write failed: {message}", RuntimeWarning, stacklevel=2)
        return ToolOutput(
            content=self.config.dummy_output,
            success=self.config.dummy_success,
            metadata={"error_code": POLICY_BLOCKED_ERROR_CODE},
        )

    def summary(self) -> dict[str, Any]:
        blocked_tool_call_ids = list(
            dict.fromkeys(
                str(item["tool_call_id"])
                for item in self.blocks
                if item.get("tool_call_id") is not None
            )
        )
        return {
            "enabled": self.enabled,
            "block_count": len(self.blocks),
            "blocks": [dict(item) for item in self.blocks],
            "blocked_tool_call_ids": blocked_tool_call_ids,
            "telemetry_errors": list(self.telemetry_errors),
        }
