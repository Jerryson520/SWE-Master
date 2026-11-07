"""Fail-closed SWE-smith Git sanitization shared by runtime and audit."""

from __future__ import annotations

import shlex


SWESMITH_SANITIZATION_VERSION = "swesmith-git-reinit-v1"


def build_swesmith_sanitize_command(instance_id: str) -> str:
    """Replace the prepared repository metadata with one parentless baseline."""
    ref = shlex.quote(f"refs/r2egym/instances/{instance_id}")
    return (
        "set -euo pipefail; cd /testbed; "
        "test -d .git; test ! -L .git; test ! -f .git; "
        "test ! -e .git/objects/info/alternates; "
        "test ! -d .git/worktrees || test -z \"$(find .git/worktrees -mindepth 1 -print -quit)\"; "
        f"git rev-parse --verify {ref}^{{commit}} > /tmp/r2egym-sensitive-shas; "
        f"git rev-parse --verify {ref}~1^{{commit}} >> /tmp/r2egym-sensitive-shas; "
        f"git rev-parse --verify {ref}~2^{{commit}} >> /tmp/r2egym-sensitive-shas; "
        "git for-each-ref --format='%(objectname)' >> /tmp/r2egym-sensitive-shas; "
        f"git checkout --detach {ref}~1; git reset --hard {ref}~1; git clean -fdq; "
        "rm -f /opt/r2egym/image-manifest.json /opt/r2egym/instances.txt; "
        "{ git config --global --get-regexp '^url\\..*\\.insteadof$' 2>/dev/null || true; } | "
        "cut -d' ' -f1 | while IFS= read -r key; do git config --global --unset-all \"$key\" || true; done; "
        "rm -rf .git; git init -q; git config core.logAllRefUpdates false; "
        "git add -A; git -c user.name=R2EGym -c user.email=r2egym@localhost "
        "commit -q -m 'R2EGym task baseline'; git branch -M r2egym-baseline; "
        "test \"$(git for-each-ref --format='%(refname)')\" = refs/heads/r2egym-baseline; "
        "test \"$(git rev-list --parents -n 1 HEAD | wc -w)\" -eq 1; "
        "test -z \"$(git remote)\"; test ! -d .git/logs; "
        "test ! -e .git/objects/info/alternates; test ! -d .git/worktrees; "
        "test -z \"$(git status --porcelain)\"; "
        "while IFS= read -r sha; do test -z \"$sha\" || ! git cat-file -e \"$sha^{object}\" 2>/dev/null; done "
        "< /tmp/r2egym-sensitive-shas; rm -f /tmp/r2egym-sensitive-shas"
    )
