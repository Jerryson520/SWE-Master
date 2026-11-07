"""R2E-Gym-Subset adapter for base-only prepared images."""

from __future__ import annotations

import json
import shlex
from typing import Any

from r2egym.commit_models.diff_classes import ParsedCommit

from .base import ImageBuildContext, VerificationContext
from .r2e_verifier import verify_r2e_image


SANITIZATION_VERSION = "r2e-base-history-v1"


class R2EImageAdapter:
    dataset_type = "r2e-gym-subset"
    requires_flatten = False

    def create_prepare_script(self, context: ImageBuildContext) -> str:
        if len(context.tasks) != 1:
            raise RuntimeError("R2E image preparation requires exactly one task")
        task = context.tasks[0]
        commit = ParsedCommit(**json.loads(task["parsed_commit_content"]))
        if commit.new_commit_hash != task["commit_hash"]:
            raise RuntimeError("R2E solution commit mismatch")
        base = shlex.quote(commit.old_commit_hash)
        solution = shlex.quote(commit.new_commit_hash)
        return f"""#!/usr/bin/env bash
set -euo pipefail
test -d /testbed/.git
test -f /testbed/run_tests.sh || test -f /root/run_tests.sh
test -d /r2e_tests || test -d /testbed/r2e_tests || test -d /root/r2e_tests
git config --global --add safe.directory /testbed
SOURCE_BASE_SHA="$(git -C /testbed rev-parse {base}^{{commit}})"
test "$(git -C /testbed rev-parse HEAD^{{commit}})" = "$SOURCE_BASE_SHA"
: > /tmp/r2egym-safe-tags
while IFS= read -r tag_ref; do
  tag_commit="$(git -C /testbed rev-parse "$tag_ref^{{commit}}" 2>/dev/null || true)"
  if [[ -n "$tag_commit" ]] && git -C /testbed merge-base --is-ancestor "$tag_commit" "$SOURCE_BASE_SHA"; then
    tag_name="${{tag_ref#refs/tags/}}"
    tag_object="$(git -C /testbed rev-parse "$tag_ref")"
    printf '%s\t%s\t%s\n' "$tag_name" "$tag_object" "$tag_commit" >> /tmp/r2egym-safe-tags
  fi
done < <(git -C /testbed for-each-ref refs/tags --format='%(refname)')
sort -u -o /tmp/r2egym-safe-tags /tmp/r2egym-safe-tags
git -C /testbed ls-files -s -z > /tmp/r2egym-tracked-index
UPSTREAM_INDEX_DIGEST="$(sha256sum /tmp/r2egym-tracked-index | cut -d' ' -f1)"
find /testbed -mindepth 2 -name .git -exec rm -rf -- {{}} +
git -C /testbed read-tree --empty
while IFS= read -r -d '' index_record; do
  index_meta="${{index_record%%$'\t'*}}"
  tracked_path="${{index_record#*$'\t'}}"
  read -r file_mode blob_sha stage <<<"$index_meta"
  test "$stage" = 0
  if [[ ! -e "/testbed/$tracked_path" && ! -L "/testbed/$tracked_path" ]]; then
    continue
  fi
  if [[ "$file_mode" = 120000 ]]; then
    actual_sha="$(printf '%s' "$(readlink -- "/testbed/$tracked_path")" | git -C /testbed hash-object -w --stdin)"
  elif [[ "$file_mode" = 160000 ]]; then
    actual_sha="$blob_sha"
  else
    actual_sha="$(git -C /testbed hash-object -w -- "/testbed/$tracked_path")"
  fi
  git -C /testbed update-index --add --cacheinfo "$file_mode" "$actual_sha" "$tracked_path"
done < /tmp/r2egym-tracked-index
PREPARED_TREE_SHA="$(git -C /testbed write-tree)"
SOURCE_TREE_SHA="$(git -C /testbed rev-parse "$SOURCE_BASE_SHA^{{tree}}")"
if [[ "$PREPARED_TREE_SHA" = "$SOURCE_TREE_SHA" ]]; then
  PREPARED_BASE_SHA="$SOURCE_BASE_SHA"
else
  PREPARED_BASE_SHA="$(printf '%s\n' 'R2EGym prepared environment' | git -C /testbed -c user.name=R2EGym -c user.email=r2egym@localhost commit-tree "$PREPARED_TREE_SHA" -p "$SOURCE_BASE_SHA")"
fi
while IFS= read -r ref_name; do
  git -C /testbed update-ref -d "$ref_name"
done < <(git -C /testbed for-each-ref --format='%(refname)')
git -C /testbed symbolic-ref HEAD refs/heads/r2egym-baseline
git -C /testbed update-ref refs/heads/r2egym-baseline "$PREPARED_BASE_SHA"
while IFS=$'\t' read -r tag_name tag_object tag_commit; do
  git -C /testbed update-ref "refs/tags/$tag_name" "$tag_object"
done < /tmp/r2egym-safe-tags
while IFS= read -r remote_name; do
  git -C /testbed remote remove "$remote_name"
done < <(git -C /testbed remote)
while IFS= read -r branch_section; do
  git -C /testbed config --local --remove-section "$branch_section" || true
done < <(git -C /testbed config --local --name-only --get-regexp '^branch\\.' 2>/dev/null | cut -d. -f1-2 | sort -u)
rm -rf /testbed/.git/logs /testbed/.git/refs/remotes /testbed/.git/refs/original /testbed/.git/worktrees
rm -f /testbed/.git/ORIG_HEAD /testbed/.git/FETCH_HEAD /testbed/.git/MERGE_HEAD /testbed/.git/CHERRY_PICK_HEAD
find /testbed/.git/hooks -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true
git -C /testbed reset --mixed "$PREPARED_BASE_SHA" >/dev/null
PREPARED_TREE_DIGEST="$(git -C /testbed ls-files -s -z | sha256sum | cut -d' ' -f1)"
git -C /testbed reflog expire --expire=now --all
git -C /testbed gc --prune=now
test "$(git -C /testbed rev-parse HEAD^{{commit}})" = "$PREPARED_BASE_SHA"
if [[ "$PREPARED_BASE_SHA" != "$SOURCE_BASE_SHA" ]]; then
  test "$(git -C /testbed rev-list --parents -n 1 HEAD | wc -w)" -eq 2
  test "$(git -C /testbed rev-parse HEAD^1)" = "$SOURCE_BASE_SHA"
fi
! git -C /testbed cat-file -e {solution}^{{commit}} 2>/dev/null
test "$(git -C /testbed for-each-ref refs/heads --format='%(refname)')" = refs/heads/r2egym-baseline
mapfile -t COMMITS_AFTER_BASE < <(git -C /testbed rev-list --all --not "$SOURCE_BASE_SHA")
if [[ "$PREPARED_BASE_SHA" = "$SOURCE_BASE_SHA" ]]; then
  test "${{#COMMITS_AFTER_BASE[@]}}" -eq 0
else
  test "${{#COMMITS_AFTER_BASE[@]}}" -eq 1
  test "${{COMMITS_AFTER_BASE[0]}}" = "$PREPARED_BASE_SHA"
fi
while IFS= read -r tag_ref; do
  tag_commit="$(git -C /testbed rev-parse "$tag_ref^{{commit}}")"
  git -C /testbed merge-base --is-ancestor "$tag_commit" "$SOURCE_BASE_SHA"
done < <(git -C /testbed for-each-ref refs/tags --format='%(refname)')
test -z "$(git -C /testbed remote)"
test ! -s /testbed/.git/logs/HEAD
mkdir -p /opt/r2egym
mapfile -t SAFE_VERSION_TAGS < <(cut -f1 /tmp/r2egym-safe-tags)
python - "$PREPARED_BASE_SHA" "$PREPARED_TREE_DIGEST" "$UPSTREAM_INDEX_DIGEST" "$SOURCE_BASE_SHA" "${{SAFE_VERSION_TAGS[@]}}" <<'PY'
import json, sys
from pathlib import Path
Path('/opt/r2egym/r2e-security.json').write_text(json.dumps({{
    'schema_version': 'r2e-prepared-security-v5',
    'source_base_selector': '{commit.old_commit_hash}',
    'source_base_sha': sys.argv[4],
    'preserved_ancestor_tags': sys.argv[5:],
    'prepared_base_sha': sys.argv[1],
    'base_tree_digest': 'sha256:' + sys.argv[2],
    'upstream_base_index_digest': 'sha256:' + sys.argv[3],
    'sanitization_version': '{SANITIZATION_VERSION}',
}}, sort_keys=True) + '\\n')
PY
printf '%s\\n' 'r2e-gym-subset-offline-v2' > /opt/r2egym/dataset-adapter
rm -f /tmp/r2egym-tracked-index /tmp/r2egym-safe-tags
"""

    def verify_image(
        self,
        client: Any,
        image: str,
        context: VerificationContext,
    ):
        return verify_r2e_image(client, image, context)
