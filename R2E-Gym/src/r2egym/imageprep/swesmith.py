"""SWE-smith-specific preparation scripts and repository hooks."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Protocol

from .manifest import PreparedImageSpec, validate_instance_id


PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
GIT_PROXY = "https://hk.gh-proxy.com/https://github.com/"


@dataclass(frozen=True)
class ImageBuildContext:
    spec: PreparedImageSpec
    test_command: str


@dataclass(frozen=True)
class VerificationContext:
    spec: PreparedImageSpec
    test_command: str
    tasks: tuple[dict, ...] = ()
    task_workers: int = 12
    image_only: bool = False


class PreparationHook(Protocol):
    def prepare_lines(self, context: ImageBuildContext) -> list[str]: ...

    def verification_command(self, context: VerificationContext) -> str | None: ...


class DefaultPythonHook:
    def prepare_lines(self, context: ImageBuildContext) -> list[str]:
        return [
            'PYTHON_BIN="/opt/miniconda3/envs/testbed/bin/python"',
            'test -x "$PYTHON_BIN"',
            'export PATH="/opt/miniconda3/envs/testbed/bin:/root/.local/bin:$PATH"',
            f'export PIP_INDEX_URL={shlex.quote(PIP_INDEX)}',
            f'export UV_INDEX_URL={shlex.quote(PIP_INDEX)}',
            f'export UV_DEFAULT_INDEX={shlex.quote(PIP_INDEX)}',
            'mkdir -p /opt/r2egym/wheelhouse /opt/r2egym/cache/pip /opt/r2egym/cache/uv',
            'export PIP_CACHE_DIR=/opt/r2egym/cache/pip',
            'export UV_CACHE_DIR=/opt/r2egym/cache/uv',
            '"$PYTHON_BIN" -m pip install chardet',
            'if printf "%s\\n" ' + shlex.quote(context.test_command)
            + ' | grep -Eq "(^|[[:space:]/])pytest([[:space:]]|$)"; then '
            '"$PYTHON_BIN" -c "import pytest" 2>/dev/null || '
            '"$PYTHON_BIN" -m pip install pytest; fi',
            '"$PYTHON_BIN" - <<\'PY\' >/tmp/r2egym-build-requires.txt\n'
            'from importlib import metadata\n'
            'from pathlib import Path\n'
            'from packaging.requirements import Requirement\n'
            'from packaging.version import Version\n'
            '\n'
            'def is_satisfied(value: str) -> bool:\n'
            '    requirement = Requirement(value)\n'
            '    if requirement.marker and not requirement.marker.evaluate():\n'
            '        return True\n'
            '    try:\n'
            '        installed = Version(metadata.version(requirement.name))\n'
            '    except metadata.PackageNotFoundError:\n'
            '        return False\n'
            '    return not requirement.specifier or installed in requirement.specifier\n'
            '\n'
            'path = Path("/testbed/pyproject.toml")\n'
            'if path.is_file():\n'
            '    try:\n'
            '        import tomllib\n'
            '    except ImportError:\n'
            '        import tomli as tomllib\n'
            '    data = tomllib.loads(path.read_text(encoding="utf-8"))\n'
            '    for requirement in data.get("build-system", {}).get("requires", []):\n'
            '        if (isinstance(requirement, str) and requirement.strip()\n'
            '                and not is_satisfied(requirement.strip())):\n'
            '            print(requirement.strip())\n'
            'PY',
            'if test -s /tmp/r2egym-build-requires.txt; then '
            'timeout 600 "$PYTHON_BIN" -m pip download --dest /opt/r2egym/wheelhouse '
            '--timeout 120 --retries 3 -r /tmp/r2egym-build-requires.txt; fi',
        ]

    def verification_command(self, context: VerificationContext) -> str | None:
        return (
            'export PATH="/opt/miniconda3/envs/testbed/bin:/root/.local/bin:$PATH"; '
            'export PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/r2egym/wheelhouse '
            'UV_OFFLINE=1 UV_CACHE_DIR=/opt/r2egym/cache/uv; '
            'python -c "import chardet"'
        )


class UvProjectHook:
    def prepare_lines(self, context: ImageBuildContext) -> list[str]:
        if "uv run" not in context.test_command:
            return []
        return [
            'UV_BIN="$(command -v uv || true)"',
            'test -n "$UV_BIN"',
            '"$UV_BIN" run --no-progress python -c "import sys"',
            'UV_OFFLINE=1 "$UV_BIN" run --offline --no-progress python -c "import sys"',
        ]

    def verification_command(self, context: VerificationContext) -> str | None:
        if "uv run" not in context.test_command:
            return None
        return (
            'export PATH="/opt/miniconda3/envs/testbed/bin:/root/.local/bin:$PATH"; '
            'UV_OFFLINE=1 UV_CACHE_DIR=/opt/r2egym/cache/uv '
            'uv run --offline --no-progress python -c "import sys"'
        )


class PydicomDataHook:
    def prepare_lines(self, context: ImageBuildContext) -> list[str]:
        if "pydicom__pydicom" not in context.spec.repo.lower():
            return []
        return [
            'if "$PYTHON_BIN" -c \'import requests\' 2>/dev/null; then '
            'echo "requests already available"; else '
            '"$PYTHON_BIN" -m pip install requests; fi',
            '"$PYTHON_BIN" - <<\'PY\'\n'
            'import hashlib\n'
            'import json\n'
            'import os\n'
            'import shutil\n'
            'import sysconfig\n'
            'from pathlib import Path\n'
            '\n'
            'from pydicom.data import (\n'
            '    get_charset_files,\n'
            '    get_palette_files,\n'
            '    get_testdata_files,\n'
            ')\n'
            'from pydicom.data.data_manager import fetch_data_files\n'
            'from pydicom.data.download import get_data_dir, get_url_map\n'
            '\n'
            '# Fetch every external file declared by this exact pydicom checkout.\n'
            '# Do not install the optional pydicom-data provider: it changes the\n'
            '# pytest parameter IDs and is therefore not environment-transparent.\n'
            'fetch_data_files()\n'
            'collections = {\n'
            '    "testdata": get_testdata_files(),\n'
            '    "palettes": get_palette_files(),\n'
            '    "charsets": get_charset_files(),\n'
            '}\n'
            'cache = Path(get_data_dir())\n'
            'online = [cache / name for name in sorted(get_url_map())]\n'
            'missing = [str(path) for path in online if not path.is_file()]\n'
            'if missing:\n'
            '    raise RuntimeError(f"pydicom data prefetch incomplete: {missing}")\n'
            'mirror = Path("/opt/r2egym/pydicom-data-mirror")\n'
            'mirror.mkdir(parents=True, exist_ok=True)\n'
            'for path in online:\n'
            '    target = mirror / path.name\n'
            '    if target.exists():\n'
            '        target.unlink()\n'
            '    try:\n'
            '        os.link(path, target)\n'
            '    except OSError:\n'
            '        shutil.copy2(path, target)\n'
            'purelib = Path(sysconfig.get_paths()["purelib"])\n'
            'shim = purelib / "r2egym_pydicom_offline.py"\n'
            'shim.write_text(\n'
            '    "from pathlib import Path\\n"\n'
            '    "import shutil\\n"\n'
            '    "try:\\n"\n'
            '    "    import pydicom.data.download as _download\\n"\n'
            '    "    _mirror = Path(\\\"/opt/r2egym/pydicom-data-mirror\\\")\\n"\n'
            '    "    def _restore(url, fpath):\\n"\n'
            '    "        source = _mirror / Path(str(url).split(\\\"?\\\", 1)[0]).name\\n"\n'
            '    "        if not source.is_file():\\n"\n'
            '    "            raise RuntimeError(f\\\"offline pydicom data unavailable: {source.name}\\\")\\n"\n'
            '    "        shutil.copyfile(source, fpath)\\n"\n'
            '    "    _download.download_with_progress = _restore\\n"\n'
            '    "except ImportError:\\n"\n'
            '    "    pass\\n"\n'
            ')\n'
            '(purelib / "r2egym_pydicom_offline.pth").write_text("import r2egym_pydicom_offline\\n")\n'
            'paths = {Path(value).resolve() for values in collections.values() for value in values}\n'
            'paths.update(path.resolve() for path in online)\n'
            'files = []\n'
            'for path in sorted(paths, key=str):\n'
            '    if not path.is_file():\n'
            '        continue\n'
            '    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n'
            '    files.append({"path": str(path), "size": path.stat().st_size, "sha256": digest})\n'
            'manifest = {\n'
            '    "schema_version": "r2egym-pydicom-data-v1",\n'
            '    "online_file_count": len(online),\n'
            '    "mirror_file_count": len(list(mirror.iterdir())),\n'
            '    "api_counts": {name: len(values) for name, values in collections.items()},\n'
            '    "files": files,\n'
            '}\n'
            'target = Path("/opt/r2egym/pydicom-data-manifest.json")\n'
            'target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")\n'
            'print(f"prefetched {len(files)} pydicom data files ({len(online)} online cache files)")\n'
            'PY',
        ]

    def verification_command(self, context: VerificationContext) -> str | None:
        if "pydicom__pydicom" not in context.spec.repo.lower():
            return None
        return (
            "/opt/miniconda3/envs/testbed/bin/python - <<'PY'\n"
            "import hashlib\n"
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "import pydicom.data.download as download\n"
            "from pydicom.data import get_charset_files, get_palette_files, get_testdata_files\n"
            "from pydicom.data.download import get_data_dir, get_url_map\n"
            "\n"
            "manifest_path = Path('/opt/r2egym/pydicom-data-manifest.json')\n"
            "manifest = json.loads(manifest_path.read_text())\n"
            "assert manifest['schema_version'] == 'r2egym-pydicom-data-v1'\n"
            "assert manifest['files'], 'pydicom data manifest is empty'\n"
            "for record in manifest['files']:\n"
            "    path = Path(record['path'])\n"
            "    assert path.is_file(), f'missing pydicom data file: {path}'\n"
            "    assert path.stat().st_size == record['size'], f'size mismatch: {path}'\n"
            "    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], f'hash mismatch: {path}'\n"
            "cache = Path(get_data_dir())\n"
            "missing = [name for name in get_url_map() if not (cache / name).is_file()]\n"
            "assert not missing, f'missing pydicom online data cache: {missing}'\n"
            "sample = sorted(get_url_map())[0]\n"
            "sample_path = cache / sample\n"
            "expected_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()\n"
            "sample_path.unlink()\n"
            "restored = download.data_path_with_download(sample)\n"
            "assert restored == sample_path.resolve()\n"
            "assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == expected_hash\n"
            "collections = {\n"
            "    'testdata': get_testdata_files(),\n"
            "    'palettes': get_palette_files(),\n"
            "    'charsets': get_charset_files(),\n"
            "}\n"
            "for name, expected in manifest['api_counts'].items():\n"
            "    assert len(collections[name]) == expected, f'pydicom {name} count mismatch'\n"
            "print(f\"verified {len(manifest['files'])} pydicom data files without network\")\n"
            "PY"
        )


class GoModuleHook:
    def prepare_lines(self, context: ImageBuildContext) -> list[str]:
        return [
            'GO_BIN="$(command -v go || true)"',
            'if test -z "$GO_BIN" && test -x /usr/local/go/bin/go; then GO_BIN=/usr/local/go/bin/go; fi',
            'test -x "$GO_BIN"',
            'ln -sf "$GO_BIN" /usr/local/bin/go',
            'export PATH="/usr/local/go/bin:/usr/local/bin:$PATH"',
            'export GOPROXY="https://goproxy.cn,direct"',
            # Populate the complete package and test dependency graph while
            # networking is available. Any go.mod/go.sum edits are discarded
            # when the preparation script resets back to BASELINE.
            'if test -f go.mod; then GOWORK=off go mod download all; '
            'GOWORK=off go list -mod=mod -deps -test ./... >/dev/null; fi',
        ]

    def verification_command(self, context: VerificationContext) -> str | None:
        return (
            'export PATH="/usr/local/go/bin:/usr/local/bin:$PATH" '
            'GOPROXY=off GOFLAGS=-mod=mod GOWORK=off; '
            'go version; if test -f go.mod; then '
            'go list -deps -test ./... >/dev/null; '
            'git checkout -- go.mod go.sum 2>/dev/null || true; fi'
        )


def hooks_for(context: ImageBuildContext) -> list[PreparationHook]:
    if context.spec.profile == "go":
        return [GoModuleHook()]
    if context.spec.profile == "python":
        return [DefaultPythonHook(), UvProjectHook(), PydicomDataHook()]
    # Other ecosystems are already installed by their SWE-smith source image.
    # Their actual offline test command is exercised by the strict task audit.
    return []


def create_prepare_script(context: ImageBuildContext) -> str:
    instances = [validate_instance_id(value) for value in context.spec.instance_ids]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /testbed",
        'git config --global --add safe.directory /testbed',
        'BASELINE="$(git rev-parse HEAD)"',
        "mapfile -t R2EGYM_INSTANCES </opt/r2egym/instances.txt",
        "fetch_refs() {",
        "  local attempt",
        "  for attempt in 1 2 3 4 5 6; do",
        f'    if git -c url.{shlex.quote(GIT_PROXY)}.insteadOf=https://github.com/ fetch --force "$@"; then',
        "      return 0",
        "    fi",
        '    printf "git fetch failed; retry %s/6\\n" "$attempt" >&2',
        "    sleep $((attempt * attempt))",
        "  done",
        "  return 1",
        "}",
        "for ((offset=0; offset<${#R2EGYM_INSTANCES[@]}; offset+=100)); do",
        "  REFSPECS=()",
        "  for instance in \"${R2EGYM_INSTANCES[@]:offset:100}\"; do",
        "    test -n \"$instance\" || continue",
        '    REFSPECS+=("+refs/heads/$instance:refs/r2egym/instances/$instance")',
        "  done",
        '  fetch_refs origin "${REFSPECS[@]}"',
        "done",
        "NEEDS_DEEPEN=0",
        "for instance in \"${R2EGYM_INSTANCES[@]}\"; do",
        "  test -n \"$instance\" || continue",
        '  if ! git rev-parse --verify "refs/r2egym/instances/$instance~2^{commit}" >/dev/null 2>&1; then',
        "    NEEDS_DEEPEN=1",
        "    break",
        "  fi",
        "done",
        'if test "$NEEDS_DEEPEN" = 1; then',
        "  for ((offset=0; offset<${#R2EGYM_INSTANCES[@]}; offset+=100)); do",
        "    REFSPECS=()",
        "    for instance in \"${R2EGYM_INSTANCES[@]:offset:100}\"; do",
        "      test -n \"$instance\" || continue",
        '      REFSPECS+=("+refs/heads/$instance:refs/r2egym/instances/$instance")',
        "    done",
        '    fetch_refs --deepen=4 origin "${REFSPECS[@]}"',
        "  done",
        "fi",
        # A small number of published SWE-smith branches contain only the Bug
        # Patch commit and omit the no-op/F2P-removal wrapper. Normalize those
        # branches internally, but only when the original tip is explicitly a
        # Bug Patch with a valid parent. This preserves the exact bug tree.
        "for instance in \"${R2EGYM_INSTANCES[@]}\"; do",
        "  test -n \"$instance\" || continue",
        '  ref="refs/r2egym/instances/$instance"',
        '  if ! git rev-parse --verify "$ref~2^{commit}" >/dev/null 2>&1; then',
        '    git rev-parse --verify "$ref~1^{commit}" >/dev/null',
        '    test "$(git log -1 --format=%s "$ref")" = "Bug Patch"',
        '    tree="$(git rev-parse "$ref^{tree}")"',
        '    wrapper="$(printf "%s\\n" "Normalize SWE-smith task wrapper" | '
        'git -c user.name=R2EGym -c user.email=r2egym@localhost '
        'commit-tree "$tree" -p "$ref")"',
        '    git update-ref "$ref" "$wrapper"',
        '    printf "normalized one-commit task ref: %s\\n" "$instance"',
        "  fi",
        "done",
        "for instance in \"${R2EGYM_INSTANCES[@]}\"; do",
        "  test -n \"$instance\" || continue",
        '  git rev-parse --verify "refs/r2egym/instances/$instance^{commit}" >/dev/null',
        '  git rev-parse --verify "refs/r2egym/instances/$instance~1^{commit}" >/dev/null',
        '  git rev-parse --verify "refs/r2egym/instances/$instance~2^{commit}" >/dev/null',
        "done",
        'git checkout --detach "$BASELINE"',
        'git reset --hard "$BASELINE"',
        "git clean -fdq",
    ]
    for hook in hooks_for(context):
        lines.extend(hook.prepare_lines(context))
    if context.spec.profile == "go":
        go_environment = ['export GOPROXY=off']
        if "denisenkom__go-mssqldb" in context.spec.repo:
            go_environment.extend(['export GOFLAGS=-mod=mod', 'export GOWORK=off'])
        quoted_environment = " ".join(shlex.quote(value) for value in go_environment)
        lines.append(
            f'printf "%s\\n" {quoted_environment} >/etc/profile.d/r2egym-offline.sh'
        )
    elif context.spec.profile == "python":
        lines.extend(
            [
                "cat >/etc/profile.d/r2egym-offline.sh <<'EOF'",
                "export PIP_NO_INDEX=1",
                "export PIP_FIND_LINKS=/opt/r2egym/wheelhouse",
                "export UV_OFFLINE=1",
                "export UV_CACHE_DIR=/opt/r2egym/cache/uv",
                "EOF",
            ]
        )
    lines.extend(
        [
            'git checkout --detach "$BASELINE"',
            'git reset --hard "$BASELINE"',
            "git clean -fdq",
            "rm -f /tmp/r2egym-build-requires.txt",
        ]
    )
    return "\n".join(lines) + "\n"
