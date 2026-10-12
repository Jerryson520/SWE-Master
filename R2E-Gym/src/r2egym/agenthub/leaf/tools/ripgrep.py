"""Resolve a pinned static ripgrep binary for repository tools."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


RG_VERSION = "15.1.0"
RG_ASSET = f"ripgrep-{RG_VERSION}-x86_64-unknown-linux-musl.tar.gz"
RG_SHA256 = "1c9297be4a084eea7ecaedf93eb03d058d6faae29bbc57ecdaf5063921491599"
DEFAULT_RG_URL = f"https://github.com/BurntSushi/ripgrep/releases/download/{RG_VERSION}/{RG_ASSET}"


class RipgrepUnavailable(RuntimeError):
    pass


def _cache_path() -> Path:
    root = Path(os.environ.get("R2EGYM_CACHE_DIR", Path.home() / ".cache" / "r2egym"))
    return root / "ripgrep" / RG_VERSION / "rg"


def _is_runnable(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and f"ripgrep {RG_VERSION}" in result.stdout


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
        archive = Path(temp_dir) / RG_ASSET
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != RG_SHA256:
            raise RipgrepUnavailable(f"ripgrep archive checksum mismatch from {url}")
        with tarfile.open(archive, "r:gz") as bundle:
            member = next((item for item in bundle.getmembers() if item.isfile() and Path(item.name).name == "rg"), None)
            if member is None:
                raise RipgrepUnavailable(f"{url} does not contain an rg binary")
            source = bundle.extractfile(member)
            if source is None:
                raise RipgrepUnavailable(f"could not extract rg from {url}")
            staged = Path(temp_dir) / "rg"
            with staged.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        staged.chmod(0o755)
        os.replace(staged, destination)


def resolve_host_rg(*, download: bool = True) -> Path:
    explicit = os.environ.get("R2EGYM_RG_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if _is_runnable(path):
            return path
        raise RipgrepUnavailable(f"R2EGYM_RG_PATH={explicit} is not ripgrep {RG_VERSION}")

    cached = _cache_path()
    if _is_runnable(cached):
        return cached
    if not download:
        raise RipgrepUnavailable(f"no cached ripgrep at {cached}")

    lock_path = cached.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass
        if not _is_runnable(cached):
            try:
                _download(os.environ.get("R2EGYM_RG_URL", DEFAULT_RG_URL), cached)
            except (OSError, tarfile.TarError) as exc:
                raise RipgrepUnavailable(f"could not prepare ripgrep: {exc}") from exc
    if not _is_runnable(cached):
        raise RipgrepUnavailable(f"cached ripgrep at {cached} is not runnable")
    return cached
