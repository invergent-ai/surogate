"""``surogate jackalope`` — launch the jackalope live-training TUI dashboard.

jackalope is a standalone, Node-free executable built separately from the CUDA
wheel (it is pure UI, with no CUDA/C++), so it iterates and updates on its own
cadence. This module resolves that binary, fetches it on first use, and
``os.execv``'s it — replacing the Python process so the dashboard owns the TTY.

Deliberately stdlib-only and free of any CUDA-touching imports: it runs before
the rest of the CLI loads, so ``surogate jackalope`` starts instantly and works
on machines with no GPU.

Update model: binaries are published to a moving ``jackalope-latest`` GitHub
release that the jackalope CI re-publishes whenever the dashboard changes — so
``surogate jackalope --update`` always pulls the newest build without waiting on
a surogate wheel release. Pin a specific build with
``SUROGATE_JACKALOPE_VERSION=<tag>`` (e.g. ``jackalope-v0.2.0``).
"""

import os
import platform
import shutil
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "invergent-ai/surogate"
DEFAULT_TAG = "jackalope-latest"


def _asset_name() -> str | None:
    """Release asset for this OS/arch, or None if unsupported."""
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if arch is None:
        return None
    system = platform.system()
    if system == "Linux":
        return f"jackalope-linux-{arch}"
    if system == "Darwin":
        return f"jackalope-darwin-{arch}"
    if system == "Windows":
        return f"jackalope-windows-{arch}.exe"
    return None


def _release_tag() -> str:
    return os.environ.get("SUROGATE_JACKALOPE_VERSION") or DEFAULT_TAG


def _cache_path() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    name = "jackalope.exe" if platform.system() == "Windows" else "jackalope"
    return base / "surogate" / name


def _candidate_paths() -> list[Path]:
    """Where a jackalope binary might already live, most-specific first."""
    paths: list[Path] = []
    override = os.environ.get("SUROGATE_JACKALOPE_BIN")
    if override:
        paths.append(Path(override))
    # Beside the running `surogate` / the venv's python — where install.sh and
    # the Docker images place it.
    paths.append(Path(sys.argv[0]).resolve().parent / "jackalope")
    paths.append(Path(sys.executable).resolve().parent / "jackalope")
    paths.append(_cache_path())
    on_path = shutil.which("jackalope")
    if on_path:
        paths.append(Path(on_path))
    return paths


def _find_binary() -> Path | None:
    for p in _candidate_paths():
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _download(dest: Path) -> bool:
    asset = _asset_name()
    if asset is None:
        print(f"jackalope: no prebuilt binary for {platform.system()}/{platform.machine()}", file=sys.stderr)
        return False
    url = f"https://github.com/{REPO}/releases/download/{_release_tag()}/{asset}"
    print(f"jackalope: fetching {asset} ({_release_tag()})…", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".jackalope-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted GitHub URL)
            shutil.copyfileobj(resp, out)
        tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(tmp, dest)  # atomic
        return True
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)  # don't leave a partial temp file behind
        print(f"jackalope: download failed: {exc}", file=sys.stderr)
        print(f"  try manually: curl -L {url} -o {dest} && chmod +x {dest}", file=sys.stderr)
        return False


def maybe_exec_jackalope() -> None:
    """If invoked as ``surogate jackalope ...``, launch the dashboard and never return.

    Returns immediately (a no-op) for every other command so normal CLI dispatch
    proceeds untouched.
    """
    argv = sys.argv
    if len(argv) < 2 or argv[1] != "jackalope":
        return
    rest = argv[2:]

    # `surogate jackalope --update` (or `update`/`upgrade`) refreshes the binary.
    if rest and rest[0] in ("--update", "update", "upgrade"):
        ok = _download(_cache_path())
        print("jackalope: updated" if ok else "jackalope: update failed", file=sys.stderr)
        sys.exit(0 if ok else 1)

    binary = _find_binary()
    if binary is None:
        if os.environ.get("SUROGATE_NO_DOWNLOAD"):
            print("jackalope: not installed and SUROGATE_NO_DOWNLOAD is set", file=sys.stderr)
            sys.exit(1)
        if not _download(_cache_path()):
            sys.exit(1)
        binary = _cache_path()

    os.execv(str(binary), [str(binary), *rest])  # replace this process; inherits the TTY
