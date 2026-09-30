#!/usr/bin/env python3
"""Plan or explicitly install the model-free clip-download environment.

Default prints a plan. --install downloads pinned Python/packages and, when
needed, the official pinned Node archive. --check inspects local tools only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv-media"
PYTHON_VERSION = "3.12.3"
NODE_VERSION = "22.13.0"
NODE_ARCHIVE_ROOT = f"node-v{NODE_VERSION}-linux-x64"
NODE_DIRECTORY = ROOT / ".cache/media-node"
NODE_LOCAL = NODE_DIRECTORY / NODE_ARCHIVE_ROOT / "bin/node"
NODE_BASE = f"https://nodejs.org/dist/v{NODE_VERSION}"
LOCK = ROOT / "requirements-media.lock.txt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def node_version(path):
    return subprocess.check_output([str(path), "--version"], text=True).strip()


def find_node():
    candidates = [NODE_LOCAL]
    system_node = shutil.which("node")
    if system_node:
        candidates.append(Path(system_node))
    for path in candidates:
        if path.is_file() and node_version(path) == f"v{NODE_VERSION}":
            return path.resolve()
    raise FileNotFoundError(f"Node {NODE_VERSION} is not available; run scripts/setup_media.py --install")


def check_environment():
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise FileNotFoundError(f"Install {name} before downloading clips")
    python = ENVIRONMENT / "bin/python"
    if not python.is_file():
        raise FileNotFoundError("Media environment is missing; run scripts/setup_media.py --install")
    requirements = {}
    for line in LOCK.read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            requirements[name.split("[")[0]] = version
    code = """import importlib.metadata,json,sys
assert sys.version_info[:2] == (3,12), sys.version
expected=json.loads(sys.argv[1])
actual={name:importlib.metadata.version(name) for name in expected}
assert actual==expected, {'expected':expected,'actual':actual}
import yt_dlp,yt_dlp_ejs,requests
print(json.dumps({'python':sys.version.split()[0],'packages':actual}))
"""
    details = json.loads(subprocess.check_output([str(python), "-B", "-c", code, json.dumps(requirements)], text=True))
    node = find_node()
    details.update({"node": str(node), "node_version": node_version(node),
                    "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
                    "requirements_sha256": sha256(LOCK), "network": "not used by check"})
    return details


def download_file(url, path):
    with urllib.request.urlopen(url, timeout=120) as response, path.open("xb") as stream:
        shutil.copyfileobj(response, stream)


def install_node():
    try:
        return find_node()
    except FileNotFoundError:
        pass
    destination = NODE_DIRECTORY / NODE_ARCHIVE_ROOT
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing unexpected Node installation: {destination}")
    NODE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive_name = NODE_ARCHIVE_ROOT + ".tar.xz"
    with tempfile.TemporaryDirectory(prefix="download-", dir=NODE_DIRECTORY) as temporary:
        temp = Path(temporary)
        checksums, archive = temp / "SHASUMS256.txt", temp / archive_name
        download_file(f"{NODE_BASE}/SHASUMS256.txt", checksums)
        matches = [line.split()[0] for line in checksums.read_text().splitlines()
                   if len(line.split()) == 2 and line.split()[1] == archive_name]
        if len(matches) != 1:
            raise ValueError("Official Node checksum list does not identify the pinned archive")
        download_file(f"{NODE_BASE}/{archive_name}", archive)
        actual = sha256(archive)
        if actual != matches[0]:
            raise ValueError("Downloaded Node archive failed its official SHA256 check")
        with tarfile.open(archive, "r:xz") as package:
            package.extractall(temp / "unpack", filter="data")
        extracted = temp / "unpack" / NODE_ARCHIVE_ROOT
        if node_version(extracted / "bin/node") != f"v{NODE_VERSION}":
            raise ValueError("Extracted Node version differs from its pinned release")
        extracted.rename(destination)
        record = {"url": f"{NODE_BASE}/{archive_name}", "archive_sha256": actual,
                  "checksum_url": f"{NODE_BASE}/SHASUMS256.txt", "checksum_file_sha256": sha256(checksums),
                  "node_version": NODE_VERSION, "node_binary_sha256": sha256(destination / "bin/node")}
        with (destination / "download-manifest.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return find_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--check", action="store_true", help="Read-only local versions/imports; no network or model loading")
    args = parser.parse_args()
    commands = [["uv", "python", "install", PYTHON_VERSION],
                ["uv", "venv", "--python", PYTHON_VERSION, str(ENVIRONMENT)],
                ["uv", "pip", "sync", "--python", str(ENVIRONMENT / "bin/python"), str(LOCK)]]
    if not (args.install or args.check):
        print(json.dumps({"commands": commands, "python": PYTHON_VERSION, "node": NODE_VERSION,
            "node_installation": "Reuse the exact version on PATH or in project cache; otherwise fetch official Linux x86_64 archive and verify published SHA256",
            "node_archive_url": f"{NODE_BASE}/{NODE_ARCHIVE_ROOT}.tar.xz",
            "requirements": str(LOCK), "system_requirements": ["uv", "ffmpeg", "ffprobe"],
            "network": "Only --install downloads runtimes/packages; never model weights or videos"}, indent=2))
        return
    if args.check:
        print(json.dumps(check_environment(), indent=2))
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("The automated bootstrap is validated for Linux x86_64 / WSL2")
    for name in ("uv", "ffmpeg", "ffprobe"):
        if not shutil.which(name):
            parser.error(f"Install {name} before running setup")
    env = os.environ.copy()
    env.update({"UV_CACHE_DIR": str(ROOT / ".cache/uv"), "UV_PYTHON_INSTALL_DIR": str(ROOT / ".cache/uv-python")})
    subprocess.run(commands[0], env=env, check=True)
    if not (ENVIRONMENT / "pyvenv.cfg").exists():
        if ENVIRONMENT.exists() or ENVIRONMENT.is_symlink():
            raise FileExistsError(f"Refusing to replace {ENVIRONMENT}")
        subprocess.run(commands[1], env=env, check=True)
    version = subprocess.check_output([str(ENVIRONMENT / "bin/python"), "-c", "import platform; print(platform.python_version())"], text=True).strip()
    if version != PYTHON_VERSION:
        raise ValueError(f"Existing media environment uses Python {version}; expected {PYTHON_VERSION}; preserve it before creating a clean environment")
    subprocess.run(commands[2], env=env, check=True)
    install_node()
    record = check_environment()
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    cache = ROOT / ".cache/media-setup"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "setup-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
