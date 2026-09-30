#!/usr/bin/env python3
"""Plan, check or explicitly install the pinned local CPU speech environment.

No arguments prints a read-only plan. --check inspects installed distributions
and hashes local assets without importing neural models. --install creates a
new environment (or verifies an already matching one). --models explicitly
downloads all three pinned asset sets; combine it with --install for first use.

The version lock reproduces the working Linux/WSL2 x86_64 package set; it is not
a hash-locked wheel archive. A fresh installation is a separate verification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

if __package__:
    from .fetch_finnish_models import (ASR_FILES, ASR_REPO, ASR_REVISION, FILES,
                                      REPO, REVISION, verify_manifest, verify_tokenizer_assets)
else:
    from fetch_finnish_models import (ASR_FILES, ASR_REPO, ASR_REVISION, FILES,
                                     REPO, REVISION, verify_manifest, verify_tokenizer_assets)

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.11.15"
UV_VERSION = "0.11.7"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
LOCK = ROOT / "requirements-audio.lock.txt"


def read_lock(path=LOCK):
    pins = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9.+!-]+", line):
            raise ValueError(f"Expected an exact package==version pin on lock line {number}")
        name, version = line.split("==")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in pins:
            raise ValueError(f"Duplicate package pin: {name}")
        pins[canonical] = version
    if pins.get("torch") != "2.6.0+cpu" or pins.get("torchaudio") != "2.6.0+cpu":
        raise ValueError("The audio lock must retain the verified CPU Torch pair")
    return pins


def environment_report(python, pins):
    if not python.is_file():
        return {"ready": False, "errors": [f"Environment interpreter missing: {python}"]}
    # Metadata inspection avoids model imports, JIT caches, downloads and inference.
    code = r'''
import importlib.metadata as md, json, platform, sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
pins = json.loads(sys.argv[1])
installed = {canonicalize_name(d.metadata['Name']): d for d in md.distributions() if d.metadata['Name']}
versions = {name: d.version for name, d in installed.items()}
errors = [f"{name}: expected {version}, found {versions.get(name, 'missing')}"
          for name, version in pins.items() if versions.get(name) != version]
conflicts = []
for name, distribution in installed.items():
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
            continue
        actual = versions.get(canonicalize_name(requirement.name))
        if actual is None or (requirement.specifier and not requirement.specifier.contains(actual, prereleases=True)):
            conflicts.append(f"{name} requires {requirement}; found {actual or 'missing'}")
print(json.dumps({'python': platform.python_version(), 'platform': sys.platform,
                  'machine': platform.machine(), 'installed_packages': len(versions),
                  'pinned_packages': len(pins), 'errors': errors,
                  'dependency_conflicts': conflicts,
                  'extra_packages': sorted(set(versions) - set(pins))}))
'''
    result = subprocess.run([str(python), "-I", "-B", "-c", code, json.dumps(pins)],
                            capture_output=True, text=True)
    if result.returncode:
        return {"ready": False, "errors": ["Unable to inspect environment metadata", result.stderr.strip()]}
    report = json.loads(result.stdout)
    if report["python"] != PYTHON_VERSION:
        report["errors"].append(f"Expected Python {PYTHON_VERSION}; found {report['python']}")
    if report["platform"] != "linux" or report["machine"] not in ("x86_64", "AMD64"):
        report["errors"].append("This pinned environment is verified only on Linux/WSL2 x86_64")
    report["ready"] = not report["errors"] and not report["dependency_conflicts"]
    return report


def asset_report(models_dir, cache_dir):
    checks = {}
    for name, repo, revision, files in (
            ("chatterbox-multilingual-v2", REPO, REVISION, FILES),
            ("faster-whisper-small", ASR_REPO, ASR_REVISION, ASR_FILES)):
        try:
            verify_manifest(models_dir / name, repo=repo, revision=revision, required=files)
            checks[name] = {"ready": True, "repository": repo, "revision": revision}
        except (OSError, ValueError, TypeError) as error:
            checks[name] = {"ready": False, "error": str(error)}
    try:
        verify_tokenizer_assets(cache_dir, models_dir / "chatterbox-multilingual-v2")
        checks["tokenizer"] = {"ready": True, "archive_and_extracted_hashes_verified": True}
    except (OSError, ValueError, TypeError) as error:
        checks["tokenizer"] = {"ready": False, "error": str(error)}
    return {"ready": all(item["ready"] for item in checks.values()), "checks": checks}


def commands(venv, models_dir, cache_dir, pins):
    python = venv / "bin/python"
    uv = ["uv", "--no-config", "--cache-dir", str(cache_dir / "uv")]
    install = [uv + ["venv", "--no-project", "--managed-python", "--python", PYTHON_VERSION, str(venv)],
               uv + ["pip", "install", "--python", str(python), "--default-index", CPU_INDEX,
                     f"torch=={pins['torch']}", f"torchaudio=={pins['torchaudio']}"],
               uv + ["pip", "install", "--python", str(python), "--default-index", PYPI_INDEX,
                     "--requirement", str(LOCK)],
               uv + ["pip", "check", "--python", str(python), "--offline"]]
    fetch = [str(python), str(ROOT / "scripts/fetch_finnish_models.py"),
             "--models-dir", str(models_dir), "--cache-dir", str(cache_dir)]
    return install, [fetch, fetch + ["--tokenizer-assets"], fetch + ["--asr"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only package/dependency and asset-hash checks")
    mode.add_argument("--install", action="store_true", help="Explicitly install pinned Python/packages using uv; may download")
    parser.add_argument("--models", action="store_true", help="Explicitly fetch synthesis, tokenizer and ASR assets")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-audio")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    if args.check and args.models:
        parser.error("--check is read-only and cannot be combined with --models")
    args.venv, args.models_dir, args.cache_dir = (p.resolve() for p in (args.venv, args.models_dir, args.cache_dir))
    pins = read_lock()
    install, fetch = commands(args.venv, args.models_dir, args.cache_dir, pins)
    python = args.venv / "bin/python"
    if not any((args.check, args.install, args.models)):
        print(json.dumps({"mode": "plan_only", "writes_or_network": False,
                          "target": "Linux/WSL2 x86_64 CPU", "python": PYTHON_VERSION,
                          "uv": UV_VERSION, "lock": str(LOCK), "packages": len(pins),
                          "install_commands": install, "model_commands": fetch,
                          "note": "Run --install and optionally --models explicitly. Version pins are not wheel hashes."}, indent=2))
        return 0
    if args.install:
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            raise ValueError("Installation targets Linux/WSL2 x86_64; other platforms are unverified")
        if args.venv.exists():
            installed = environment_report(python, pins)
            if not installed["ready"]:
                raise ValueError("Existing environment differs; use --check or select a new --venv. It will not be replaced.")
            print("Existing pinned environment verified; no package changes needed.", flush=True)
        else:
            if not shutil.which("uv"):
                raise FileNotFoundError(f"Install uv {UV_VERSION} before requesting --install")
            version = subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True).stdout.split()
            if len(version) < 2 or version[1] != UV_VERSION:
                raise ValueError(f"Use uv {UV_VERSION} to reproduce this setup")
            env = os.environ.copy()
            for key in ("UV_INDEX", "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS", "UV_INSECURE_HOST", "UV_VENV_CLEAR"):
                env.pop(key, None)
            for command in install:
                subprocess.run(command, check=True, cwd=ROOT, env=env)
    environment = environment_report(python, pins)
    if args.models:
        if not environment["ready"]:
            print(json.dumps({"environment": environment}, indent=2))
            raise ValueError("Verify/install the pinned audio environment before fetching models")
        for command in fetch:
            subprocess.run(command, check=True, cwd=ROOT)
    assets = asset_report(args.models_dir, args.cache_dir)
    ready = environment["ready"] and assets["ready"]
    print(json.dumps({"ready": ready, "mode": "check" if args.check else "setup",
                      "environment": environment, "assets": assets,
                      "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                      "note": "No neural inference or voice-quality assessment is performed."}, indent=2))
    # Installing packages alone is successful even when assets have not been requested.
    return 0 if (ready or (args.install and not args.models and environment["ready"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
