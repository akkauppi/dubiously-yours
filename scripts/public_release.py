#!/usr/bin/env python3
"""Audit and copy only allowlisted source files into a clean public directory."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DENIED_PARTS = {"input", "outputs", "projects", "models", "vendor", "sample", "local", "dist", ".cache", ".git", ".aws", ".codex", ".agents"}
PATTERNS = {
    "personal absolute path": re.compile(r"/(?:home|Users)/[^\s/]+/|[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "cloud access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def file_list(root):
    result = []
    for line in (root / "PUBLIC_FILES.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or any(c in name for c in "*?[]\\"):
            raise ValueError(f"Unsafe publication path: {name}")
        if name in result or any(part in DENIED_PARTS or part.startswith(".venv") for part in path.parts):
            raise ValueError(f"Duplicate or private publication path: {name}")
        result.append(name)
    if not result:
        raise ValueError("Publication manifest is empty")
    return result


def ignore_text(names):
    directories = {str(p) for name in names for p in Path(name).parents if str(p) != "."}
    lines = ["# Generated from PUBLIC_FILES.txt by scripts/public_release.py ignore.",
             "# New files remain private until deliberately added to that manifest.", "/*"]
    for directory in sorted(directories, key=lambda p: (p.count("/"), p)):
        lines.extend((f"!/{directory}/", f"/{directory}/*"))
    lines.extend(f"!/{name}" for name in names)
    return "\n".join(lines) + "\n"


def update_ignore(root):
    root = root.resolve()
    destination = root / ".gitignore"
    if destination.is_symlink():
        raise ValueError("Refusing to update a symlinked .gitignore")
    content = ignore_text(file_list(root))
    permissions = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".gitignore-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        if destination.is_symlink():
            raise ValueError("Refusing to update a symlinked .gitignore")
        temporary.chmod(permissions)
        # Replace the directory entry instead of truncating a linked target.
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit(root, *, check_git=True, deny_terms_file=None):
    root = root.resolve()
    names = file_list(root)
    if (root / ".gitignore").read_text(encoding="utf-8") != ignore_text(names):
        raise ValueError(".gitignore does not match PUBLIC_FILES.txt; run public_release.py ignore")
    forbidden = []
    if deny_terms_file:
        forbidden = [line.strip().casefold() for line in Path(deny_terms_file).read_text().splitlines()
                     if line.strip() and not line.startswith("#")]
    records = {}
    for name in names:
        path = root / name
        if any((root / parent).is_symlink() for parent in (Path(name), *Path(name).parents)):
            raise ValueError(f"Publication may not follow symlinks: {name}")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing source file: {name}")
        content = path.read_bytes()
        if len(content) > 1024 * 1024 or b"\0" in content:
            raise ValueError(f"Unexpected binary or large file: {name}")
        text = content.decode("utf-8")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                raise ValueError(f"Review {label} in {name}; matched values are deliberately not printed")
        if any(term in text.casefold() for term in forbidden):
            raise ValueError(f"Private review term found in {name}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "urn:")):
                    continue
                link = (path.parent / target).resolve()
                if not link.is_relative_to(root) or str(link.relative_to(root)) not in names:
                    raise ValueError(f"Markdown link leaves the public bundle: {name} -> {target}")
        records[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if check_git and shutil.which("git"):
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == root:
            tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
            extra = sorted(set(filter(None, tracked)) - set(names))
            if extra:
                raise ValueError(f"Tracked files outside publication manifest: {extra}")
    return records


def build(root, output, *, deny_terms_file=None):
    root, output = root.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace release directory: {output}")
    report = output.with_name(output.name + "-manifest.json")
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"Release report already exists: {report}")
    records = audit(root, deny_terms_file=deny_terms_file)
    output.mkdir(parents=True)
    for name in records:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    copied = audit(output, check_git=False, deny_terms_file=deny_terms_file)
    if copied != records:
        raise RuntimeError("Release copy differs from the audited source")
    # Outside the repository copy, so this local audit isn't accidentally
    # confused with source that belongs in the public Git history.
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"Release report already exists: {report}")
    with report.open("x", encoding="utf-8") as stream:
        json.dump({"files": records, "scope": "Only explicit source allowlist copied; private runtime assets excluded",
                   "limitations": "Pattern checks are not a complete secret or personal-data audit; review before publishing"}, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build", "ignore"))
    parser.add_argument("--output", type=Path, default=ROOT / "dist/dubiously-yours")
    parser.add_argument("--deny-terms-file", type=Path, help="Optional private one-term-per-line extra review list; never copied")
    args = parser.parse_args()
    try:
        if args.command == "ignore":
            update_ignore(ROOT)
            print("Updated .gitignore from the explicit publication manifest")
        elif args.command == "check":
            records = audit(ROOT, deny_terms_file=args.deny_terms_file)
            print(f"Publication checks passed for {len(records)} source files")
        else:
            report = build(ROOT, args.output, deny_terms_file=args.deny_terms_file)
            print(f"Prepared {args.output}\nFile hashes: {report}\nNothing has been uploaded or pushed.")
    except (ValueError, OSError, UnicodeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Publication check failed: {error}\n")


if __name__ == "__main__":
    main()
