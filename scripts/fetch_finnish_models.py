#!/usr/bin/env python3
"""Explicitly download pinned multilingual speech models and offline tokenizer assets.

Only this setup command uses the network. Importing it or requesting --help does
not import model libraries or download anything.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = "ResembleAI/chatterbox"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
FILES = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
         "grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json", "README.md"]
ASR_REPO = "Systran/faster-whisper-small"
ASR_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
ASR_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt", "README.md"]
TOKENIZER_URL = "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip"
TOKENIZER_FILES = {
    "spacy_ontonotes.zip": (34567143, "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe"),
    "spacy_ontonotes/features.msgpack": (22685181, "fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18"),
    "spacy_ontonotes/weights.npz": (37508754, "5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e"),
}


def file_record(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": Path(path).stat().st_size, "sha256": digest}


def verify_manifest(directory, *, repo=None, revision=None, required=()):
    """Read-only verification; never bless changed files with replacement hashes."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "download-manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest.get("repo"), str) or not manifest["repo"]
            or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", ""))):
        raise ValueError("Model manifest must identify a repository and full commit SHA")
    if repo is not None and manifest["repo"] != repo:
        raise ValueError("Model manifest identifies an unexpected repository")
    if revision is not None and manifest["revision"] != revision:
        raise ValueError("Model manifest identifies an unexpected revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(name not in files for name in required):
        raise ValueError("Model manifest lacks required files")
    for name, saved in files.items():
        path = directory / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Unsafe model manifest path: {name}")
        if (not isinstance(saved, dict) or type(saved.get("bytes")) is not int
                or not re.fullmatch(r"[a-f0-9]{64}", saved.get("sha256", ""))):
            raise ValueError(f"Model manifest lacks size/SHA256: {name}")
        if file_record(path) != {"bytes": saved["bytes"], "sha256": saved["sha256"]}:
            raise ValueError(f"Model file differs from its manifest: {name}")
    return manifest


def verify_tokenizer_assets(cache_dir, model_dir=None):
    """The upstream PKUSEG downloader ignores HF offline flags: verify first."""
    directory = Path(cache_dir) / "pkuseg"
    records = {}
    for name, (size, digest) in TOKENIZER_FILES.items():
        actual = file_record(directory / name)
        if actual != {"bytes": size, "sha256": digest}:
            raise ValueError(f"Tokenizer asset differs from its pinned checksum: {name}")
        records[name] = actual
    if model_dir is not None:
        model_dir = Path(model_dir)
        hub = model_dir / "models--ResembleAI--chatterbox"
        if (hub / "refs/main").read_text().strip() != REVISION:
            raise ValueError("Cangjie cache alias does not identify the pinned model revision")
        if file_record(hub / "snapshots" / REVISION / "Cangjie5_TC.json") != file_record(model_dir / "Cangjie5_TC.json"):
            raise ValueError("Cangjie cache and pinned model mapping differ")
    return records


def write_manifest(path, record):
    # Setup may enrich an existing compatible manifest; replace atomically.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".manifest-", delete=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def fetch_tokenizer_assets(cache_dir, models_dir):
    directory = cache_dir / "pkuseg"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "spacy_ontonotes.zip"
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            candidate = Path(temporary) / archive.name
            with urlopen(TOKENIZER_URL, timeout=60) as response, candidate.open("xb") as stream:
                shutil.copyfileobj(response, stream)
            size, digest = TOKENIZER_FILES[archive.name]
            if file_record(candidate) != {"bytes": size, "sha256": digest}:
                raise ValueError("Downloaded tokenizer archive checksum mismatch")
            with candidate.open("rb") as source, archive.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    size, digest = TOKENIZER_FILES[archive.name]
    if file_record(archive) != {"bytes": size, "sha256": digest}:
        raise ValueError("Existing tokenizer archive checksum mismatch; refusing to replace")
    with zipfile.ZipFile(archive) as bundle:
        for member in ("features.msgpack", "weights.npz"):
            path = directory / "spacy_ontonotes" / member
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    verify_tokenizer_assets(cache_dir)
    from huggingface_hub import hf_hub_download
    destination = models_dir / "chatterbox-multilingual-v2"
    verify_manifest(destination, repo=REPO, revision=REVISION, required=FILES)
    hf_hub_download(REPO, filename="Cangjie5_TC.json", revision=REVISION, cache_dir=destination)
    # Upstream omits a revision here; bind its private main alias to our pin.
    alias = destination / "models--ResembleAI--chatterbox/refs/main"
    if alias.exists() and alias.read_text().strip() != REVISION:
        raise ValueError("Tokenizer cache main already identifies another snapshot")
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists():
        with alias.open("x") as stream:
            stream.write(REVISION)
    records = verify_tokenizer_assets(cache_dir, destination)
    write_manifest(directory / "download-manifest.json", {
        "source": TOKENIZER_URL, "archive_sha256": digest, "bytes": size,
        "purpose": "Auxiliary tokenizer initialization for offline multilingual synthesis",
        "cangjie_revision": REVISION, "files": records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--asr", action="store_true", help="Fetch multilingual Whisper small instead of Chatterbox")
    selection.add_argument("--tokenizer-assets", action="store_true", help="Prepare auxiliary tokenizer assets for offline model initialization")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    args.models_dir, args.cache_dir = args.models_dir.resolve(), args.cache_dir.resolve()
    os.environ.update({"HF_HOME": str(args.cache_dir / "huggingface"),
                       "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_XET": "1"})
    if args.tokenizer_assets:
        fetch_tokenizer_assets(args.cache_dir, args.models_dir)
        print("Offline tokenizer assets ready", flush=True)
        return
    repo, revision, files = REPO, REVISION, FILES
    variant = "multilingual-v2"
    directory = "chatterbox-multilingual-v2"
    if args.asr:
        repo, revision, files = ASR_REPO, ASR_REVISION, ASR_FILES
        variant = "multilingual-small"
        directory = "faster-whisper-small"
    destination = args.models_dir / directory
    manifest = destination / "download-manifest.json"
    if manifest.exists():
        verify_manifest(destination, repo=repo, revision=revision, required=files)
        print(f"Already verified: {destination}", flush=True)
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=destination,
                      allow_patterns=files, max_workers=2)
    records = {}
    for name in files:
        path = destination / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records[name] = {"bytes": path.stat().st_size, "sha256": digest}
    write_manifest(manifest, {"repo": repo, "revision": revision,
                              "variant": variant, "files": records})
    print(f"Ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
