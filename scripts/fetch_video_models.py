#!/usr/bin/env python3
"""Fetch the public weights needed for MuseTalk 1.5 inference; record revisions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/musetalk"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


MODELS = (
    ("TMElyralab/MuseTalk", ".", "2bcb936e2fddb4d86db4c62fd45b387d0c061571", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]),
    ("stabilityai/sd-vae-ft-mse", "sd-vae", "31f26fdeee1355a5c34592e401dd41e45d25a493", ["config.json", "diffusion_pytorch_model.bin"]),
    ("openai/whisper-tiny", "whisper", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ("yzd-v/DWPose", "dwpose", "1a7144101628d69ee7a3768d1ee3a094070dc388", ["dw-ll_ucoco_384.pth"]),
)
URLS = (
    ("https://drive.usercontent.google.com/download?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download&confirm=t", "face-parse-bisent/79999_iter.pth"),
    ("https://download.pytorch.org/models/resnet18-5c106cde.pth", "face-parse-bisent/resnet18-5c106cde.pth"),
    ("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", "s3fd/s3fd-619a316812.pth"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# SHA256 values from the pinned, locally validated public model files. These
# are independent of mutable download manifests and cover non-HF URLs too.
EXPECTED_HASHES = {'musetalkV15/musetalk.json': '5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47', 'musetalkV15/unet.pth': '7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007', 'sd-vae/config.json': '92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e', 'sd-vae/diffusion_pytorch_model.bin': '1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc', 'whisper/config.json': 'ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98', 'whisper/pytorch_model.bin': '9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d', 'whisper/preprocessor_config.json': '9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666', 'dwpose/dw-ll_ucoco_384.pth': '0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07'}
EXPECTED_HASHES['face-parse-bisent/79999_iter.pth'] = '468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567'
EXPECTED_HASHES['face-parse-bisent/resnet18-5c106cde.pth'] = '5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8'
EXPECTED_HASHES['s3fd/s3fd-619a316812.pth'] = '619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543'


def verify_models(destination=DEST):
    """Verify all required weights offline; never initialize a model or network."""
    destination = Path(destination)
    for name, expected in EXPECTED_HASHES.items():
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/fetch_video_models.py first")
        if sha256(path) != expected:
            raise ValueError(f"Model hash mismatch: {path}; preserve and inspect this file before retrying")
    manifests = []
    for repo, directory, revision, files in MODELS:
        path = destination / directory / "download-manifest.json"
        record = json.loads(path.read_text())
        if record.get("repo") != repo or record.get("revision") != revision:
            raise ValueError(f"Model revision differs from the pinned release: {path}")
        for name in files:
            relative = str((Path(directory) / name).as_posix())
            if record.get("files", {}).get(name, {}).get("sha256") != EXPECTED_HASHES[relative]:
                raise ValueError(f"Model manifest hash differs: {path}: {name}")
        manifests.append({"path": str(path), "sha256": sha256(path), "repo": repo, "revision": revision})
    for url, name in URLS:
        path = (destination / name).with_suffix(Path(name).suffix + ".json")
        record = json.loads(path.read_text())
        if record.get("url") != url or record.get("sha256") != EXPECTED_HASHES[name]:
            raise ValueError(f"Direct-download model manifest differs: {path}")
        manifests.append({"path": str(path), "sha256": sha256(path), "url": url,
                          "model_sha256": EXPECTED_HASHES[name]})
    return manifests


def write_manifest(path, record):
    # Never silently rewrite a different existing provenance record.
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(key) != value for key, value in record.items()):
            raise ValueError(f"Existing manifest differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--verify-only", action="store_true", help="Offline hash/revision validation; no downloads")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if args.verify_only:
        print(json.dumps({"verified": verify_models(destination)}, indent=2))
        return
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    for repo, directory, revision, files in MODELS:
        target = destination / directory
        manifest = target / "download-manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old.get("repo") != repo or old.get("revision") != revision:
                raise ValueError(f"Refusing a different model revision in {manifest}")
        missing = []
        for name in files:
            path = target / name
            expected = EXPECTED_HASHES[str((Path(directory) / name).as_posix())]
            if path.exists() and sha256(path) != expected:
                raise ValueError(f"Refusing to replace a model with an unexpected hash: {path}")
            if not path.exists():
                missing.append(name)
        if missing:
            print(f"Fetching {repo}@{revision}", flush=True)
            snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=missing, max_workers=2)
        for name in files:
            expected = EXPECTED_HASHES[str((Path(directory) / name).as_posix())]
            if sha256(target / name) != expected:
                raise ValueError(f"Downloaded model hash mismatch: {target / name}")
        write_manifest(manifest, {"repo": repo, "revision": revision,
            "files": {name: {"bytes": (target / name).stat().st_size, "sha256": sha256(target / name)} for name in files}})
    for url, filename in URLS:
        path = destination / filename
        expected = EXPECTED_HASHES[filename]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".partial")
            if temp.exists():
                raise FileExistsError(f"Partial download already exists; inspect it before retrying: {temp}")
            print(f"Fetching {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if sha256(temp) != expected:
                raise ValueError(f"Downloaded model hash mismatch: {temp}")
            temp.replace(path)
        if sha256(path) != expected:
            raise ValueError(f"Existing model hash mismatch: {path}")
        write_manifest(path.with_suffix(path.suffix + ".json"),
            {"url": url, "bytes": path.stat().st_size, "sha256": expected})
    print(json.dumps({"verified": verify_models(destination)}, indent=2))


if __name__ == "__main__":
    main()
