#!/usr/bin/env python3
"""Generate speech samples with local Chatterbox Multilingual V2 on CPU.

Inference is offline. The installed multilingual generator limits each take to
1,000 new speech tokens; long scripts should be split into reviewed passages.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import time

if __package__:
    from .fetch_finnish_models import verify_manifest, verify_tokenizer_assets
else:
    from fetch_finnish_models import verify_manifest, verify_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]
REPO = "ResembleAI/chatterbox"
REQUIRED_MODELS = ("ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                   "grapheme_mtl_merged_expanded_v1.json")
MAX_NEW_TOKENS = 1000


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_offline(threads, cache_dir=None, model_dir=None):
    # Force these values before importing Hugging Face or the model libraries.
    cache_dir = Path(cache_dir or ROOT / ".cache").resolve()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": str(threads),
                       "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                       "PKUSEG_HOME": str(cache_dir / "pkuseg"),
                       "HF_HOME": str(cache_dir / "huggingface"),
                       "XDG_CACHE_HOME": str(cache_dir),
                       "NUMBA_CACHE_DIR": str(cache_dir / "numba")})
    # MTLTokenizer initializes this Chinese segmenter even for Finnish.
    # pkuseg has its own downloader, which does not honor HF_HUB_OFFLINE.
    try:
        verify_tokenizer_assets(cache_dir, model_dir)
    except (OSError, ValueError) as error:
        raise ValueError(
            "Offline tokenizer assets are missing or changed. Run setup_audio.py --check; "
            "prepare assets explicitly with setup_audio.py --models. " + str(error)) from error
    return cache_dir


def output_paths(directory, seed, takes, prefix="finnish-parody"):
    paths = []
    for value in range(seed, seed + takes):
        wav = directory / f"{prefix}-seed{value}.wav"
        metadata = wav.with_suffix(".json")
        for path in (wav, metadata):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing take: {path}")
        paths.append((value, wav, metadata))
    return paths


def validate_audio(audio, sample_rate, np, require_unclipped=True):
    if sample_rate != 24000:
        raise ValueError(f"Expected 24000 Hz audio, received {sample_rate}")
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Model returned empty, nonfinite or non-mono audio")
    peak = float(np.max(np.abs(audio)))
    if require_unclipped and peak >= 1.0:
        raise ValueError(f"Audio would clip at PCM export: peak={peak}")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms == 0:
        raise ValueError("Model returned silent audio")
    return {"finite": True, "channels": 1, "sample_rate": sample_rate,
            "frames": int(audio.size), "duration_seconds": audio.size / sample_rate,
            "peak": peak, "rms": rms, "unclipped": peak < 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, required=True,
                        help="Provenance JSON containing sha256 and source_url")
    parser.add_argument("--text", type=Path, required=True, help="UTF-8 script to synthesize")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="finnish-parody",
                        help="Filename prefix; legacy default retained for existing explicit commands")
    parser.add_argument("--language", choices=("fi", "en"), default="fi")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--takes", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.takes < 1 or args.threads < 1:
        parser.error("takes and threads must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_prefix):
        parser.error("--output-prefix must contain only letters, digits, underscores or hyphens")
    if args.seed < 0 or args.seed + args.takes - 1 >= 2**32:
        parser.error("Every take's seed must be between 0 and 2**32 - 1")
    if not math.isfinite(args.cfg_weight) or not 0 <= args.cfg_weight <= 1:
        parser.error("cfg-weight must be finite and between 0 and 1")
    for name in ("model_dir", "reference", "reference_metadata", "text", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())

    # Check every requested output before model loading or any generation.
    takes = output_paths(args.output_dir, args.seed, args.takes, args.output_prefix)
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)
    text = args.text.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Script is empty")
    reference_details = json.loads(args.reference_metadata.read_text(encoding="utf-8"))
    if not isinstance(reference_details, dict):
        raise ValueError("Reference metadata must be a JSON object")
    expected_reference_hash = reference_details.get("sha256")
    if not isinstance(expected_reference_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_reference_hash):
        raise ValueError("Reference metadata must contain a SHA256 digest in sha256")
    if not isinstance(reference_details.get("source_url"), str) or not reference_details["source_url"].strip():
        raise ValueError("Reference metadata must contain a nonempty source_url")
    reference_hash = sha256(args.reference)
    if expected_reference_hash.lower() != reference_hash:
        raise ValueError("Reference metadata hash does not match the supplied audio")

    manifest_path = args.model_dir / "download-manifest.json"
    required = [*REQUIRED_MODELS, "Cangjie5_TC.json"]
    if (args.model_dir / "conds.pt").is_file():
        required.append("conds.pt")
    model_manifest = verify_manifest(args.model_dir, repo=REPO, required=required)
    revision = model_manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model manifest must contain a full commit SHA revision")
    for filename in REQUIRED_MODELS:
        path = args.model_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing local multilingual model file: {path}")
    model_hashes = {}
    for filename in (*REQUIRED_MODELS, "conds.pt", "Cangjie5_TC.json"):
        path = args.model_dir / filename
        if path.is_file():
            model_hashes[filename] = sha256(path)
            recorded = model_manifest.get("files", {}).get(filename)
            if isinstance(recorded, dict) and recorded.get("sha256") and recorded["sha256"] != model_hashes[filename]:
                raise ValueError(f"Model file hash differs from manifest: {filename}")

    cache_dir = configure_offline(args.threads, args.cache_dir, args.model_dir)
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # A changed upstream implementation must not silently remove the take cap.
    if not re.search(r"max_new_tokens\s*=\s*1000\b", inspect.getsource(ChatterboxMultilingualTTS.generate)):
        raise RuntimeError("Multilingual generation's 1000-token cap changed; review before running")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    packages = {name: importlib.metadata.version(name) for name in (
        "chatterbox-tts", "torch", "torchaudio", "transformers", "diffusers", "safetensors",
        "resemble-perth", "setuptools", "numpy", "librosa", "soundfile")}
    settings = {"language_id": args.language, "exaggeration": 0.5, "cfg_weight": args.cfg_weight,
                "temperature": 0.8, "top_p": 1.0, "min_p": 0.05, "repetition_penalty": 2.0}
    print("Loading local Chatterbox Multilingual V2 on CPU (offline)", flush=True)
    started = time.monotonic()
    model = ChatterboxMultilingualTTS.from_local(args.model_dir, device="cpu")
    load_seconds = time.monotonic() - started
    if model.sr != 24000:
        raise RuntimeError(f"Unexpected model sample rate: {model.sr}")
    if not callable(getattr(model.watermarker, "apply_watermark", None)):
        raise RuntimeError("The real Perth watermarker is unavailable")
    print(f"Model loaded in {load_seconds:.1f}s", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed, output, metadata_path in takes:
        # Recheck immediately before each take; exclusive writes also prevent races.
        output_paths(args.output_dir, seed, 1, args.output_prefix)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        started = time.monotonic()
        with torch.inference_mode():
            wav = model.generate(text, audio_prompt_path=str(args.reference), **settings)
        render_seconds = time.monotonic() - started
        audio = wav.detach().cpu().numpy()
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        raw_checks = validate_audio(audio, model.sr, np, require_unclipped=False)
        # Neural float output may exceed PCM full scale. Attenuate once, never
        # amplify, and leave both model sampling and stopping safeguards intact.
        gain = min(1.0, 0.95 / raw_checks["peak"])
        gain_db = 20 * math.log10(gain)
        audio = audio * gain
        checks = validate_audio(audio, model.sr, np)
        print(json.dumps({"raw_float_peak": raw_checks["peak"], "export_ceiling_peak": 0.95,
                          "export_gain_linear": gain, "export_gain_db": gain_db,
                          "export_adjustment": "Downward-only linear gain before PCM16; no amplification"}), flush=True)
        with output.open("xb") as stream:
            sf.write(stream, audio, model.sr, subtype="PCM_16", format="WAV")
        exported, exported_sr = sf.read(output, dtype="float32", always_2d=True)
        if exported.shape != (audio.size, 1):
            raise ValueError("Exported WAV channel/frame count differs from the model output")
        exported_checks = validate_audio(exported[:, 0], exported_sr, np)
        metadata = {
            "label": "AI-generated synthetic speech; not an authentic recording",
            "created_utc": datetime.now(timezone.utc).isoformat(), "script": text,
            "text_path": str(args.text), "text_sha256": sha256(args.text),
            "generator_sha256": sha256(Path(__file__)),
            "model": REPO, "model_variant": "multilingual-v2", "model_snapshot": str(args.model_dir),
            "model_revision": revision, "model_file_sha256": model_hashes,
            "model_manifest": model_manifest, "model_manifest_sha256": sha256(manifest_path),
            "model_source_sha256": sha256(Path(inspect.getfile(ChatterboxMultilingualTTS))),
            "pkuseg_cache": str(cache_dir / "pkuseg"),
            "pkuseg_manifest_sha256": sha256(cache_dir / "pkuseg/download-manifest.json")
                if (cache_dir / "pkuseg/download-manifest.json").is_file() else None,
            "package_versions": packages, "device": "cpu", "offline": True,
            "threads": args.threads, "interop_threads": 1, "seed": seed, "sampling": settings,
            "max_new_speech_tokens": MAX_NEW_TOKENS,
            "token_limit_note": "Verified built-in per-take cap; reaching it may truncate a long passage",
            "reference": str(args.reference), "reference_sha256": reference_hash,
            "reference_source": reference_details["source_url"], "reference_details": reference_details,
            "reference_metadata_sha256": sha256(args.reference_metadata),
            "reference_conditioning": {"speech_prompt_seconds": 6, "decoder_seconds": 10,
                                       "speaker_embedding": "full supplied reference"},
            "watermark": "Chatterbox's built-in Perth watermark retained; detector not independently run",
            "sample_rate": model.sr, "duration_seconds": checks["duration_seconds"],
            "peak": checks["peak"], "rms": checks["rms"], "numeric_checks": checks,
            "raw_float_peak": raw_checks["peak"], "raw_float_rms": raw_checks["rms"],
            "raw_float_checks": raw_checks, "export_ceiling_peak": 0.95,
            "export_gain_linear": gain, "export_gain_db": gain_db,
            "export_adjustment": "gain=min(1, 0.95/raw_float_peak), applied once after built-in watermarking; no upward normalization",
            "exported_wav_checks": exported_checks, "subtype": "PCM_16",
            "model_load_seconds": load_seconds, "render_seconds": render_seconds,
            "output_sha256": sha256(output),
            "assessment": "Generated and numerically checked; pronunciation, accent and voice likeness pending human listening",
        }
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"output": str(output), "duration_seconds": checks["duration_seconds"],
                          "render_seconds": render_seconds}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
