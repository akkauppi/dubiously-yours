#!/usr/bin/env python3
"""Diagnose WAV integrity and speech with independent offline CPU ASR.

Transcripts and word differences are review aids, not proof of synthesis errors,
pronunciation quality, fluency, or resemblance to the reference speaker.
"""

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_words(text):
    # Keep Finnish letters and literal numbers; do not correct names or phonetics.
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)


def compare_words(expected_text, transcript):
    expected, recognized = normalized_words(expected_text), normalized_words(transcript)
    row = list(range(len(recognized) + 1))
    for i, word in enumerate(expected, 1):
        next_row = [i]
        for j, actual in enumerate(recognized, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (word != actual)))
        row = next_row
    differences = [{"operation": tag, "expected": expected[a:b],
                    "recognized": recognized[c:d],
                    "expected_word_range": [a, b], "recognized_word_range": [c, d]}
                   for tag, a, b, c, d in SequenceMatcher(
                       None, expected, recognized, autojunk=False).get_opcodes()
                   if tag != "equal"]
    return {"expected_text": expected_text, "normalized_expected_words": expected,
            "normalized_recognized_words": recognized, "word_edit_distance": row[-1],
            "normalized_words_match": row[-1] == 0, "differences": differences,
            "interpretation": "Differences can arise from ASR itself; they require listening review."}


def model_provenance(directory):
    manifest_path = directory / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model manifest must identify a full commit SHA revision")
    recorded = manifest.get("files", {})
    if not isinstance(recorded, dict) or not all(name in recorded for name in REQUIRED_MODEL_FILES):
        raise ValueError("Model manifest lacks required Whisper files")
    actual = {}
    for name, entry in recorded.items():
        relative = Path(name)
        if relative.is_absolute() or not (directory / relative).resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Invalid model manifest path: {name}")
        if (not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError(f"Model manifest lacks a full SHA256 digest: {name}")
        path = directory / relative
        actual[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual[name] != {"bytes": entry.get("bytes"), "sha256": entry["sha256"]}:
            raise ValueError(f"Model file differs from its manifest: {name}")
    return {"repository": manifest.get("repo"), "revision": revision,
            "directory": str(directory), "manifest": manifest,
            "manifest_sha256": sha256(manifest_path), "verified_files": actual}


def audio_checks(path, np, sf):
    record = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    try:
        with sf.SoundFile(path) as stream:
            rate, channels, declared_frames = stream.samplerate, stream.channels, stream.frames
            subtype, audio_format = stream.subtype, stream.format
            samples = stream.read(dtype="float64", always_2d=True)
        frames = samples.shape[0]
        finite = bool(np.isfinite(samples).all())
        pcm_bits = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}
        bits = pcm_bits.get(subtype)
        rails = int(np.count_nonzero((samples <= -1.0) | (samples >= 1.0 - 2 ** (1 - bits)))) if bits else None
        full_scale = int(np.count_nonzero(np.abs(samples) >= 1.0))
        peak = float(np.max(np.abs(samples))) if frames and finite else None
        rms = float(np.sqrt(np.mean(samples ** 2))) if frames and finite else None
        if rms is not None and not math.isfinite(rms):
            rms = None
        record.update({"full_decode": "passed", "format": audio_format, "subtype": subtype,
                       "sample_rate": rate, "channels": channels, "frames": frames,
                       "header_frames": declared_frames, "header_frame_count_matches": frames == declared_frames,
                       "duration_seconds": frames / rate, "finite_samples": finite,
                       "nonfinite_samples": int(np.count_nonzero(~np.isfinite(samples))),
                       "peak": peak, "rms": rms, "silent": peak == 0 if peak is not None else None,
                       "samples_at_pcm_limit": rails, "samples_at_or_above_full_scale": full_scale,
                       "potential_clipping": bool((rails or 0) or full_scale),
                       "clipping_note": "PCM rail/full-scale samples flag possible clipping; they do not prove waveform clipping.",
                       "ready_for_asr": bool(frames and finite and frames == declared_frames and peak > 0)})
    except (RuntimeError, ValueError, OSError) as error:
        record.update(full_decode="failed", ready_for_asr=False,
                      decode_error=f"{type(error).__name__}: {error}")
    return record


def optional_number(obj, name):
    value = getattr(obj, name, None)
    return float(value) if value is not None and math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+", help="WAV files to diagnose independently")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing paths are refused")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/faster-whisper-small",
                        help="Local multilingual Whisper directory with hashed download-manifest.json")
    parser.add_argument("--expected-text", help="Literal expected text, compared ONLY after decoding; never an ASR prompt")
    parser.add_argument("--language", choices=("fi", "en"), default="fi")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    # Do not resolve the final component before checking a dangling output symlink.
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    args.output = args.output.absolute()
    paths = [path.resolve(strict=True) for path in args.audio]
    if len(paths) != len(set(paths)):
        raise ValueError("The same input WAV was supplied more than once")
    if any(path.suffix.lower() != ".wav" or not path.is_file() for path in paths):
        raise ValueError("Each input must be an existing WAV file")
    if args.expected_text is not None and not args.expected_text.strip():
        parser.error("--expected-text cannot be empty")
    args.model_dir = args.model_dir.resolve(strict=True)

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "4",
                       "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    os.environ["HF_HOME"] = str(args.cache_dir.resolve() / "huggingface")
    os.environ["XDG_CACHE_HOME"] = str(args.cache_dir.resolve())
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    provenance = model_provenance(args.model_dir)
    results = [audio_checks(path, np, sf) for path in paths]
    model = None
    settings = {"language": args.language, "task": "transcribe", "beam_size": 5,
                "condition_on_previous_text": False, "vad_filter": False,
                "initial_prompt": None, "hotwords": None, "prefix": None,
                "word_timestamps": False, "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    for record in results:
        if not record["ready_for_asr"]:
            record["asr_status"] = "skipped_invalid_or_silent_audio"
            continue
        if model is None:
            model = WhisperModel(str(args.model_dir), device="cpu", compute_type="int8",
                                 cpu_threads=4, num_workers=1, local_files_only=True)
            if args.language != "en" and not model.model.is_multilingual:
                raise ValueError("Non-English diagnostics require a multilingual Whisper model")
        started = time.monotonic()
        decoded, info = model.transcribe(record["path"], **settings)
        decoded = list(decoded)
        transcript = " ".join(segment.text.strip() for segment in decoded)
        record.update(asr_status="decoded", asr_seconds=time.monotonic() - started,
                      transcript=transcript, reported_language=info.language,
                      segments=[{"start": float(s.start), "end": float(s.end),
                                 "text": s.text.strip(), "avg_logprob": optional_number(s, "avg_logprob"),
                                 "no_speech_probability": optional_number(s, "no_speech_prob"),
                                 "compression_ratio": optional_number(s, "compression_ratio"),
                                 "temperature": optional_number(s, "temperature")} for s in decoded])
        # This sample's decoding has completed before expected text is consulted.
        if args.expected_text is not None:
            record["word_comparison"] = compare_words(args.expected_text, transcript)
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Input audio changed during diagnosis: {record['path']}")
        print(json.dumps({"path": record["path"], "asr_transcript": transcript,
                          "word_comparison": record.get("word_comparison")}, ensure_ascii=False), flush=True)

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checker": str(Path(__file__).resolve()),
              "checker_sha256": sha256(__file__), "model": provenance,
              "model_source_sha256": sha256(inspect.getfile(WhisperModel)),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("faster-whisper", "ctranslate2", "numpy", "soundfile")},
              "device": "cpu", "compute_type": "int8", "cpu_threads": 4,
              "local_files_only": True, "offline": True, "asr_settings": settings,
              "sample_count": len(results), "samples": results,
              "limitations": ["Transcripts quote the recognizer's output, not a verified record of what was spoken.",
                              "ASR mismatches do not by themselves establish missing/repeated words or pronunciation errors.",
                              "The requested language is explicit; reported language is not independent language verification.",
                              "No name correction, phonetic hints, expected-text prompt or hotwords are used.",
                              "This report does not assess voice likeness or replace listening to the samples."]}
    # Exclusive creation also protects against an output created while ASR ran.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Saved diagnostic report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
