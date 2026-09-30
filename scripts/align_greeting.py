#!/usr/bin/env python3
"""Check every greeting segment with offline CPU ASR and align original captions."""

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import unicodedata

if __package__:
    from .check_finnish_samples import model_provenance
else:
    from check_finnish_samples import model_provenance

ROOT = Path(__file__).resolve().parents[1]
NAME_WORDS = frozenset()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_atoms(text, language="fi"):
    """Comparison tokens only; never use these spellings as caption text."""
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    tokens = re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, flags=re.UNICODE)
    # Possessive apostrophes are transcription punctuation, not different audio.
    tokens = [token.replace("'", "") for token in tokens]
    expansions = {"cannot": ["can", "not"]} if language == "en" else {}
    return [atom for token in tokens for atom in expansions.get(token, [token])]


def expected_atoms(words, language="fi"):
    return [{"token": token, "word_index": i}
            for i, word in enumerate(words) for token in normalized_atoms(word, language)]


def recognized_atoms(words, language="fi"):
    atoms = []
    for i, word in enumerate(words):
        tokens = normalized_atoms(word["word"], language)
        for j, token in enumerate(tokens):
            duration = word["end"] - word["start"]
            atoms.append({"token": token, "word_index": i,
                          "start": word["start"] + duration * j / len(tokens),
                          "end": word["start"] + duration * (j + 1) / len(tokens)})
    return atoms


def align_words(display, recognized, lower, upper, *, language="fi", name_words=None):
    name_words = NAME_WORDS if name_words is None else set(name_words)
    words = display.split()
    expected, heard = expected_atoms(words, language), recognized_atoms(recognized, language)
    matcher = SequenceMatcher(None, [w["token"] for w in expected],
                              [w["token"] for w in heard], autojunk=False)
    assignments = [[] for _ in words]
    differences = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag != "equal":
            expected_tokens = [w["token"] for w in expected[a0:a1]]
            heard_tokens = [w["token"] for w in heard[b0:b1]]
            name_only = bool(expected_tokens) and set(expected_tokens) <= name_words
            name_variant = name_only and tag == "replace"
            kind = ("name_asr_difference" if name_variant else
                    "name_missing_from_asr" if name_only and tag == "delete" else
                    {"replace": "substitution", "delete": "missing_from_asr",
                     "insert": "extra_or_repeated_in_asr"}[tag])
            differences.append({"operation": tag, "kind": kind,
                                "material_review": not name_variant,
                                "expected": expected_tokens, "recognized": heard_tokens,
                                "expected_atom_range": [a0, a1],
                                "recognized_atom_range": [b0, b1]})
        if tag == "equal":
            for wanted, actual in zip(expected[a0:a1], heard[b0:b1]):
                assignments[wanted["word_index"]].append(
                    (actual["start"], actual["end"], "asr_token_match"))
        elif tag == "replace":
            # A name may occupy several ASR words. Preserve its original spelling
            # and distribute only timing within the recognized replacement span.
            start, end = heard[b0]["start"], heard[b1 - 1]["end"]
            for j, wanted in enumerate(expected[a0:a1]):
                assignments[wanted["word_index"]].append(
                    (start + (end - start) * j / (a1 - a0),
                     start + (end - start) * (j + 1) / (a1 - a0),
                     "replacement_span_estimate"))
    aligned = []
    for word, spans in zip(words, assignments):
        aligned.append({"word": word,
                        "start": min(s[0] for s in spans) if spans else None,
                        "end": max(s[1] for s in spans) if spans else None,
                        "timing_sources": sorted({s[2] for s in spans})})
    cursor = 0
    while cursor < len(aligned):
        if aligned[cursor]["start"] is not None:
            cursor += 1
            continue
        stop = cursor
        while stop < len(aligned) and aligned[stop]["start"] is None:
            stop += 1
        start = aligned[cursor - 1]["end"] if cursor else lower
        end = aligned[stop]["start"] if stop < len(aligned) else upper
        end = max(start, end)
        for j in range(cursor, stop):
            aligned[j].update(start=start + (end - start) * (j - cursor) / (stop - cursor),
                              end=start + (end - start) * (j + 1 - cursor) / (stop - cursor),
                              timing_sources=["missing_asr_interpolation"])
        cursor = stop
    for word in aligned:
        word["start"] = round(max(lower, min(upper, word["start"])), 4)
        word["end"] = round(max(word["start"], min(upper, word["end"])), 4)
    return aligned, differences


def wrap_words(words, width):
    text = " ".join(words)
    if len(text) <= width:
        return text
    candidates = [(" ".join(words[:i]), " ".join(words[i:]))
                  for i in range(1, len(words))]
    candidates = [(a, b) for a, b in candidates if max(len(a), len(b)) <= width]
    if not candidates:
        return None
    # Keep lines reasonably balanced; a comma is a useful optional break.
    a, b = min(candidates, key=lambda pair: abs(len(pair[0]) - len(pair[1]))
               - (4 if pair[0].endswith((",", ";", ":")) else 0))
    return a + "\n" + b


def make_cues(segment, aligned, width):
    cues, offset = [], 0
    for paragraph in re.split(r"\n\s*\n", segment["display_text"]):
        count = len(paragraph.split())
        stop = offset + count
        while offset < stop:
            end = offset + 1
            while end <= stop:
                chunk = aligned[offset:end]
                fits = wrap_words([w["word"] for w in chunk], width) is not None
                duration = chunk[-1]["end"] - chunk[0]["start"]
                if not fits or (duration > 4.8 and end > offset + 1):
                    if end == offset + 1 and not fits:
                        raise ValueError(f"A caption word exceeds the requested line width: {chunk[0]['word']}")
                    end -= 1
                    break
                if end == stop:
                    break
                end += 1
            if end < stop:
                boundaries = [j for j in range(offset + 3, end + 1)
                              if aligned[j - 1]["word"].endswith((",", ";", ":", ".", "!", "?"))
                              and j - offset >= 0.6 * (end - offset)]
                if boundaries:
                    end = boundaries[-1]
            chunk = aligned[offset:end]
            wrapped = wrap_words([w["word"] for w in chunk], width)
            if wrapped is None:
                raise ValueError("A caption word exceeds the requested line width")
            cue_start = max(segment["start"], chunk[0]["start"] - 0.05)
            cue_end = min(segment["end_with_pause"],
                          max(chunk[-1]["end"] + 0.16, cue_start + 0.85))
            cues.append({"start": cue_start, "end": cue_end, "text": wrapped})
            offset = end
    return cues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--line-width", type=int, default=40)
    parser.add_argument("--language", choices=("en", "fi"), default="fi",
                        help="ASR/comparison language (default: fi)")
    parser.add_argument("--model-dir", type=Path,
                        help="Local Whisper model; defaults to models/faster-whisper-small")
    parser.add_argument("--name-words", default="",
                        help="Optional comma-separated name forms for review flags only; never an ASR hint")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    if args.line_width < 15:
        parser.error("--line-width must be at least 15")
    name_words = set()
    for spelling in args.name_words.split(","):
        if not spelling.strip():
            continue
        atoms = normalized_atoms(spelling.strip(), args.language)
        if len(atoms) != 1:
            parser.error("Each --name-words entry must be a single name word")
        name_words.add(atoms[0])
    output_paths = [args.output_dir / filename for filename in
                    ("caption-alignment.json", "caption-cues.json")]
    for path in output_paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing caption review: {path}")
    model_dir = (args.model_dir or ROOT / "models/faster-whisper-small").resolve()
    model_manifest_path = model_dir / "download-manifest.json"
    provenance = model_provenance(model_dir)
    model_manifest = provenance["manifest"]

    metadata = json.loads(args.metadata.read_text())
    if not metadata["segments"]:
        raise ValueError("The greeting must contain at least one segment")
    joined = "\n\n".join(segment["display_text"] for segment in metadata["segments"])
    if joined != metadata["script"].strip():
        raise ValueError("Segment display text no longer reproduces the original script")
    if digest(metadata["output"]) != metadata["sha256"]:
        raise ValueError("Assembled audio changed after its metadata was written")
    cache_dir = args.cache_dir.resolve()
    os.environ.update({"HF_HOME": str(cache_dir / "huggingface"),
                       "XDG_CACHE_HOME": str(cache_dir), "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8",
                         cpu_threads=4, num_workers=1, local_files_only=True)
    if args.language == "fi" and not model.model.is_multilingual:
        raise ValueError("Finnish alignment requires a multilingual Whisper model")
    records, cues = [], []
    for segment in metadata["segments"]:
        if digest(segment["path"]) != segment["sha256"]:
            raise ValueError(f"Segment audio changed: {segment['id']}")
        results, info = model.transcribe(segment["path"], language=args.language, task="transcribe", beam_size=5,
                                        word_timestamps=True, vad_filter=False,
                                        condition_on_previous_text=False,
                                        initial_prompt=None, hotwords=None)
        results = list(results)
        recognized = [{"word": word.word.strip(),
                       "start": round(segment["start"] + word.start, 4),
                       "end": round(segment["start"] + word.end, 4),
                       "probability": word.probability}
                      for result in results for word in (result.words or [])]
        transcript = " ".join(result.text.strip() for result in results)
        aligned, differences = align_words(segment["display_text"], recognized,
                                           segment["start"], segment["speech_end"],
                                           language=args.language, name_words=name_words)
        material = [item for item in differences if item["material_review"]]
        status = "review_required" if material else (
            "name_difference_only" if differences else "normalized_words_match")
        records.append({"id": segment["id"], "start": segment["start"],
                        "speech_end": segment["speech_end"],
                        "end_with_pause": segment["end_with_pause"],
                        "display_text": segment["display_text"],
                        "audio_sha256": segment["sha256"],
                        "transcript": transcript, "status": status,
                        "differences": differences, "recognized_words": recognized,
                        "aligned_display_words": aligned})
        cues.extend(make_cues(segment, aligned, args.line_width))
        print(json.dumps({"segment": segment["id"], "status": status,
                          "transcript": transcript, "differences": differences}), flush=True)

    duration = metadata["duration_seconds"]
    for previous, following in zip(cues, cues[1:]):
        if previous["end"] > following["start"]:
            boundary = (previous["end"] + following["start"]) / 2
            previous["end"] = boundary
            following["start"] = boundary
    for cue in cues:
        cue["start"] = round(max(0, min(duration, cue["start"])), 3)
        cue["end"] = round(max(0, min(duration, cue["end"])), 3)
        if cue["end"] <= cue["start"]:
            raise ValueError(f"Invalid caption interval: {cue}")
        if len(cue["text"].splitlines()) > 2 or any(
                len(line) > args.line_width for line in cue["text"].splitlines()):
            raise ValueError(f"Caption exceeds line limits: {cue}")
    # Only whitespace may change: all original spelling, numbers and punctuation survive.
    if " ".join(" ".join(cue["text"].split()) for cue in cues) != " ".join(metadata["script"].split()):
        raise ValueError("Caption text no longer reproduces the original script")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audio": metadata["output"], "audio_sha256": metadata["sha256"],
        "duration_seconds": duration, "model": model_manifest["repo"],
        "model_revision": model_manifest["revision"], "model_directory": str(model_dir.resolve()),
        "model_manifest_sha256": digest(model_manifest_path),
        "model_verified_files": provenance["verified_files"],
        "faster_whisper_version": importlib.metadata.version("faster-whisper"),
        "settings": {"device": "cpu", "compute_type": "int8", "cpu_threads": 4,
                     "language": args.language, "task": "transcribe",
                     "word_timestamps": True, "beam_size": 5, "vad_filter": False,
                     "condition_on_previous_text": False, "initial_prompt": None, "hotwords": None},
        "comparison_name_words": sorted(name_words),
        "segments_checked": len(records),
        "material_review_segments": [r["id"] for r in records if r["status"] == "review_required"],
        "segments": records, "caption_count": len(cues),
        "caption_constraints": {"max_lines": 2, "max_characters_per_line": args.line_width,
                                "original_words_and_punctuation_preserved": True,
                                "non_overlapping": True, "clamped_to_audio": True},
        "limitations": ["ASR mismatches are review flags, not proof of a generation error.",
                        "Name transcription neither proves nor disproves pronunciation quality.",
                        "Caption words come only from original display text, never phonetic TTS hints.",
                        "Matched tokens use ASR timing; replacements and missing tokens have estimated timing.",
                        "This automated check is not a human listening assessment."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing caption review: {path}")
    for filename, data in (("caption-alignment.json", report), ("caption-cues.json", cues)):
        path = args.output_dir / filename
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"segments_checked": len(records), "caption_count": len(cues),
                      "material_review_segments": report["material_review_segments"],
                      "alignment": str(args.output_dir / "caption-alignment.json"),
                      "cues": str(args.output_dir / "caption-cues.json")}), flush=True)


if __name__ == "__main__":
    main()
