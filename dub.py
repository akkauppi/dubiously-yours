#!/usr/bin/env python3
"""Dubiously Yours: local, staged synthetic-speech and video workflow."""

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent
STAGES = ("reference", "audio", "align", "video", "render", "export", "verify")
SAMPLING = {"language_id": "fi", "exaggeration": 0.5, "cfg_weight": 0.5,
            "temperature": 0.8, "top_p": 1.0, "min_p": 0.05, "repetition_penalty": 2.0}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def relative_path(value, root=ROOT):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Use nonempty repository-relative paths in job files")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the repository: {value}")
    return path


def intervals(values, *, frame_aligned=False):
    if not isinstance(values, list) or not values:
        raise ValueError("Select at least one [start,end] interval in seconds")
    result = []
    for value in values:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Each interval must be [start,end]")
        a, b = value
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in value) or not 0 <= a < b:
            raise ValueError("Intervals need finite times with 0 <= start < end")
        if frame_aligned and any(abs(t * 25 - round(t * 25)) > 1e-6 for t in value):
            raise ValueError("Video intervals must use 0.04-second (25 fps) boundaries")
        result.append([a, b])
    return result


def fit_video_ranges(available, frame_count):
    """Use forward intervals in the chosen order; never loop or reverse."""
    result, remaining = [], frame_count
    ranges = intervals(available, frame_aligned=True)
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Video intervals must be chronological and nonoverlapping")
    for start, end in ranges:
        count = min(round((end - start) * 25), remaining)
        if count:
            result.append([start, round(start + count / 25, 8)])
            remaining -= count
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"Not enough video: select another {remaining / 25:.2f} seconds of close-ups; no automatic looping")
    return result


def load_job(path, root=ROOT):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Keep job files inside this repository")
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1:
        raise ValueError("Unsupported job schema_version; expected 1")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", job.get("id", "")):
        raise ValueError("Job id must be a short lowercase filename, without directory separators")
    if job.get("language") not in ("fi", "en"):
        raise ValueError("This workflow supports language fi or en")
    for key in ("recipient",):
        if not isinstance(job.get(key), str) or not job[key].strip() or any(c in job[key] for c in "\r\n"):
            raise ValueError(f"{key} must be a nonempty single line")
    relative_path(job["script"], root)
    for kind in ("voice", "video"):
        media = job[kind]
        relative_path(media["media"], root)
        for key in ("source_url", "title", "credit" if kind == "voice" else "publisher"):
            if not isinstance(media.get(key), str) or not media[key].strip() or any(c in media[key] for c in "\r\n"):
                raise ValueError(f"{kind}.{key} must be a nonempty single line")
        if not re.match(r"^(https?://[^/\s]+|urn:[^\s]+)", media["source_url"]):
            raise ValueError(f"{kind}.source_url must be an HTTP(S) URL or urn:recording:... identifier")
        intervals(media["intervals_seconds"], frame_aligned=kind == "video")
    offset = job["video"].get("source_offset_seconds", 0)
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0:
        raise ValueError("video.source_offset_seconds must be finite and nonnegative")
    tts = job.get("tts", {})
    seed, pause = tts.get("seed", 43), tts.get("pause_seconds", 0.5)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("tts.seed must be an integer in [0, 2**32)")
    if isinstance(pause, bool) or not isinstance(pause, (int, float)) or not math.isfinite(pause) or not 0 <= pause <= 10:
        raise ValueError("tts.pause_seconds must be between 0 and 10")
    export = job.get("export", {})
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", export.get("basename", "greeting")):
        raise ValueError("export.basename must be a simple filename stem")
    gain = export.get("gain_db", 0)
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or not -60 <= gain <= 12:
        raise ValueError("export.gain_db must be between -60 and 12")
    return job


def paths(job, root=ROOT):
    work = root / "projects" / job["id"] / "work"
    out = root / "outputs" / job["id"]
    result = {"work": work, "out": out, "reference": work / "reference.wav",
            "reference_meta": work / "reference.json", "plan": work / "speech-plan.json",
            "audio": out / "audio/greeting.wav", "audio_meta": out / "audio/greeting.json",
            "cues": out / "caption-cues.json", "inputs": work / "video",
            "source_info": work / "video-source.json",
            "lipsync": out / "lipsync/v15/portrait-lipsync.mp4",
            "export": out / (job.get("export", {}).get("basename", "greeting") + ".mp4")}
    if any(not path.resolve().is_relative_to(root.resolve()) for path in result.values()):
        raise ValueError("A project/output symlink points outside the repository")
    return result


def init_job(identity, language, root=ROOT):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", identity):
        raise ValueError("Use a lowercase project name such as birthday-demo")
    directory = root / "projects" / identity
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Project directory points outside the repository")
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Project already exists: {directory}; choose a new name")
    template = root / "examples" / f"demo-{language}.txt"
    text = template.read_text(encoding="utf-8")
    directory.mkdir(parents=True)
    (directory / "script.txt").write_text(text, encoding="utf-8")
    job = {"schema_version": 1, "id": identity, "language": language,
           "recipient": "Ystävä" if language == "fi" else "A friend",
           "script": f"projects/{identity}/script.txt",
           "voice": {"media": "input/voice.wav", "source_url": "urn:recording:my-voice",
                     "title": "My voice recording", "credit": "Recorded by me", "intervals_seconds": [[0, 15]]},
           "video": {"media": "input/video.mp4", "source_url": "urn:recording:my-video",
                     "title": "My video recording", "publisher": "Recorded by me", "source_offset_seconds": 0,
                     "intervals_seconds": [[0, 40]]},
           "tts": {"seed": 43, "pause_seconds": 0.5, "groups": None},
           "export": {"basename": "greeting", "gain_db": 0}}
    write_json(directory / "job.json", job)
    return directory / "job.json"


def execute(command, *, dry_run=False, root=ROOT):
    print("$ " + shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), cwd=root, check=True)


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                               "-of", "json", str(path)], text=True))


def new_files(*files):
    for path in files:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; keep this take and create a new project for changed inputs")


def prepare_reference(job, p, *, dry_run=False, root=ROOT):
    voice = job["voice"]
    source = relative_path(voice["media"], root)
    ranges = intervals(voice["intervals_seconds"])
    if sum(b - a for a, b in ranges) < 10:
        raise ValueError("Use at least 10 seconds of clear speech for this workflow")
    filters = [f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,aresample=24000,aformat=channel_layouts=mono[a{i}]"
               for i, (a, b) in enumerate(ranges)]
    filters.append("".join(f"[a{i}]" for i in range(len(ranges))) + f"concat=n={len(ranges)}:v=0:a=1[a]")
    command = ["ffmpeg", "-v", "error", "-n", "-i", source, "-filter_complex", ";".join(filters),
               "-map", "[a]", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", p["reference"]]
    if dry_run:
        execute(command, dry_run=True, root=root)
        return
    new_files(p["reference"], p["reference_meta"])
    media_info = probe(source)
    if not any(s["codec_type"] == "audio" for s in media_info["streams"]):
        raise ValueError("Voice media has no audio track")
    if max(b for _, b in ranges) > float(media_info["format"]["duration"]) + 0.001:
        raise ValueError("Voice interval extends beyond the source recording")
    p["work"].mkdir(parents=True, exist_ok=True)
    execute(command, root=root)
    with wave.open(str(p["reference"]), "rb") as stream:
        rate, count = stream.getframerate(), stream.getnframes()
        pcm = array("h", stream.readframes(count))
    if sys.byteorder != "little":
        pcm.byteswap()
    if not pcm or max(abs(x) for x in pcm) == 0:
        raise ValueError("Reference is empty or silent; select speech and use a fresh project")
    if count / rate < 10:
        raise ValueError("Use at least 10 seconds of clear speech for this workflow")
    if abs(count / rate - sum(b - a for a, b in ranges)) > 0.002:
        raise ValueError("Extracted reference length differs from the requested intervals")
    if min(pcm) <= -32768 or max(pcm) >= 32767:
        raise ValueError("Reference reaches PCM clipping limits; fix the source before conditioning")
    write_json(p["reference_meta"], {"source_url": voice["source_url"], "source_title": voice["title"],
        "credit": voice["credit"], "source_media": str(source), "source_sha256": sha256(source),
        "source_excerpts_seconds": ranges, "sha256": sha256(p["reference"]),
        "sample_rate": rate, "channels": 1, "duration_seconds": count / rate,
        "processing": "Forward selected speech, mono 24 kHz PCM16, no gain/denoise/time stretch",
        "selection_note": "User-selected intervals; speaker isolation and recording quality require listening"})


def make_plan(job, p, root=ROOT):
    original = relative_path(job["script"], root).read_text(encoding="utf-8").strip()
    if not original:
        raise ValueError("Script is empty")
    paragraphs = re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", original)
    tts = job.get("tts", {})
    groups = tts.get("groups")
    if groups is None:
        groups = [{"id": f"part-{i+1:02d}", "paragraphs": [i], "pause_after": tts.get("pause_seconds", 0.5)}
                  for i in range(len(paragraphs))]
    # Reuse the same paragraph/seed checks as the inference helper, without
    # importing any neural libraries or loading a model.
    from scripts.generate_finnish_greeting import planned_groups
    plan = {"label": "Fictional synthetic greeting; not an authentic statement", "recipient": job["recipient"],
            "script": job["script"], "model_dir": "models/chatterbox-multilingual-v2",
            "reference": str(p["reference"].relative_to(root)),
            "reference_metadata": str(p["reference_meta"].relative_to(root)),
            "seed": tts.get("seed", 43), "sampling": {**SAMPLING, "language_id": job["language"]}, "groups": groups}
    planned_groups(plan, original)
    return plan


def check_hash(path, expected, label):
    if sha256(path) != expected:
        raise ValueError(f"Hash mismatch: {label}; preserve this take and use a new project for changed inputs")


def validate_dependencies(job, p, stage, root=ROOT):
    """Refuse stale artifacts after edits to the job, source media or script."""
    voice = job["voice"]
    source = relative_path(voice["media"], root)
    reference = json.loads(p["reference_meta"].read_text())
    expected = {"source_media": str(source), "source_url": voice["source_url"],
                "source_title": voice["title"], "credit": voice["credit"],
                "source_excerpts_seconds": voice["intervals_seconds"]}
    if any(reference.get(key) != value for key, value in expected.items()):
        raise ValueError("Voice settings changed since reference preparation; use a new project")
    check_hash(source, reference["source_sha256"], "voice source")
    check_hash(p["reference"], reference["sha256"], "voice reference")
    if stage == "audio":
        return
    metadata = json.loads(p["audio_meta"].read_text())
    check_hash(p["audio"], metadata["sha256"], "generated audio")
    check_hash(relative_path(job["script"], root), metadata["script_sha256"], "script")
    check_hash(p["reference_meta"], metadata["reference_metadata_sha256"], "voice metadata")
    check_hash(p["reference"], metadata["reference_sha256"], "audio conditioning reference")
    check_hash(p["plan"], metadata["plan_sha256"], "speech plan")
    if json.loads(p["plan"].read_text()) != make_plan(job, p, root):
        raise ValueError("Speech settings changed since generation; use a new project")
    if stage not in ("render", "export", "verify"):
        return
    prepared = json.loads((p["inputs"] / "inputs.json").read_text())
    video = job["video"]
    source = relative_path(video["media"], root)
    with wave.open(str(p["audio"]), "rb") as stream:
        frames = (stream.getnframes() + 959) // 960 + 2
    chosen = fit_video_ranges(video["intervals_seconds"], frames)
    expected = {"source_clip": str(source), "source_url": video["source_url"],
                "source_title": video["title"], "uploader": video["publisher"],
                "downloaded_source_offset_seconds": video.get("source_offset_seconds", 0),
                "local_frame_intervals": [[round(a * 25), round(b * 25)] for a, b in chosen]}
    if any(prepared.get(key) != value for key, value in expected.items()):
        raise ValueError("Video settings changed since preparation; use a new project")
    for path, key in ((source, "source_clip_sha256"), (p["audio"], "original_audio_sha256"),
                      (p["audio_meta"], "audio_metadata_sha256"), (p["source_info"], "source_info_sha256"),
                      (p["inputs"] / "source-video.mp4", "sha256"),
                      (p["inputs"] / "inference-audio.wav", "inference_audio_sha256")):
        check_hash(path, prepared[key], key)


def verify_export(job, p):
    record = json.loads(p["export"].with_suffix(".json").read_text())
    if (Path(record["output"]).resolve() != p["export"].resolve()
            or record["recipient"] != job["recipient"] or record["language"] != job["language"]
            or record["gain_db"] != job.get("export", {}).get("gain_db", 0)):
        raise ValueError("Export settings differ from the current job")
    for file_key, hash_key in (("output", "sha256"), ("mp3", "mp3_sha256"), ("audio", "audio_sha256"),
                               ("caption_cues", "caption_cues_sha256"), ("lipsync_input", "lipsync_input_sha256"),
                               ("audio_metadata", "audio_metadata_sha256"), ("caption_alignment", "caption_alignment_sha256")):
        check_hash(record[file_key], record[hash_key], file_key)
    if record.get("lipsync_metadata"):
        check_hash(record["lipsync_metadata"], record["lipsync_metadata_sha256"], "lipsync_metadata")
    check_hash(p["export"].with_suffix(".ass"), record["captions_sha256"], "captions")
    source = record["source_video"]
    check_hash(source["metadata"], source["metadata_sha256"], "source metadata")
    for file_key in ("output", "mp3"):
        execute(["ffmpeg", "-v", "error", "-i", record[file_key], "-f", "null", "-"])
    current = probe(Path(record["output"]))
    video = next(s for s in current["streams"] if s["codec_type"] == "video")
    if (video["width"], video["height"], video["avg_frame_rate"], int(video["nb_frames"])) != (1280, 720, "25/1", record["frames"]):
        raise ValueError("Export video differs from its recorded format/frame count")
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "output": record["output"],
              "sha256": record["sha256"], "full_decode": "MP4 and MP3 passed", "artifact_and_sidecar_hashes": "passed",
              "duration_seconds": record["duration_seconds"], "frames": record["frames"],
              "remaining_review": "Listen for words/pronunciation and watch full playback for lip sync, cuts, blinks and mouth artifacts"}
    print(json.dumps(report, indent=2))
    return report


def run_stage(job, stage, *, dry_run=False, root=ROOT):
    p = paths(job, root)
    audio_python = root / ".venv-audio/bin/python"
    video_python = root / ".venv-video/bin/python"
    if stage == "reference":
        prepare_reference(job, p, dry_run=dry_run, root=root)
        return
    if not dry_run:
        validate_dependencies(job, p, stage, root)
    if stage == "audio":
        plan = make_plan(job, p, root)
        if not dry_run:
            if p["plan"].exists():
                if json.loads(p["plan"].read_text()) != plan:
                    raise ValueError("Speech plan changed; preserve this take and initialize a new project")
            else:
                write_json(p["plan"], plan)
        command = [audio_python, root / "scripts/generate_finnish_greeting.py", "--plan", p["plan"],
                   "--output-dir", p["out"] / "audio"]
    elif stage == "align":
        command = [audio_python, root / "scripts/align_greeting.py", "--metadata", p["audio_meta"],
                   "--output-dir", p["out"], "--language", job["language"],
                   "--model-dir", root / "models/faster-whisper-small"]
    elif stage == "video":
        if dry_run and not p["audio_meta"].exists():
            print("Video plan: fit the selected forward ranges to the generated audio, plus two silent inference frames.")
            print("Run audio first to calculate exact frame counts; no files changed.")
            return
        metadata = json.loads(p["audio_meta"].read_text())
        if sha256(p["audio"]) != metadata["sha256"]:
            raise ValueError("Generated audio differs from its metadata")
        with wave.open(str(p["audio"]), "rb") as stream:
            if stream.getframerate() != 24000:
                raise ValueError("Expected 24 kHz generated audio")
            frames = (stream.getnframes() + 959) // 960 + 2
        chosen = fit_video_ranges(job["video"]["intervals_seconds"], frames)
        source_info = {"webpage_url": job["video"]["source_url"], "title": job["video"]["title"],
                       "uploader": job["video"]["publisher"]}
        if not dry_run:
            if p["source_info"].exists():
                if json.loads(p["source_info"].read_text()) != source_info:
                    raise ValueError("Video provenance changed; use a fresh project")
            else:
                write_json(p["source_info"], source_info)
        command = [sys.executable, root / "scripts/prepare_interview_video.py",
                   "--source-video", relative_path(job["video"]["media"], root), "--source-info", p["source_info"],
                   "--source-offset", str(job["video"].get("source_offset_seconds", 0)),
                   "--audio-metadata", p["audio_meta"], "--ranges", json.dumps(chosen), "--output-dir", p["inputs"]]
    elif stage == "render":
        command = [video_python, root / "scripts/render_portrait.py", "--input-video", p["inputs"] / "source-video.mp4",
                   "--source-metadata", p["inputs"] / "inputs.json", "--audio", p["inputs"] / "inference-audio.wav",
                   "--output-dir", p["out"] / "lipsync"]
    elif stage == "export":
        command = [sys.executable, root / "scripts/export_video_greeting.py", "--audio-metadata", p["audio_meta"],
                   "--caption-cues", p["cues"], "--lipsync-input", p["lipsync"], "--output-dir", p["out"],
                   "--basename", job.get("export", {}).get("basename", "greeting"),
                   "--source-metadata", p["inputs"] / "inputs.json", "--recipient", job["recipient"],
                   "--language", job["language"], "--gain-db", str(job.get("export", {}).get("gain_db", 0))]
    elif stage == "verify":
        if dry_run:
            print(f"Will verify hashes, frame count and full MP4/MP3 decoding: {p['export']}")
        else:
            verify_export(job, p)
        return
    else:
        raise ValueError(f"Unknown stage: {stage}")
    execute(command, dry_run=dry_run, root=root)


def doctor(*, gpu=False, root=ROOT):
    required = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "git", "uv")}
    print(json.dumps({"python": sys.version.split()[0], "tools": required,
                      "environments": {name: (root / f".venv-{name}/bin/python").is_file() for name in ("audio", "video")},
                      "model_manifests": {name: (root / f"models/{name}/download-manifest.json").is_file()
                                          for name in ("chatterbox-multilingual-v2", "faster-whisper-small", "musetalk")},
                      "note": "Presence check only. Use setup_audio.py --check and setup_video.py --check for detailed checks."}, indent=2))
    if gpu:
        execute(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv"], root=root)
    return 0 if all(required.values()) else 1


def stage_log_path(job, root=ROOT):
    path = paths(job, root)["work"] / "stage-times.jsonl"
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Stage timing log may not be a symlink or leave the repository")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a private job and a neutral example script")
    init.add_argument("name")
    init.add_argument("--language", choices=("fi", "en"), default="fi")
    check = commands.add_parser("doctor", help="Read-only tools/environment presence check")
    check.add_argument("--gpu", action="store_true")
    download = commands.add_parser("download", help="Explicitly download a source excerpt with the pinned local yt-dlp tools")
    download.add_argument("--url", required=True)
    download.add_argument("--start", type=float, required=True)
    download.add_argument("--end", type=float, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status", help="Show which job artifacts exist")
    status.add_argument("job", type=Path)
    run = commands.add_parser("run", help="Run one explicit stage; files are never silently overwritten")
    run.add_argument("job", type=Path)
    run.add_argument("stage", choices=STAGES)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "init":
            print(f"Created {init_job(args.name, args.language)}\nEdit its media paths, credits, intervals and script before running reference.")
        elif args.command == "doctor":
            return doctor(gpu=args.gpu)
        elif args.command == "download":
            command = [sys.executable, ROOT / "scripts/download_clip.py", "--url", args.url,
                       "--start", str(args.start), "--end", str(args.end), "--output", args.output]
            if args.dry_run:
                command.append("--dry-run")
            execute(command)
        else:
            job = load_job(args.job)
            if args.command == "status":
                print(json.dumps({key: {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
                                  for key, path in paths(job).items()}, indent=2))
            else:
                log_path = stage_log_path(job) if not args.dry_run else None
                started = time.monotonic()
                run_stage(job, args.stage, dry_run=args.dry_run)
                if not args.dry_run:
                    elapsed = time.monotonic() - started
                    log_path = stage_log_path(job)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"stage": args.stage, "wall_seconds": elapsed,
                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                            "job_sha256": sha256(args.job)}) + "\n")
                    print(f"Stage {args.stage} finished in {elapsed:.2f} seconds")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    # The standalone helpers also run directly from scripts/. Keep their
    # sibling imports available when the orchestration layer imports validation.
    sys.path.insert(0, str(ROOT / "scripts"))
    raise SystemExit(main())
