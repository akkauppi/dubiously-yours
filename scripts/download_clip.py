#!/usr/bin/env python3
"""Download one timed HTTP(S) video excerpt with pinned yt-dlp and provenance.

Native frame timing/resolution are retained; this does not prepare 25 fps
MuseTalk input. --dry-run validates the request and prints an argv plan only.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlparse

from setup_media import ROOT, ENVIRONMENT, NODE_LOCAL, check_environment, find_node, sha256

FORMAT = "bv[vcodec^=avc1]+ba/b[vcodec^=avc1]/bv+ba/b"


def validate_request(url, start, end, output):
    parsed = urlparse(url)
    if (parsed.scheme not in ("https", "http") or not parsed.netloc or not parsed.hostname
            or parsed.username or parsed.password or any(c.isspace() for c in url)):
        raise ValueError("Supply an HTTP(S) video URL without embedded credentials or whitespace")
    if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
        raise ValueError("Clip times must be finite and satisfy 0 <= start < end")
    if round(start, 9) >= round(end, 9):
        raise ValueError("Clip interval is smaller than supported timestamp precision")
    if output.suffix != ".mp4" or output.stem in ("", ".", ".."):
        raise ValueError("Output must be a named .mp4 file")
    if any(ord(c) < 32 or ord(c) == 127 for c in str(output)):
        raise ValueError("Output path cannot contain control characters")
    # Do not resolve a final symlink away before checking it for collisions.
    output = output.absolute()
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError("Output parent is not a directory")
    if output.parent.exists():
        prefix = output.stem + "."
        collisions = [p for p in output.parent.iterdir() if p.name.startswith(prefix)]
        if collisions:
            raise FileExistsError("Refusing existing output/sidecar/partial files: " + ", ".join(str(p) for p in collisions))
    return output


def timestamp(seconds):
    return f"{seconds:.9f}".rstrip("0").rstrip(".")


def build_command(url, start, end, staging_output, node, *, python=None):
    # yt-dlp output paths are templates: escape literal percent signs so user
    # filenames cannot become extra template fields or change the destination.
    template = str(staging_output).replace("%", "%%")
    return [str(python or ENVIRONMENT / "bin/python"), "-m", "yt_dlp", "--ignore-config",
            "--no-plugin-dirs", "--no-remote-components", "--no-playlist", "--no-overwrites",
            "--no-continue", "--no-js-runtimes", "--js-runtimes", f"node:{node}",
            "--cache-dir", str(ROOT / ".cache/yt-dlp"), "--format", FORMAT,
            "--merge-output-format", "mp4", "--remux-video", "mp4", "--write-info-json",
            "--download-sections", f"*{timestamp(start)}-{timestamp(end)}", "--force-keyframes-at-cuts",
            "--downloader-args", "ffmpeg_o:-fps_mode passthrough -threads 4",
            "--output", template, "--", url]


def inspect_clip(video, info, start, end):
    # Requested timeline and actual output timing are separate facts. Reject an
    # extractor silently shortening/rebasing the requested interval.
    for field, expected in (("section_start", start), ("section_end", end)):
        actual = info.get(field)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Extractor {field}={actual!r} differs from requested {expected}; inspect the staged files")
    if info.get("_type", "video") not in ("video", None):
        raise ValueError("The URL did not identify one video; playlist downloads are not accepted")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                               "-of", "json", str(video)], text=True))
    streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError("Downloaded clip must contain exactly one video stream")
    stream = streams[0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Downloaded video has no valid frame rate")
    expected_duration = end - start
    video_duration = float(stream.get("duration", probe["format"]["duration"]))
    container_duration = float(probe["format"]["duration"])
    tolerance = max(0.12, 2 / fps)
    if not all(math.isfinite(t) and t > 0 and abs(t - expected_duration) <= tolerance
               for t in (video_duration, container_duration)):
        raise ValueError(f"Clip timing differs from requested {expected_duration:.6f}s: video={video_duration:.6f}s, container={container_duration:.6f}s")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    return {"ffprobe": probe, "requested_duration_seconds": expected_duration,
            "video_duration_seconds": video_duration, "container_duration_seconds": container_duration,
            "duration_tolerance_seconds": tolerance, "source_reported_fps": info.get("fps"),
            "output_avg_frame_rate": stream["avg_frame_rate"], "output_r_frame_rate": stream["r_frame_rate"],
            "frame_rate_conversion_requested": False, "full_decode": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--start", type=float, required=True, help="Start in source timeline seconds")
    parser.add_argument("--end", type=float, required=True, help="Exclusive end in source timeline seconds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="No network or output writes; print validated argv plan")
    args = parser.parse_args()
    output = validate_request(args.url, args.start, args.end, args.output)
    info_output, provenance_output = output.with_suffix(".info.json"), output.with_suffix(".source.json")
    if args.dry_run:
        try:
            node = find_node()
        except FileNotFoundError:
            node = NODE_LOCAL
        planned = output.parent / f".{output.stem}-download-STAGING" / output.name
        print(json.dumps({"dry_run": True, "command": build_command(args.url, args.start, args.end, planned, node),
            "output": str(output), "info": str(info_output), "provenance": str(provenance_output),
            "source_interval_seconds": [args.start, args.end],
            "frame_rate_policy": "Keep source timestamps; no 25 fps conversion or resizing",
            "prerequisites": "Run setup_media.py --check separately to validate local executables/packages"}, indent=2))
        return
    environment = check_environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.stem}-download-", dir=output.parent))
    staging_output = stage / output.name
    command = build_command(args.url, args.start, args.end, staging_output, Path(environment["node"]))
    started = time.monotonic()
    print(json.dumps({"staging_directory": str(stage), "source_interval_seconds": [args.start, args.end]}, ensure_ascii=False), flush=True)
    try:
        subprocess.run(command, check=True)
        staging_info = staging_output.with_suffix(".info.json")
        if not staging_output.is_file() or not staging_info.is_file():
            raise FileNotFoundError("yt-dlp did not create the requested MP4 and metadata")
        info = json.loads(staging_info.read_text(encoding="utf-8"))
        checks = inspect_clip(staging_output, info, args.start, args.end)
        record = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_url": args.url,
            "webpage_url": info.get("webpage_url", args.url), "source_title": info.get("title"), "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("channel"), "source_id": info.get("id"),
            "requested_source_interval_seconds": [args.start, args.end],
            "extractor_source_interval_seconds": [info["section_start"], info["section_end"]],
            "source_offset_seconds": args.start, "output": str(output), "sha256": sha256(staging_output),
            "bytes": staging_output.stat().st_size, "info_json": str(info_output), "info_json_sha256": sha256(staging_info),
            "checks": checks, "environment": environment, "command": command,
            "download_and_validation_seconds": round(time.monotonic() - started, 3),
            "processing": "AVC preferred with fallback; clip re-encoded at requested cuts; frame timestamps passed through; no resizing or requested fps conversion",
            "rights": info.get("license"), "generator_sha256": sha256(Path(__file__))}
        staging_provenance = stage / provenance_output.name
        with staging_provenance.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        # Hard links publish each completed file with atomic no-replace
        # semantics. The staging directory is on the same filesystem.
        validate_request(args.url, args.start, args.end, output)
        for src, dest in ((staging_output, output), (staging_info, info_output), (staging_provenance, provenance_output)):
            os.link(src, dest)
        shutil.rmtree(stage)
    except Exception:
        print(f"Download did not complete; staged files, if any, remain at {stage}", flush=True)
        raise
    print(json.dumps({"output": str(output), "info": str(info_output), "provenance": str(provenance_output),
                      "sha256": record["sha256"], "full_decode": "passed"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
