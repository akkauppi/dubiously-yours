#!/usr/bin/env python3
"""Prepare forward interview shots and a silent inference tail for MuseTalk."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import wave

from render_portrait import sha256, valid_source_identifier

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, required=True)
    parser.add_argument("--source-offset", type=float, required=True)
    parser.add_argument("--audio-metadata", type=Path, required=True)
    parser.add_argument("--ranges", required=True, help="JSON array of [start,end] seconds within the downloaded clip")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.source_offset) or args.source_offset < 0:
        parser.error("source-offset must be a finite nonnegative time")
    source, info_path, meta_path, out = [p.resolve() for p in
        (args.source_video, args.source_info, args.audio_metadata, args.output_dir)]
    video, audio, manifest = [out / name for name in ("source-video.mp4", "inference-audio.wav", "inputs.json")]
    for path in (video, audio, manifest):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not valid_source_identifier(info.get("webpage_url")):
        raise ValueError("Source info requires webpage_url as HTTP(S) URL or URN, such as urn:recording:my-video")
    for field in ("title", "uploader"):
        if not isinstance(info.get(field), str) or not info[field].strip() or any(c in info[field] for c in "\r\n"):
            raise ValueError(f"Source info requires a nonempty single-line {field}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    original_audio = Path(metadata["output"])
    if not original_audio.is_absolute():
        original_audio = ROOT / original_audio
    original_digest = sha256(original_audio)
    if original_digest != metadata["sha256"]:
        raise ValueError("Greeting audio hash differs from metadata")
    with wave.open(str(original_audio), "rb") as stream:
        if (stream.getcomptype(), stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != ("NONE", 1, 2, 24000):
            raise ValueError("Expected mono 24 kHz PCM16 greeting")
        samples = stream.getnframes()
        pcm = stream.readframes(samples)
    if samples <= 0 or len(pcm) != samples * 2:
        raise ValueError("Greeting audio is empty or incomplete")
    if not math.isclose(metadata["duration_seconds"], samples / 24000, rel_tol=0, abs_tol=1 / 24000):
        raise ValueError("Greeting duration differs from its metadata")
    # Two extra silent frames compensate for MuseTalk's short audio-feature
    # window. Export removes only surplus video and uses the original audio.
    target_frames = (samples + 959) // 960
    inference_frames = target_frames + 2
    padding_samples = inference_frames * 960 - samples
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
        "-of", "json", str(source)], text=True))
    source_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    if source_stream["avg_frame_rate"] != "25/1" or source_stream["r_frame_rate"] != "25/1":
        raise ValueError("Source must already be constant 25 fps")
    ranges = json.loads(args.ranges)
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Select at least one forward interval")
    frame_ranges = []
    for interval in ranges:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Each interval must be a [start,end] pair")
        start, end = interval
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (start, end)):
            raise ValueError("Interval times must be numbers")
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid forward interval")
        a, b = round(start * 25), round(end * 25)
        if any(abs(t * 25 - f) > 1e-6 for t, f in ((start, a), (end, b))):
            raise ValueError("Intervals must lie on 25 fps frame boundaries")
        if b > int(source_stream["nb_frames"]):
            raise ValueError("Interval extends past source frames")
        if frame_ranges and a < frame_ranges[-1][1]:
            raise ValueError("Intervals must be chronological and nonoverlapping; no source repetition is implicit")
        frame_ranges.append((a, b))
    if sum(b - a for a, b in frame_ranges) != inference_frames:
        raise ValueError(f"Selected intervals must total {inference_frames} frames including the inference tail")
    filters = [f"[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]"
               for i, (a, b) in enumerate(frame_ranges)]
    filters.append("".join(f"[v{i}]" for i in range(len(ranges))) +
                   f"concat=n={len(ranges)}:v=1:a=0,scale=1280:720,setsar=1[v]")
    command = ["ffmpeg", "-v", "error", "-n", "-i", str(source), "-filter_complex_threads", "4",
               "-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-r", "25",
               "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "4", str(video)]
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    with audio.open("xb") as handle, wave.open(handle, "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm + bytes(padding_samples * 2))
    actual = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
        "-of", "json", str(video)], text=True))["streams"][0]
    if int(actual["nb_frames"]) != inference_frames:
        raise RuntimeError("Prepared video frame count differs from plan")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_url": info["webpage_url"], "source_title": info["title"], "uploader": info["uploader"],
              "source_clip": str(source), "source_clip_sha256": sha256(source),
              "source_info": str(info_path), "source_info_sha256": sha256(info_path),
              "downloaded_source_offset_seconds": args.source_offset,
              "source_clip_duration_seconds": int(source_stream["nb_frames"]) / 25,
              "used_intervals_seconds": [[args.source_offset + a / 25, args.source_offset + b / 25]
                                         for a, b in frame_ranges],
              "local_frame_intervals": frame_ranges, "sha256": sha256(video), "output_video": str(video),
              "target_frames": target_frames, "inference_frames": inference_frames, "fps": 25,
              "original_audio": str(original_audio), "original_audio_sha256": original_digest,
              "audio_metadata": str(meta_path), "audio_metadata_sha256": sha256(meta_path),
              "inference_audio": str(audio), "inference_audio_sha256": sha256(audio),
              "inference_padding_samples": padding_samples,
              "processing": "Forward close-ups concatenated at normal speed; 1280x720; original audio removed; no loop or reversal",
              "audio_processing": "Original PCM unchanged followed by zero samples for MuseTalk; final exporter uses original greeting audio",
              "source_rights": info.get("rights", info.get("license", "Recording rights are not established by this tool")),
              "command": command, "full_video_decode": "passed"}
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"video": str(video), "audio": str(audio), "manifest": str(manifest),
                      "target_frames": target_frames, "inference_frames": inference_frames}))


if __name__ == "__main__":
    main()
