#!/usr/bin/env python3
"""Run pinned MuseTalk locally on an explicitly supplied video, WAV and provenance.

Inference is offline and never manages other GPU workloads. By default the
source must cover the audio so upstream cannot silently reverse-cycle frames.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse
import wave

from fetch_video_models import verify_models
from setup_video import VENDOR, MODEL_DIR, runtime_environment, source_state

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_source_identifier(value):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        return False
    parsed = urlparse(value)
    return bool((parsed.scheme in ("http", "https") and parsed.netloc) or
                (parsed.scheme == "urn" and parsed.path))


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                               "-of", "json", str(path)], text=True))


def video_details(path):
    result = probe(path)
    streams = [s for s in result["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError("Supply one prepared video stream")
    stream = streams[0]
    if stream["r_frame_rate"] != "25/1" or stream["avg_frame_rate"] != "25/1":
        raise ValueError("Prepare a constant 25 fps video before rendering")
    frames = int(stream["nb_frames"])
    if frames <= 0 or stream["width"] % 2 or stream["height"] % 2:
        raise ValueError("Video must have frames and even dimensions")
    return {"frames": frames, "width": stream["width"], "height": stream["height"], "fps": 25}


def audio_details(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getcomptype(), stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != ("NONE", 1, 2, 24000):
            raise ValueError("Supply a mono PCM16 24 kHz WAV")
        frames = stream.getnframes()
        if frames <= 0 or len(stream.readframes(frames)) != frames * 2:
            raise ValueError("Audio is empty or incomplete")
    return {"samples": frames, "sample_rate": 24000, "duration_seconds": frames / 24000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Render only the first 0.6 seconds")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--source-metadata", type=Path, required=True, help="JSON identifying and hashing the supplied video")
    parser.add_argument("--preflight", action="store_true", help="Offline file/hash checks only; no inference or GPU allocation")
    parser.add_argument("--allow-frame-cycle", action="store_true", help="Explicitly allow upstream forward/reverse repetition for shorter input")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("batch-size must be between 1 and 32")
    for binary in ("ffmpeg", "ffprobe", "git"):
        if not shutil.which(binary):
            parser.error(f"Required executable is missing: {binary}")
    source, audio, out, source_metadata = [p.resolve() for p in
        (args.input_video, args.audio, args.output_dir, args.source_metadata)]
    output = out / "v15/portrait-lipsync.mp4"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing to reuse nonempty render directory: {out}")
    source_record = json.loads(source_metadata.read_text(encoding="utf-8"))
    if source_record.get("sha256") != sha256(source):
        raise ValueError("Source metadata hash does not match the input video")
    if not valid_source_identifier(source_record.get("source_url")):
        raise ValueError("Source metadata needs an HTTP(S) URL or URN source identifier")
    source_info, audio_info = video_details(source), audio_details(audio)
    if source_record.get("inference_frames", source_record.get("frames", source_info["frames"])) != source_info["frames"]:
        raise ValueError("Source frame count differs from provenance")
    original_audio_hash = sha256(audio)
    if source_record.get("inference_audio_sha256", original_audio_hash) != original_audio_hash:
        raise ValueError("Inference WAV differs from the prepared source manifest")
    duration = min(audio_info["duration_seconds"], 0.6) if args.preview else audio_info["duration_seconds"]
    needed_frames = math.ceil(duration * 25 - 1e-9)
    if source_info["frames"] < needed_frames and not args.allow_frame_cycle:
        raise ValueError(f"Video has {source_info['frames']} frames but audio needs {needed_frames}; prepare more source footage")
    source_code = source_state()
    manifests = verify_models(MODEL_DIR)
    if (VENDOR / "models").resolve() != MODEL_DIR.resolve():
        raise ValueError("MuseTalk models link differs; run setup_video.py --patch-only")
    detector_cache = ROOT / ".cache/torch/hub/checkpoints/s3fd-619a316812.pth"
    detector_model = MODEL_DIR / "s3fd/s3fd-619a316812.pth"
    if not detector_cache.is_file() or sha256(detector_cache) != sha256(detector_model):
        raise ValueError("Local S3FD cache is absent or differs; run setup_video.py --patch-only; rendering will not download it")
    preflight = {"source_code": source_code, "models": manifests, "video": source_info,
                 "audio": audio_info, "required_frames": needed_frames,
                 "frame_cycle_allowed": args.allow_frame_cycle, "offline": True}
    if args.preflight:
        print(json.dumps(preflight, indent=2))
        return
    env = runtime_environment()
    # Set cache/offline flags before any model-related import in this process.
    os.environ.update({k: v for k, v in env.items() if k in
        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "TORCH_HOME", "HF_HUB_DISABLE_TELEMETRY")})
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check the NVIDIA driver and this process's GPU access")
    free, _ = torch.cuda.mem_get_info()
    if free < 4 * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.2f} GiB CUDA memory free; need at least 4 GiB; no other workload was changed")
    out.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview = out / "preview-audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(audio), "-t", "0.6", "-c:a", "pcm_s16le", str(preview)], check=True)
        audio = preview
    config = out / "inference.json"
    with config.open("x", encoding="utf-8") as stream:
        json.dump({"parody": {"video_path": str(source), "audio_path": str(audio), "result_name": output.name}}, stream, indent=2)
    command = [sys.executable, "-m", "scripts.inference", "--inference_config", str(config), "--result_dir", str(out),
               "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
               "--version", "v15", "--fps", "25", "--batch_size", str(args.batch_size), "--use_float16"]
    started = time.monotonic()
    print(f"Rendering; GPU {torch.cuda.get_device_name(0)}, {free / 1024**3:.2f} GiB free", flush=True)
    log = out / "render.log"
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=VENDOR, env=env, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if result.returncode or not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError(f"MuseTalk failed or output is missing. Inspect {log}; choose a new output directory for a retry")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    actual = video_details(output)
    # Upstream feature framing can omit one trailing frame; missing face/frame
    # results are separately fatal in the verified source patch.
    expected_floor = math.floor(duration * 25 + 1e-9)
    if not expected_floor - 1 <= actual["frames"] <= needed_frames:
        raise RuntimeError("Rendered frame count differs from the expected audio feature window")
    if (actual["width"], actual["height"]) != (source_info["width"], source_info["height"]):
        raise RuntimeError("Rendered dimensions differ from the prepared source")
    record = {"label": "AI-generated fictional parody; not an authentic statement", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_commit": source_code["commit"], "source_code": source_code, "command": command,
        "working_directory": str(VENDOR), "model_manifests": manifests,
        "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "torchaudio", "diffusers", "transformers", "mmcv", "mmpose")},
        "input_video": str(source), "input_video_sha256": sha256(source), "input_audio": str(audio), "input_audio_sha256": sha256(audio),
        "source_metadata": str(source_metadata), "source_metadata_sha256": sha256(source_metadata), "source_provenance": source_record,
        "scope": "Lip synchronization applied to the supplied video; base movement comes from that input",
        "frame_cycle_allowed": args.allow_frame_cycle, "gpu": torch.cuda.get_device_name(0), "gpu_free_bytes_before_render": free,
        "render_seconds": elapsed, "decode_check": "passed", "ffprobe": probe(output), "preflight": preflight,
        "output": str(output), "output_sha256": sha256(output)}
    with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "render_seconds": elapsed, "frames": actual["frames"]}), flush=True)


if __name__ == "__main__":
    main()
