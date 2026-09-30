#!/usr/bin/env python3
"""Plan or explicitly install the pinned local MuseTalk inference environment.

Default: print the plan only. --install downloads Python, packages and source,
never model weights. --check verifies the existing installation offline.
Linux x86_64/WSL2, FFmpeg, git, uv and an NVIDIA CUDA-capable driver are required.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/MuseTalk"
ENVIRONMENT = ROOT / ".venv-video"
MODEL_DIR = ROOT / "models/musetalk"
COMMIT = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
UPSTREAM = "https://github.com/TMElyralab/MuseTalk.git"
ORIGINAL_SHA256 = "aaf9c8289c8f07eaf00b87fb11442ac0e1005ac4e2f469966ae3dab2dc4a03aa"
LEGACY_PATCH_SHA256 = "1b5ec1d0534f53abc94542b0dda78d1d0fd2399b674c02b628f22114cd5a84ee"
EARLY_GUARD_SHA256 = "5001c7b4c2b3c5f2e1271ac9795f117e8ab988e42f9960278cee6f21a41c5766"
PYTHON_VERSION = "3.10.20"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patched_inference(source):
    """Apply explicit, anchored changes to the exact pinned upstream file."""
    replacements = [
        ('            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")',
         '            crop_coord_save_path = os.path.join(temp_dir, input_basename+"_coordinates.pkl")'),
        ("            # Extract frames from source video\n", "            # Extract frames from source video\n            save_dir_full = None\n"),
        ("            shutil.rmtree(save_dir_full)\n", "            if save_dir_full is not None:\n                shutil.rmtree(save_dir_full)\n"),
        ('                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"\n                os.system(cmd)',
         '                cmd = ["ffmpeg", "-v", "error", "-n", "-i", video_path, "-start_number", "0", os.path.join(save_dir_full, "%08d.png")]\n                subprocess.run(cmd, check=True)'),
        ('            # Process each frame\n',
         '            # Fail before inference rather than dropping frames or shifting lip timing.\n'
         '            if not frame_list or len(coord_list) != len(frame_list):\n'
         '                raise RuntimeError("Face detection did not return one result for every source frame")\n'
         '            for frame_index, (bbox, image) in enumerate(zip(coord_list, frame_list)):\n'
         '                if image is None or bbox == coord_placeholder:\n'
         '                    raise RuntimeError(f"No usable face in source frame {frame_index}; choose a continuous close-up")\n'
         '                x1, y1, x2, y2 = bbox\n'
         '                if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):\n'
         '                    raise RuntimeError(f"Invalid face bounds in source frame {frame_index}: {bbox}")\n'
         '            # Process each frame\n'),
        ('                except:\n                    continue\n',
         '                except Exception as error:\n                    raise RuntimeError(f"Cannot resize generated face in frame {i}") from error\n'),
        ('                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)',
         '                if not cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame):\n                    raise RuntimeError(f"Cannot write generated frame {i}")'),
        ('            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"',
         '            cmd_img2video = ["ffmpeg", "-n", "-v", "warning", "-r", str(fps), "-f", "image2", "-i", os.path.join(result_img_save_path, "%08d.png"), "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", temp_vid_path]'),
        ('            os.system(cmd_img2video)', '            subprocess.run(cmd_img2video, check=True)'),
        ('            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"',
         '            cmd_combine_audio = ["ffmpeg", "-n", "-v", "warning", "-i", audio_path, "-i", temp_vid_path, output_vid_name]'),
        ('            os.system(cmd_combine_audio)', '            subprocess.run(cmd_combine_audio, check=True)'),
    ]
    text = source.decode("utf-8")
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError("Pinned upstream patch anchor changed; refusing to guess a patch")
        text = text.replace(old, new, 1)
    compile(text, "MuseTalk/scripts/inference.py", "exec")
    return text.encode("utf-8")


def source_state(vendor=VENDOR, *, apply=False):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    if commit != COMMIT:
        raise ValueError(f"MuseTalk checkout must be {COMMIT}; current checkout is {commit}")
    upstream = subprocess.check_output(["git", "show", f"{COMMIT}:scripts/inference.py"], cwd=vendor)
    if digest(upstream) != ORIGINAL_SHA256:
        raise ValueError("Pinned inference source hash differs")
    expected = patched_inference(upstream)
    path = vendor / "scripts/inference.py"
    actual = path.read_bytes()
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=vendor, text=True).splitlines()
    if any(name != "scripts/inference.py" for name in changed):
        raise ValueError(f"Unexpected tracked MuseTalk modifications: {changed}")
    if actual != expected:
        if not apply or digest(actual) not in (ORIGINAL_SHA256, LEGACY_PATCH_SHA256, EARLY_GUARD_SHA256):
            raise ValueError("MuseTalk needs the checked local patch (setup_video.py --patch-only); unknown local edits are never replaced")
        backup = ROOT / "local/vendor-backups/MuseTalk" / digest(actual) / "inference.py"
        if backup.exists():
            if backup.read_bytes() != actual:
                raise ValueError(f"Existing source backup differs: {backup}")
        else:
            backup.parent.mkdir(parents=True, exist_ok=True)
            with backup.open("xb") as stream:
                stream.write(actual)
        path.write_bytes(expected)
    return {"repository": UPSTREAM, "commit": COMMIT, "inference_sha256": digest(expected),
            "patch": "Checked FFmpeg argv, complete face/frame results, private per-run coordinate cache, and image-input cleanup"}


def ensure_link(link, target):
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise FileExistsError(f"Refusing to replace {link}; expected a link to {target}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target == MODEL_DIR)


def runtime_environment():
    env = os.environ.copy()
    paths = [str(VENDOR / "musetalk/utils"), str(VENDOR)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update({"PYTHONPATH": os.pathsep.join(paths), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": str(ROOT / ".cache/huggingface"),
                "TORCH_HOME": str(ROOT / ".cache/torch"), "XDG_CACHE_HOME": str(ROOT / ".cache"),
                "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
                "NUMBA_CACHE_DIR": str(ROOT / ".cache/numba-video"), "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return env


def check_installation():
    source = source_state()
    if (VENDOR / "models").resolve() != MODEL_DIR.resolve():
        raise ValueError("MuseTalk/models link does not point to the project model directory")
    python = ENVIRONMENT / "bin/python"
    subprocess.run(["uv", "--cache-dir", str(ROOT / ".cache/uv"), "pip", "check",
                    "--python", str(python), "--offline"], check=True)
    code = """import importlib,json,sys
assert sys.version_info[:2] == (3,10), sys.version
modules=['torch','torchvision','torchaudio','diffusers','transformers','mmcv.ops','mmengine','mmdet','mmpose.apis','musetalk.utils.audio_processor','musetalk.utils.face_parsing','face_detection']
for name in modules: importlib.import_module(name)
import torch
assert torch.__version__ == '2.0.1+cu118', torch.__version__
print(json.dumps({'python':sys.version,'torch':torch.__version__,'imports':modules,'gpu_inference':'not run'}))
"""
    subprocess.run([str(python), "-c", code], cwd=VENDOR, env=runtime_environment(), check=True)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--install", action="store_true", help="Download/install pinned source and packages, never weights")
    action.add_argument("--check", action="store_true", help="Offline source/dependency/import checks, no model or GPU run")
    action.add_argument("--patch-only", action="store_true", help="Apply the verified local source patch and links without downloads")
    args = parser.parse_args()
    cache = ROOT / ".cache/video-setup"
    config = cache / "uv.toml"
    env = os.environ.copy()
    env.update({"UV_PYTHON_INSTALL_DIR": str(ROOT / ".cache/uv-python"), "UV_CACHE_DIR": str(ROOT / ".cache/uv")})
    commands = [
        ["uv", "python", "install", PYTHON_VERSION],
        ["uv", "venv", "--python", PYTHON_VERSION, str(ENVIRONMENT)],
        ["uv", "--config-file", str(config), "pip", "sync", "--python", str(ENVIRONMENT / "bin/python"),
         "--default-index", "https://pypi.org/simple", "--index", "https://download.pytorch.org/whl/cu118",
         "--index-strategy", "unsafe-best-match", str(ROOT / "requirements-video.lock.txt")],
        ["git", "clone", "--no-checkout", UPSTREAM, str(VENDOR)],
        ["git", "-C", str(VENDOR), "checkout", "--detach", COMMIT],
    ]
    if not (args.install or args.check or args.patch_only):
        print(json.dumps({"platform": "Linux x86_64 / WSL2; NVIDIA driver required for rendering",
            "commands": commands, "notes": ["Existing environment/checkouts are reused only if compatible; unknown source edits are refused",
            "Installs exact runtime versions from requirements-video.lock.txt; package archive hashes are not locked",
            "Applies checked local inference guard patch; links model directory and local S3FD cache",
            "Model weights are a separate explicit fetch_video_models.py command; no model licence or recording rights are granted by setup"]}, indent=2))
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("The pinned MMCV/CUDA wheel set supports Linux x86_64/WSL2 only")
    for binary in (("git", "uv", "ffmpeg", "ffprobe") if not args.patch_only else ("git",)):
        if not shutil.which(binary):
            parser.error(f"Install {binary} before running setup")
    if args.check:
        print(json.dumps(check_installation(), indent=2))
        return
    if args.install:
        cache.mkdir(parents=True, exist_ok=True)
        config.write_text('[extra-build-dependencies]\nchumpy = ["pip==25.0.1"]\n')
        subprocess.run(commands[0], env=env, check=True)
        if not (ENVIRONMENT / "pyvenv.cfg").exists():
            if ENVIRONMENT.exists():
                raise FileExistsError(f"Refusing to replace existing non-venv {ENVIRONMENT}")
            subprocess.run(commands[1], env=env, check=True)
        existing_version = subprocess.check_output([str(ENVIRONMENT / "bin/python"), "-c",
            "import platform; print(platform.python_version())"], text=True).strip()
        if existing_version != PYTHON_VERSION:
            raise ValueError(f"Existing video environment uses Python {existing_version}; expected {PYTHON_VERSION}. Preserve it and choose a clean checkout")
        subprocess.run(commands[2], env=env, check=True)
        if not VENDOR.exists():
            VENDOR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(commands[3], env=env, check=True)
            subprocess.run(commands[4], env=env, check=True)
    source = source_state(apply=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ensure_link(VENDOR / "models", MODEL_DIR)
    ensure_link(ROOT / ".cache/torch/hub/checkpoints/s3fd-619a316812.pth", MODEL_DIR / "s3fd/s3fd-619a316812.pth")
    if args.install:
        check_installation()
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
              "requirements_sha256": digest((ROOT / "requirements-video.lock.txt").read_bytes()),
              "python": PYTHON_VERSION, "model_weights": "not downloaded by setup", "gpu_inference": "not run"}
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "setup-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
