# Setup

The supported setup target is **Linux x86_64, including WSL2**. The audio and video environments are separate because their Python and PyTorch requirements differ. Model execution is local after the explicit setup/download steps.

The generic Finnish CLI has completed reference preparation, CPU synthesis, alignment, GPU lip synchronization, export and verification in the installed environments. The new bootstrap scripts have not yet been validated by a complete fresh installation on another machine. Other operating systems and architectures are untested.

## System tools

An agent can guide the setup using the bundled skill; see the [agent guide](AGENT_GUIDE.md). Before installation or a new render, run `python3 scripts/check_resources.py --mode video --duration 10 --json` with the intended greeting length. Use `--mode audio` for speech only. This read-only report checks the actual storage locations and current memory/device visibility, while separating estimates from measurements. It does not replace the environment and model integrity checks below.

Provide Python **3.11 or later** as `python3` for the top-level CLI, plus Git, FFmpeg/`ffprobe` and `uv`. The CLI's Python is separate from the managed model environments. FFmpeg needs H.264, AAC, MP3 and libass subtitle support. For video, provide a working NVIDIA driver accessible inside Linux or WSL2.

```bash
python3 --version
git --version
uv --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

The bootstrap uses `uv` to select managed Python versions. A 12 GiB NVIDIA GPU has been used for the local video workflow; that is a tested configuration, not a demonstrated minimum. The pinned model downloads total **8.213 GB**, excluding Python/PyTorch/CUDA packages. Audio assets account for approximately 3.730 GB of that total. The existing environments/models occupy roughly 15 GiB before additional caches and media; **25–30 GB free disk** is a planning allowance for first setup, not a measured requirement. Allow more for decoded video frames and outputs. See the [exact payload sizes and observed runtimes](PERFORMANCE.md).

## Optional source downloads: yt-dlp

Skip this environment if you already have local recordings. To download a chosen YouTube interval, inspect the plan and explicitly install the pinned media tools:

```bash
python3 scripts/setup_media.py
python3 scripts/setup_media.py --install
python3 scripts/setup_media.py --check
python3 dub.py download --url 'YOUTUBE_URL' --start 60 --end 120 \
  --output input/source.mp4
```

Replace `YOUTUBE_URL` with the source page. Default setup prints a plan; `--install` uses the network to create `.venv-media` with pinned Python 3.12.3 and packages, plus Node.js 22.13.0 when that exact runtime is not already available. `--check` checks the local versions without downloading. There are no model weights in this environment.

The downloader writes the video, extractor metadata and a source record containing the requested/reported interval, hashes, duration, frame rate and command. It refuses existing outputs and publishes the finished files only after decoding/duration checks. It does not normalize frame rate or resolution. The video workflow needs 25 fps, so use the explicit conversion in [WORKFLOW.md](WORKFLOW.md) if the original differs. The new public downloader has not been validated by a fresh live network download during this packaging work.

## Audio: Python 3.11 and CPU PyTorch

First inspect the setup plan:

```bash
python3 scripts/setup_audio.py
```

Install only when ready for environment creation and package downloads:

```bash
python3 scripts/setup_audio.py --install --models
python3 scripts/setup_audio.py --check
```

`--install` creates `.venv-audio` with Python 3.11.15 and the pinned audio lock. `--models` downloads the pinned Chatterbox Multilingual V2, tokenizer assets and Whisper-small weights. To download models later into an existing validated environment, run `python3 scripts/setup_audio.py --models`.

The audio stack uses `chatterbox-tts==0.1.7`, `torch==2.6.0+cpu` and `torchaudio==2.6.0+cpu`. The CPU PyTorch packages come from the official CPU wheel index; the remaining pinned packages come from PyPI. Preserve `setuptools==80.9.0`: the installed Perth version uses its legacy `pkg_resources` API.

| Asset | Pinned revision |
| --- | --- |
| Chatterbox Multilingual V2 | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |
| Whisper-small | `536b0662742c02347bc0e980a01041f333bce120` |

Tokenizer assets include a local PKUSEG cache and a nested Hugging Face cache for the Cangjie mapping, even for Finnish/English jobs. The setup and generation helpers configure these paths; do not substitute a newer upstream loader or weight variant without checking compatibility.

The audio lock records the existing working environment. Its exact package versions are pinned, but wheels are not hash-locked. `--check` checks local requirements and assets; it does not generate speech.

## Video: Python 3.10 and CUDA PyTorch

Inspect the plan, then install the separate environment:

```bash
python3 scripts/setup_video.py
python3 scripts/setup_video.py --install
python3 scripts/setup_video.py --check
.venv-video/bin/python scripts/fetch_video_models.py
.venv-video/bin/python scripts/fetch_video_models.py --verify-only
```

The installer targets Python 3.10.20 in `.venv-video`, with Torch 2.0.1 + CUDA 11.8, torchvision 0.15.2, torchaudio 2.0.2 and a prebuilt MMCV 2.0.1 wheel. MuseTalk is pinned at commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`. The local patch adds quoted subprocess paths, missing-face/frame failures and cleanup protection.

Model downloads are explicit in `fetch_video_models.py`; `--verify-only` checks local hashes without downloading. The source checkout lives under `vendor/`, and model assets live under `models/`. Neither is included in the public repository.

`setup_video.py --check` is an offline dependency/checkout check, not GPU inference. Confirm device access separately:

```bash
python3 dub.py doctor
python3 dub.py doctor --gpu
```

A visible device or successful package check does not establish that an entire render succeeds. Start with a short clip and inspect the result.

## Offline use and reusable inputs

After dependencies, tokenizer data and weights are present, generation and rendering use local assets. Setup, explicit model fetches and the optional `dub.py download` command require network access. The job stages themselves consume local media; entering a URL in a job does not trigger a download.

Create a job with `python3 dub.py init my-greeting --language fi`, then follow the [workflow](WORKFLOW.md). Keep environments separate and preserve download manifests, source credits and successful run records.
