# Dubiously Yours

A local command-line workflow for clearly labelled AI parody greetings. Write a short script, audition reference-conditioned speech, then synchronize an existing video and export captions, MP4 and MP3.

The project uses Chatterbox Multilingual V2 for Finnish or English speech, Whisper for alignment, MuseTalk for lip synchronization, and FFmpeg for assembly. Audio runs on CPU; the video workflow uses a separate NVIDIA GPU environment. Source footage supplies head, eye and body movement.

This repository contains code, neutral example scripts and documentation. Bring your own local voice reference and video. Models, recordings, generated media and personal project history are excluded from the public release.

## Download size and local processing

The pinned model payloads total **8.213 GB (7.649 GiB)**, before Python, PyTorch/CUDA packages and caches:

| Download | Payload |
| --- | ---: |
| Chatterbox Multilingual V2 | 3.209 GB |
| Whisper-small for alignment | 0.486 GB |
| MuseTalk and all required supporting models | 4.483 GB |
| Auxiliary tokenizer archive | 0.035 GB |

The existing environments, models and tokenizer cache occupy roughly **15 GiB** before additional caches and media. Budget **25–30 GB free disk** for initial setup as an estimate, then allow more for source videos, decoded frames and outputs; this is not a measured minimum.

On the tested **Ryzen 7 7700X / RTX 3080 Ti 12 GiB / WSL2** machine, a 7.84-second Finnish sample used **34.962 seconds in CPU speech-generation calls** and **51.477 seconds in the MuseTalk GPU subprocess**. Audio used four CPU threads. Model loading and other overhead are separate; these figures are not a complete job time or a guarantee for another machine. See the [measured sizes, hardware and timing limits](docs/PERFORMANCE.md).

## Use it with an AI agent

Give an agent a video link, a passage to use and the greeting you want. The bundled [Dubiously Yours skill](skills/dubiously-yours/SKILL.md) guides it through resource checks, setup, source selection, a short voice audition and the final export. You can direct the result without editing configuration files yourself.

See the [agent guide](docs/AGENT_GUIDE.md) for a ready-to-use prompt and skill installation options. The skill can research alternative models for a specific quality or hardware need while preserving the working setup.

The resource helper can also run independently, without downloading or loading models:

```bash
python3 scripts/check_resources.py --mode video --duration 10 --json
```

It separates measured resources from estimated remaining downloads and working space. A passing report supports a short trial; it is not proof that a full neural render fits.

## Start here

The supported setup target is Linux x86_64, including WSL2, with FFmpeg, Git, `uv` and an NVIDIA GPU for video. Read the [setup guide](docs/SETUP.md) before downloading the model environments.

```bash
python3 scripts/setup_audio.py
python3 scripts/setup_video.py
python3 dub.py doctor
python3 dub.py init my-greeting --language fi
```

The setup commands above print their plans; they do not install anything. `init` creates a local job and example script under `projects/my-greeting/`. Edit `job.json` with your media paths, source information and usable intervals, then edit `script.txt`.

You can use your own recording or explicitly download a source clip with the optional yt-dlp media tools. Install them once, then replace the URL placeholder and select an interval in seconds:

```bash
python3 scripts/setup_media.py --install
python3 dub.py download --url 'YOUTUBE_URL' --start 60 --end 120 \
  --output input/source.mp4
```

Source frame rate is retained. The video stage requires 25 fps; the [workflow guide](docs/WORKFLOW.md) includes an explicit conversion when needed. Network access occurs during requested setup/model downloads and this download command; speech generation and rendering use local assets.

Run stages individually so you can listen before spending time on video:

```bash
python3 dub.py run projects/my-greeting/job.json reference
python3 dub.py run projects/my-greeting/job.json audio
python3 dub.py run projects/my-greeting/job.json align
# Listen to the audio and check the wording before continuing.
python3 dub.py run projects/my-greeting/job.json video
python3 dub.py run projects/my-greeting/job.json render
python3 dub.py run projects/my-greeting/job.json export
python3 dub.py run projects/my-greeting/job.json verify
python3 dub.py status projects/my-greeting/job.json
```

Add `--dry-run` to a `run` command to inspect its commands without loading models. Job paths are relative to the repository; inputs belong under `input/` or `projects/`. A URL in a job is provenance; source downloads require the explicit `download` command.

The [workflow guide](docs/WORKFLOW.md) explains the job format, reference selection, speech groups and review steps. Neutral [Finnish](examples/demo-fi.txt) and [English](examples/demo-en.txt) scripts are included.

## What has been checked

The **generic Finnish CLI pipeline has run successfully from reference preparation through final verification** on a new neutral sample using the installed environments. It produced a 7.84-second, 196-frame, 1280 × 720 / 25 fps export. Both MP4 and MP3 passed full decoding and finite/unclipped audio checks. Automated tests also exercise command/configuration boundaries, release isolation and model-free Finnish/English media export. Sampled frames of earlier local outputs were inspected separately.

These results do not establish a successful fresh installation on another machine or approved speech quality. ASR flagged minor wording differences in the new sample; listening remains necessary. The public bootstrap has separate offline checks, and model-free tests do not prove neural output quality. The English language interface is available, but the new generic Multilingual V2 English path has not received a fresh neural audition.

Voice likeness, names, Finnish pronunciation, comic timing and lip-sync quality need human review. Automatic transcription can mishear correct speech. Generated mouth detail can soften, and the mouth may remain open during a quiet ending. The workflow preserves the intended caption wording, source records and built-in speech watermark; exports carry a visible AI-parody label.

## Documentation

- [Setup](docs/SETUP.md)
- [Using an AI agent](docs/AGENT_GUIDE.md)
- [Download sizes and measured performance](docs/PERFORMANCE.md)
- [Workflow and job format](docs/WORKFLOW.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Developer diary](docs/DEVELOPER_DIARY.md)
- [Building a clean public release](docs/PUBLISHING.md)
- [Credits and dependencies](CREDITS.md)

## Development and licence

```bash
python3 -m unittest discover -s tests -v
python3 scripts/public_release.py check
```

Project code and neutral examples use the [MIT licence](LICENSE). Dependencies, model weights and your source media retain their own terms. See [CREDITS.md](CREDITS.md).
