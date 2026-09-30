# Credits

Dubiously Yours combines existing local media tools with project-specific orchestration, checks and documentation. The project code and neutral example scripts are distributed under the [MIT licence](LICENSE).

## Tools used by the workflow

| Component | Role | Upstream source |
| --- | --- | --- |
| Chatterbox / Chatterbox Multilingual V2 | Reference-conditioned speech synthesis | [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) |
| Chatterbox model weights | Pinned multilingual speech model | [Model repository](https://huggingface.co/ResembleAI/chatterbox) |
| Perth | Built-in speech watermarking used by Chatterbox | [Resemble AI Perth](https://github.com/resemble-ai/Perth) |
| Whisper and faster-whisper | Independent transcription and word alignment | [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| MuseTalk | Lip synchronization on existing video frames | [MuseTalk](https://github.com/TMElyralab/MuseTalk) |
| PyTorch and associated libraries | Local model execution | [PyTorch](https://pytorch.org/) |
| FFmpeg / ffprobe | Media preparation, encoding and verification | [FFmpeg](https://ffmpeg.org/) |
| yt-dlp | Explicit source-clip downloads and extractor metadata | [yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Node.js | Optional JavaScript runtime for source extraction | [Node.js](https://nodejs.org/) |
| uv | Isolated Python environment setup | [uv](https://github.com/astral-sh/uv) |

MuseTalk's upstream checkout includes its MIT notice and notices for its additional components. The setup retains the upstream licence files. Its model stack includes a VAE, Whisper features, pose estimation and face parsing; these are separate dependencies, not original Dubiously Yours models. Fetch helpers record pinned sources and file hashes.

Python package versions are recorded in the audio and video lock files. The locks are version pins, not a claim that every downloaded wheel has been independently audited or hash-locked.

## Your recordings and exports

No voice recording, photograph, interview clip, trained personal voice or generated likeness is supplied in this public repository. The neutral example text was written for this project and contains no real recipient or source-speaker identity.

Job configuration records the origin and credit for each input. A source can be an HTTP(S) page or an identifier such as `urn:recording:my-video` for your own recording. Export credits identify the video source and the changes made, and the provenance files retain the voice reference and hashes. Fill those fields with meaningful attribution before export.

The project's MIT licence applies to its own code and examples. It does not relicense dependencies, model weights or recordings supplied by the user. Keep the notices and attribution that apply to the assets you use.

## Disclosure

Speech uses Chatterbox's built-in Perth watermark. The workflow does not disable it or claim an independently verified watermark detection result. Video exports include a visible AI-parody label and source credit; media metadata identifies the generated greeting as fictional.
