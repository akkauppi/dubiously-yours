# Downloads, storage and processing time

Measurements below were taken on 30 September 2026. They help estimate local setup and iteration costs; they are observations from one machine, not performance guarantees or minimum requirements.

The [machine-readable measurement snapshot](performance-example.json) records the pinned payload sizes, hardware, timing scope and limits. GB below means 1,000,000,000 bytes; GiB means 1,073,741,824 bytes.

## Model download budget

| Pinned payload | Bytes | GB | GiB |
| --- | ---: | ---: | ---: |
| Chatterbox Multilingual V2, including its small configuration/tokenizer files | 3,208,854,455 | 3.209 | 2.988 |
| Whisper-small for alignment | 486,214,370 | 0.486 | 0.453 |
| MuseTalk 1.5 and all 11 required supporting-model files | 4,482,904,130 | 4.483 | 4.175 |
| Compressed auxiliary tokenizer archive | 34,567,143 | 0.035 | 0.032 |
| **Total** | **8,212,540,098** | **8.213** | **7.649** |

These figures sum the pinned files and auxiliary archive. The MuseTalk total includes its VAE, Whisper-tiny, pose estimation, face parsing, ResNet and face-detector weights. It is not just the main lip-sync checkpoint. The auxiliary tokenizer archive expands to a different installed size.

Python runtimes, PyTorch/CUDA libraries, remaining Python packages, Git source, download caches and HTTP overhead are **additional**. A changed upstream revision may change the payload; this snapshot describes the current pins rather than future releases. Audio setup with models downloads approximately 3.730 GB before its Python packages; adding video brings model payloads to the total above.

## Tested computer

| Component | Measured configuration |
| --- | --- |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 logical CPUs |
| RAM visible to WSL2 | 31.21 GiB |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GiB VRAM |
| Speech execution | CPU, four threads; Python 3.11 / CPU PyTorch |
| Video execution | NVIDIA GPU; separate Python 3.10 / CUDA environment |

This is a tested GPU configuration, not a proven 12 GiB minimum. Peak system RAM and VRAM consumption were not profiled. The source video, face visibility, clip length and competing workloads affect runtime.

## Observed processing times

A new neutral Finnish smoke test produced **7.84 seconds of audio**, comprising **6.84 seconds of generated speech plus 1.00 second of inserted pauses**.

All generic CLI stages completed in the installed environments: reference, audio, alignment, video preparation, GPU render, export and verification. The final video is 196 frames, 1280 × 720 at 25 fps. MP4 and MP3 fully decode, and their audio is finite and unclipped with peaks around 0.455. ASR flagged minor grammar/compound-word differences; this technical pass is not a listening approval. A fresh network installation and a new generic English neural audition remain untested.

| Stage or operation | Observed time | Scope |
| --- | --- | --- |
| Chatterbox model load | 8.291 s | Local model loading inside the speech helper |
| Reference conditioning | 1.069 s | Preparing the voice reference once |
| First speech passage | 18.499 s | The model's generation call |
| Second speech passage | 16.464 s | The model's generation call |
| Speech generation calls, combined | 34.962 s | Unrounded recorded sum; excludes load and conditioning |
| MuseTalk subprocess | 51.477 s | Framework/model initialization, frames, face processing and media decode/encode; excludes wrapper preflight |

The generation calls do **not** represent complete audio-stage wall time. Python imports, file hashing, export work and other overhead were not timed together. Likewise, adding the rows does not produce a measured end-to-end job duration.

The MuseTalk run produced 198 raw inference frames, or 7.92 seconds. The completed export uses 196 frames for the 7.84-second greeting, removing only the two surplus trailing inference frames. An earlier 36.60-second greeting on the same machine took **195.180 seconds** in the MuseTalk subprocess. Loading overhead and content differences mean these observations should not be scaled linearly to predict another clip.

Whisper alignment, source preparation, downloads, final export and full human review do not have a combined timing measurement here. No electricity or monetary cost was measured. This workflow uses local CPU/GPU processing rather than a metered hosted inference API.

New successful `dub.py run` stages write wall time to the ignored `projects/<job-id>/work/stage-times.jsonl`. Use these per-stage records to measure future jobs on your own machine. Earlier runs predate that logger; their missing whole-stage timings have not been reconstructed from partial model timings.

## Installed component footprint

The existing installation occupied approximately the following space, measured with `du -s`:

| Component | Observed allocated storage |
| --- | --- |
| Audio environment | 1.9 GiB |
| Video environment | 5.4 GiB |
| Optional media/download environment | 63 MiB |
| Chatterbox model directory | 3.0 GiB |
| Whisper model directory | 464 MiB |
| MuseTalk model directory | 4.2 GiB |
| Tokenizer cache | 91 MiB |

These are local component footprints, not download-size estimates. Additional package-download caches, source media, decoded video frames, generated outputs and other temporary files need space too. They are not included in the figures above. Separate environments prevent dependency conflicts but increase disk use.

The listed components total roughly **15 GiB** before those additions. As an initial planning allowance, leave **25–30 GB free disk**, then add room for the footage and number of experiments you expect. This allowance is an estimate, not a measured fresh-install minimum.

## Keep experiments inexpensive

Start with one short passage, listen to it, and choose the reference before rendering a full greeting. Retain a useful take rather than regenerating it for every visual change. Use fixed comparison settings so a change of reference or seed is interpretable.

Run models sequentially when GPU memory is shared with other applications. Use a short close-up first, then check the mouth, captions, ending and cut positions. A technically valid export can still need another performance take.

See [setup](SETUP.md) for explicit installation/model downloads and the [workflow](WORKFLOW.md) for stage boundaries. Generation and rendering use local model assets after setup. Network access belongs to the requested setup, weight-fetch or source-download commands.
