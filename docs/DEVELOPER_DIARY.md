# Developer diary

This public diary records the method and its limits. Personal scripts, source identities, recordings, recipient names and generated greetings are excluded. The earlier detailed local diary remains archived outside the public release.

## 2026-09-25 — Start with a short local voice experiment

The initial goal was a short, recognizably fictional greeting made mainly with local tools. An English zero-shot speech model provided a fast way to test the reference, intelligibility and comic delivery without training a dedicated voice model.

A small reference and a longer concatenated reference were compared with the same script and seed. The longer version was preferred by the listener in that comparison. More reference material did not mean every model component used the full recording.

Technical checks recorded finite samples, clipping, duration and independent transcription. Listening determined the voice preference. A transcription success was not treated as a likeness assessment.

## 2026-09-26 — Audio first, then motion

Still-portrait lip synchronization worked technically but looked too static. Motion transfer and explicit blinking were explored locally. These showed why facial motion and mouth synchronization are distinct tasks.

A longer greeting was assembled from short speech groups and pauses. Pronunciation experiments separated intended displayed spelling from internal synthesis hints. Exact captions, provenance and the selected audio were preserved through export.

Some generated mouths remained open in quiet endings, and facial detail softened. Sampled frames and full media decoding were useful checks, but neither replaced watching and listening to the complete performance.

## 2026-09-30 — Finnish speech and real video

A local Finnish experiment used Chatterbox Multilingual V2 with both Finnish and English reference recordings. The listener preferred the English-reference result in the tested comparison. This was a preference for those particular samples, not a general rule that one reference language is better.

A later greeting used actual forward video close-ups for head, eye and body movement, with MuseTalk generating lip synchronization. Selected shot intervals avoided automatic reversal or looping. Captions retained the intended Finnish wording; ASR supplied timing and diagnostic flags.

The completed local export passed full MP4/MP3 decoding, audio-level checks and source/caption hash checks. Sampled frames covered the beginning, cuts, long captions and ending. Mouth softness and incomplete closure during a quiet ending remained visible limitations. Full-performance quality was not inferred from those checks.

## 2026-09-30 — Prepare Dubiously Yours for reuse

The user chose the public project name **Dubiously Yours**, repository name `dubiously-yours`, and MIT licensing for the project's own code. The reusable scope is a staged CLI: reference preparation, speech generation, alignment, forward-video preparation, lip synchronization, export and verification.

The public workflow has neutral Finnish/English text examples, separate audio/video setup helpers, explicit provenance and a visible AI-parody label. It retains built-in speech watermarking and keeps personal media and models outside the release.

Original local documentation was copied and byte-verified before public-facing replacements. Publication uses an explicit file allowlist and a fresh staging directory rather than exporting the entire working folder. No repository push is part of this preparation.

The existing Finnish neural workflow has been exercised. A fresh installation of the public bootstrap and a fresh generic English Multilingual V2 audition are not claimed. Lightweight tests and model-free media checks document separate evidence from neural output and human listening.

## 2026-09-30 — Generic Finnish run and practical resource budgeting

The reusable CLI completed every stage on a new neutral Finnish sample using the installed environments: reference preparation, speech, alignment, forward-video preparation, GPU rendering, export and verification. The final clip is 7.84 seconds, 196 frames and 1280 × 720 at 25 fps. MP4 and MP3 passed full decoding and finite/unclipped audio checks. ASR still flagged minor wording differences; no listening-quality approval is inferred. Automated tests also cover publication isolation and model-free media exports in both languages.

Measured the current model payloads at **8,212,540,098 bytes**, excluding Python/PyTorch/CUDA packages, caches and source media. The [performance guide](PERFORMANCE.md) and [measurement snapshot](performance-example.json) record the tested hardware, disk footprints and runtime scope. Speech generation calls took 34.962 seconds for the two short passages, while MuseTalk's subprocess took 51.477 seconds. Those partial measurements are not a complete end-to-end wall time. New successful CLI stages write their own local wall-time records; older missing measurements are not backfilled.

Added optional source downloading through explicit yt-dlp commands and a separate media setup helper. Downloaded clips retain source metadata and frame rate; the documented workflow explains original versus local time intervals and optional 25 fps normalization. Network access is explicit in setup/fetch/download operations, while inference consumes local assets. The new downloader has not received a fresh live-network test in this packaging work.

## Continuing the diary

The final source checks passed 30 automated tests, including model-free Finnish/English media assembly, downloader validation, publication isolation, changed-input refusal and metadata/caption hash verification. The downloader environment passed an offline version/import check; a mocked download also exercised real FFmpeg validation. These checks do not substitute for a live extractor test or a fresh installation.

For a new result, record the input provenance, intended script, changed variable, environment/model revision, output checks and listening decision. Distinguish a running step from a finished artifact and a technical pass from a convincing performance. Keep private identities and recording details in ignored local run records.
