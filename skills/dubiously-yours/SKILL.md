---
name: dubiously-yours
description: Make clearly labelled local AI voice greetings and lip-synced videos with Dubiously Yours from a script and source recordings or video links. Handle resource checks, setup, source selection, voice auditions and exports for a nontechnical user; evaluate alternative models when requested or when a specific quality or hardware problem warrants it.
---

# Dubiously Yours

Turn the user's creative direction into a short, reviewed greeting. Operate the CLI and write the job files yourself. Explain decisions in ordinary language; ask about the intended words, source passage and listening preference when those cannot be inferred. Do not make the user learn JSON, model settings or terminal commands.

This skill requires a local Dubiously Yours checkout and an agent able to read files and run commands. Locate the checkout from the working directory and its nearby parents using `dub.py`, `PUBLIC_FILES.txt` and `docs/WORKFLOW.md`. The installed skill directory may be elsewhere: run project commands from the checkout, not from the skill folder. If no checkout is present, obtain `https://github.com/akkauppi/dubiously-yours` within the user's intended workspace. If several installations could apply, determine which owns the user's assets before installing another stack.

Read the checkout's current `docs/WORKFLOW.md`. Read `docs/SETUP.md` only for missing dependencies and `docs/TROUBLESHOOTING.md` when a matching problem occurs. These describe the actual command and job schema; do not invent unsupported options. Use `--help` if the checkout and this skill differ.

## Check the machine before committing resources

Run the read-only report for the intended output duration and mode:

```bash
python3 scripts/check_resources.py --mode video --duration 10 --json
python3 dub.py doctor
```

Use `--mode audio` when making only speech. Inspect disk space on the actual model, environment, cache and media filesystems, available RAM, CPU/platform and accessible GPU memory. The helper does not download assets, allocate model memory or stop other programs. Its model-presence credit uses sizes; verify integrity with the setup/fetch checks before inference.

Translate the report into a brief explanation: what is already installed, additional download/storage estimates, what can run now and any unresolved limit. The dated reference configuration in `docs/PERFORMANCE.md` is an example, not the current machine. Separate model downloads from installed environments, caches and frame workspace. Source-download size can be unknown; check again after obtaining a clip and before rendering.

`ready_to_try` means the checks found no reported obstacle, not that inference is proven to fit. `needs_attention` needs interpretation: missing tools and inaccessible GPU diagnostics are different from a confirmed shortage. Treat `insufficient_resources` as a stop for the affected expensive operation; offer a smaller clip, another user-selected storage location or audio-only work if its own report permits it. Do not delete media/caches, eject another application's models, change GPU device selection or move to hosted services without authorization for that action.

For a fresh setup on another drive, prefer placing the whole checkout there and running its standard commands. Existing environments can contain absolute paths; do not promise that moving them or inventing model/cache flags will work. Inspect the supported layout before proposing relocation of an existing installation.

The current renderer enforces at least 4 GiB of free memory on its active CUDA device. Clearing that guard does not establish peak-memory fit; the previously tested 12 GiB card is not a universal minimum. Recheck just before GPU use because another application may have loaded a model. A failed `nvidia-smi` call can reflect WSL, permissions or driver visibility; do not equate it with absent hardware.

For setup, show the one-time download and storage estimate before starting, then use the pinned setup commands within the user's authorization. Avoid repeatedly requesting permission already given. Ask when the proposed route changes a material constraint such as cost, destination, use of a hosted service or disruption to another workload. Read-only setup checks establish installed versions and assets; a short real inference establishes more. A fresh machine installation is not guaranteed by the existing test results.

## Turn links and a request into a short audition

Collect only missing essentials: intended visible text or tone, recipient, Finnish/English language, approximate duration, and source file/link or passage. Preserve wording and names the user has already approved. If they provide creative direction without a script, draft a short candidate for their review while doing independent resource and source checks.

Use a new descriptive job ID and ignored `projects/`, `input/` and `outputs/` locations. Keep private identities and source URLs in those local records. Use `dub.py download` for an explicitly selected excerpt, retaining both sidecars. If download access is blocked, explain the concrete problem and request an accessible source or local recording; do not keep retrying without a changed cause.

Inspect the excerpt before treating it as suitable. Select at least 10 seconds of clear speech from the intended speaker without overlapping voices/music. For video, inspect sampled frames and shot boundaries for an unobstructed face and sufficient forward footage. Automated transcripts and face detection do not establish who is speaking or whether the reference sounds right. If the agent cannot hear the source, say so and get the user's listening judgment.

Keep source time separate from local time: downloading source seconds 60–120 produces roughly local 0–60. Set `video.source_offset_seconds` to 60 and select local intervals. Copy title, URL and uploader from the sidecars. Retain conversion records and the offset if creating a derivative. Video must be constant 25 fps; use a deliberate crop or pad for non-16:9 footage because the preparation script otherwise stretches to 1280×720. Choose chronological, nonoverlapping intervals and leave room beyond the speech for the inference tail.

Create/edit the job and script yourself, then run `reference`, `audio` and `align` on a short passage. Start with the requested ten-second scale or a representative opening of a longer greeting. Keep the current pinned baseline and change one comparison variable at a time. Give the user playable file links and a concise listening choice about names, accent, missing words and delivery. Wait for that judgment before an expensive full-length continuation unless the user has explicitly delegated the audition choice. Do not describe ASR as a likeness or pronunciation verdict.

The generic job format has no separate phonetic-text override. Do not silently replace the displayed name with a pronunciation hint or promise that this is already supported. Retain successful takes; changing inputs to an already-completed stage should use a new job rather than overwrite it. Correcting video settings before the video stage has run can remain in the existing job while preserving its approved audio and checked hashes.

## Complete the chosen take

After the audition decision, generate/align any remaining text, recheck resources and run `video`, `render`, `export`, then `verify`. Supply enough actual forward footage instead of enabling frame cycling to fill time. Head motion and blinks come from the source video; the current workflow does not synthesize a new acted performance or schedule blinks.

Use the actual generated audio length for the video resource check. The requested duration is a planning target, and the export follows the speech and inserted pauses. Report the resulting duration accurately. If the user requires an exact duration, resolve how to meet it instead of silently stretching speech, trimming words or treating the inference tail as final padding.

For an audio-only request, deliver the generated WAV and its metadata; the video export stage requires video. If MP3 is wanted, make a separate derivative with FFmpeg and record it without changing the speech master. Preserve the built-in speech watermark and the visible AI-parody label on video exports.

Verify full media decoding, input/metadata/caption hashes, duration and audio levels, then inspect readable captions, shot joins and the quiet ending. Ask the user to watch/listen where the agent's tools cannot judge playback. Report technical success separately from a convincing performance. Link the final media and local provenance, note material remaining artifacts, and record the chosen take and actual stage times locally. Generating a greeting does not imply permission to publish the personal media.

If a run fails, preserve logs and outputs. Fix an identified cause before a bounded short retry; do not loop expensive renders or alter the known working environment speculatively.

## When a newer model might help

Keep routine greetings on the tested pins. Research alternatives when the user asks, a concrete accent/motion/artifact problem remains after a short controlled comparison, or the requested device/language needs a different backend. Check the last dated evaluation so repeated greetings do not trigger repeated research. When freshness matters, browse current primary sources; do not present remembered model rankings as current facts.

Follow [model evaluation](references/model-evaluation.md) for a sourced comparison and isolated trial. Research and a candidate suggestion do not authorize replacing the working stack or downloading every candidate. A newer checkpoint is not a drop-in change to an unrelated loader. Recommend adoption only with evidence for the user's actual goal and a rollback path.
