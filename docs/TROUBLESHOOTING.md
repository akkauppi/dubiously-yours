# Troubleshooting

Start with `python3 dub.py status projects/my-greeting/job.json`, the failed stage's log and `python3 dub.py doctor`. Preserve the output and provenance before trying another take. Use a new job ID for an independent experiment.

## Missing packages, models or GPU

Run the appropriate read-only checks:

```bash
python3 scripts/setup_audio.py --check
python3 scripts/setup_video.py --check
.venv-video/bin/python scripts/fetch_video_models.py --verify-only
python3 dub.py doctor --gpu
```

The environments use different Python/PyTorch versions. Run audio helpers with `.venv-audio/bin/python` and video helpers with `.venv-video/bin/python`; do not repair one by installing the other environment's requirements.

Package checks, weight hashes and GPU visibility establish different things. None proves that a full neural render works. Check the active GPU workload and available memory before rendering; pause your own competing workloads when appropriate.

## Perth fails with a missing or non-callable watermarker

The tested Perth package relies on `pkg_resources`. Keep the pinned `setuptools==80.9.0` in the audio environment. A later setuptools release can remove that API, and Perth may obscure the original import error. Restore the pinned environment instead of disabling watermarking.

## Tokenizer tries to write outside the project

PKUSEG and the Cangjie mapping require local cache paths even when generating Finnish or English. Use the setup/model helpers so `.cache/pkuseg` and the expected nested model cache exist. A model-weight file alone is not the complete tokenizer setup.

## Speech sounds wrong, or ASR disagrees

Listen to the reference and output. Check for mixed speakers, music, unfamiliar pronunciation, abrupt reference boundaries and lost segment endings. Compare a short passage with another seed or reference while retaining the same intended text.

ASR can merge names, change inflections or spell numbers differently. A transcript mismatch is a review flag, not proof of incorrect pronunciation. A clean transcript is not proof of a convincing accent. Native-language reference material can help, but longer or same-language references did not always win local listening comparisons.

The public job format keeps script and caption wording together. Do not silently substitute an ASR transcript for the script or display a phonetic name hint as if it were the intended name.

## Audio exceeds the export peak limit

Generation records raw levels and any downward export gain. Final packaging applies the configured `export.gain_db` once, equally to the MP4 and MP3 signals. A gain that removes headroom is rejected. Lower the gain and use a fresh output/job for the revised export; do not accept clipped audio to make it louder.

Lossy codecs can produce a different peak from the source WAV. Full float decoding checks the encoded files as well as the master.

## Video is too short or has an unsuitable frame rate

Supply 25 fps input and enough forward close-up material for the completed speech plus the inference tail. The pipeline rejects insufficient coverage. Add usable intervals or a longer local source; it does not reverse or loop a face to fill time.

For explicit local conversion when needed:

```bash
ffmpeg -n -i input/original.mp4 -vf fps=25 -c:v libx264 -crf 18 \
  -c:a aac input/video-25fps.mp4
```

Record the conversion and use the converted file in the job. Inspect it before rendering; conversion cannot fix a hidden face, fast turn or shot of another person.

## Missing face, soft mouth or an open mouth during silence

Choose a clear close-up with adequate face detail. MuseTalk changes a limited mouth region; it can soften texture or show artifacts despite a technically valid file. A face/frame failure should stop the render rather than silently shorten it.

Quiet audio does not guarantee a naturally closed mouth. Watch the ending and pauses. This public workflow retains synchronized frames during pauses; it does not blend back to the original speaking mouth or add artificial blinks.

## Captions are wrong or difficult to read

Caption words and punctuation must match the script exactly; timing comes from alignment. Verify that the alignment refers to the same audio hash as the export. Inspect long lines, cut boundaries and the last cue in the rendered video, not only a width calculation.

A successful decode verifies file integrity, not readable layout, comic timing or convincing lip synchronization.

## A stage refuses to overwrite an output

Preserve the prior run and create a new job ID or fresh output location. Do not delete successful files just to hide a failed experiment. The [developer diary](DEVELOPER_DIARY.md) explains the comparison-first approach; the [publishing guide](PUBLISHING.md) describes how private runs stay out of the public bundle.
