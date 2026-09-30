# Workflow

Dubiously Yours builds a local parody greeting in stages. Finish and listen to the voice before rendering the video. A successful command or transcript is evidence of a completed step, not proof that the performance sounds right.

## Optional: download a source clip

Use your own local recording or install the optional media tools described in [setup](SETUP.md), then request a specific clip:

```bash
python3 dub.py download --url 'YOUTUBE_URL' --start 60 --end 120 \
  --output input/source.mp4
```

Replace `YOUTUBE_URL` with the source page. Start/end are seconds on its original timeline: **downloading 60–120 seconds creates a clip whose local timeline begins at 0 and runs for approximately 60 seconds**. This is an explicit network operation; the later job stages use local files. The result is `input/source.mp4` with `source.info.json` extractor metadata and `source.source.json` provenance/checks beside it. Keep these source records and attribution with the clip. Existing outputs are not overwritten. Add `--dry-run` to inspect the download command before making a network request.

The downloader retains the source frame rate. The video pipeline requires 25 fps. When conversion is needed, create a separate local derivative:

```bash
ffmpeg -n -i input/source.mp4 -vf fps=25 -c:v libx264 -crf 18 \
  -c:a aac input/source-25fps.mp4
```

Inspect the converted file and record that processing. For this example, set `video.media` to `input/source-25fps.mp4`, `video.source_offset_seconds` to `60`, and use local intervals such as `[0, 15]`, which correspond to original source times 60–75 seconds. Voice reference intervals are also local; the same clip may provide audio and picture. Copy the original URL, title and uploader from the source sidecars into the job's voice/video source and credit fields. Preserve those sidecars and the source offset after frame-rate conversion. Recording rights remain separate from the code's MIT licence. A new frame rate cannot fix a hidden face or unsuitable shot.

## 1. Create a job

```bash
python3 dub.py init my-greeting --language fi
```

Use `--language en` for English. The command creates `projects/my-greeting/job.json` and `script.txt`. These are local working files, excluded from the public release. Use a new job ID for a separate experiment.

A job has this shape:

```json
{
  "schema_version": 1,
  "id": "my-greeting",
  "language": "fi",
  "recipient": "A friend",
  "script": "projects/my-greeting/script.txt",
  "voice": {
    "media": "input/voice.wav",
    "source_url": "urn:recording:my-voice",
    "title": "My voice recording",
    "credit": "Recorded by me",
    "intervals_seconds": [[0, 15]]
  },
  "video": {
    "media": "input/video.mp4",
    "source_url": "urn:recording:my-video",
    "title": "My video recording",
    "publisher": "Recorded by me",
    "source_offset_seconds": 0,
    "intervals_seconds": [[0, 40]]
  },
  "tts": {
    "seed": 43,
    "pause_seconds": 0.5,
    "groups": null
  },
  "export": {
    "basename": "greeting",
    "gain_db": 0
  }
}
```

All file paths are repository-relative. Place local media under `input/` or `projects/`; a URL is provenance, not a request to download it. The source identifier can be an HTTP(S) page or an own-recording URN.

Later stages check that their inputs still match the current job. Changing the script, voice source, synthesis settings or already-prepared video stops dependent work instead of reusing a stale take. Keep completed projects and create a new job for changed inputs.

Intervals are seconds within the supplied local media. Video boundaries must align to 25 fps frames: multiples of **0.04 seconds**. For a downloaded excerpt, `video.source_offset_seconds` records where that file starts on the original source timeline. The pipeline uses that offset for provenance. Check your media before setting intervals.

## 2. Prepare references

Choose clear speech from one intended speaker, without overlapping voices, music or abrupt edits in the middle of a word. Listen to the source. Silence detection and transcripts can suggest boundaries, but they cannot establish speaker identity or clean isolation.

```bash
python3 dub.py run projects/my-greeting/job.json reference
```

The reference stage extracts the ordered voice intervals to mono 24 kHz PCM, then records provenance and hashes. Use only the usable material; longer references do not guarantee a better voice. Different model components consume different portions of the reference, so concatenating more speech is not equivalent to training a model.

Reference language also matters. In local experiments, a same-language reference did not automatically beat a different-language reference. Compare actual outputs while keeping the script and generation settings steady.

## 3. Generate and choose speech

Write the intended visible wording in `script.txt`, separating paragraphs with blank lines. With `tts.groups: null`, each paragraph is a speech group.

```bash
python3 dub.py run projects/my-greeting/job.json audio
```

The audio stage writes a validated generation plan and creates speech locally on CPU. Listen to `outputs/my-greeting/audio/greeting.wav`; its adjacent JSON records the assembly. Synthesis uses Chatterbox Multilingual V2, with the job's `fi` or `en` language. Fixed seeds help record a comparison; they are not a guarantee of bit-identical output across hardware or library versions.

For control over phrasing, supply groups using zero-based paragraph indices:

```json
"groups": [
  {"id": "opening", "paragraphs": [0], "seed": 43, "pause_after": 0.6},
  {"id": "reply", "paragraphs": [1, 2], "seed": 44, "pause_after": 0.8},
  {"id": "ending", "paragraphs": [3], "seed": 43, "pause_after": 0.5}
]
```

Cover every script paragraph in order when setting groups. Listen for names, long words, missing words, pronunciation, pauses and the end of every segment. Compare a small opening before regenerating an entire greeting. Keep the original displayed name even when separately exploring pronunciation; the generic job format does not currently expose a separate phonetic-text override.

Use a new job for a new take when preserving an existing output. Do not treat an automatically cleaner transcript as a human preference. Numerical audio checks detect invalid values and clipping, not accent quality.

## 4. Align captions

```bash
python3 dub.py run projects/my-greeting/job.json align
```

Whisper estimates timing. Captions keep the script's words and punctuation instead of replacing them with recognition guesses. Review both alignment and speech: a correct-looking caption does not establish that the generated voice said the same thing.

## 5. Prepare forward video

Use a 25 fps video with a visible, unobstructed face. Prefer 16:9 landscape footage: preparation resizes to 1280 × 720, so another aspect ratio needs your own crop or padding first to avoid stretching. Choose close-ups that remain on the intended speaker. The source footage supplies real head, eye and body movement; MuseTalk modifies the mouth. This workflow does not generate a new acted performance or schedule artificial blinks.

```bash
python3 dub.py run projects/my-greeting/job.json video
```

The video intervals are available chronological, nonoverlapping forward shots. The CLI takes enough of them, in order, to cover the audio rounded up to a 25 fps frame boundary plus two inference frames. It rejects insufficient coverage instead of looping or reversing the footage. Give the video more usable duration than the speech, then check where the selected shots join. Put cuts in natural pauses when possible.

Prepared video is 1280 × 720. Original source audio is removed. The speech master remains preserved; a small silent tail supports MuseTalk's audio window and the final frame boundary.

## 6. Render and export

```bash
python3 dub.py doctor --gpu
python3 dub.py run projects/my-greeting/job.json render
python3 dub.py run projects/my-greeting/job.json export
```

The render stage uses the separate MuseTalk GPU environment. Export retains the ordered synchronized frames, adds exact-word captions, a visible language-appropriate AI-parody label and source credit, and creates MP4 plus MP3. One fixed gain is shared before the audio splits to the two encoders. There is no speech time stretching; surplus inference frames are removed after the speech.

The built-in speech watermark is retained. Source metadata, input hashes, commands and export checks provide a record of what was made.

## 7. Verify and watch

```bash
python3 dub.py run projects/my-greeting/job.json verify
python3 dub.py status projects/my-greeting/job.json
```

Verification checks media decoding, provenance hashes and captions. Also watch and listen to the whole clip at normal speed. Inspect the opening, shot cuts, longest captions and quiet ending. Generated mouth detail may soften, and silence does not guarantee a naturally closed mouth.

Outputs belong under `outputs/<job-id>/`; intermediate job material belongs under `projects/<job-id>/work/`. Successful CLI stages log their observed wall time to `projects/<job-id>/work/stage-times.jsonl`. Those new measurements do not fill in missing timing for earlier runs. See [PERFORMANCE.md](PERFORMANCE.md) for measured examples and setup sizes. Keep the useful provenance with a final export. Publication of the code and sharing a personal greeting are separate actions.

For an execution preview, use:

```bash
python3 dub.py run projects/my-greeting/job.json render --dry-run
```

This prints the planned command without loading models. There is intentionally no automatic “run everything” stage: listening and video selection sit between generation and rendering.
