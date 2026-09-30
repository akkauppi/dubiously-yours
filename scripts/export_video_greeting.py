#!/usr/bin/env python3
"""Export a real-video parody with captions and matching MP4/MP3 audio.

The rendered 1280x720/25fps frames remain in order, including during pauses.
At most three surplus trailing frames are removed after all source speech.
Subtitles/labels and a documented fixed audio gain complete the export.
"""

import argparse
from array import array
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse
import wave

from render_portrait import sha256, valid_source_identifier

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"fi": "TEKOÄLYLLÄ TEHTY PARODIA", "en": "AI-GENERATED PARODY"}


def ass_time(seconds):
    cs = round(seconds * 100)
    hours, cs = divmod(cs, 360000)
    minutes, cs = divmod(cs, 6000)
    secs, cs = divmod(cs, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def probe(path):
    return json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True))


def pcm_details(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("The metadata-linked source audio must be mono PCM16 WAV")
        rate, frames = stream.getframerate(), stream.getnframes()
        if rate != 24000:
            raise ValueError("The metadata-linked source WAV must have a 24000 Hz sample rate")
        samples = array("h", stream.readframes(frames))
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples or len(samples) != frames:
        raise ValueError("Source WAV is empty or incomplete")
    peak = max(abs(n) for n in samples) / 32768
    rms = math.sqrt(sum((n / 32768) ** 2 for n in samples) / frames)
    if not rms:
        raise ValueError("Source WAV is silent")
    return {"sample_rate": rate, "frames": frames, "channels": 1, "subtype": "PCM_16",
            "duration_seconds": frames / rate, "peak": peak, "rms": rms}


def ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\r\n", "\n").replace("\n", "\\N")


def subtitles(cues, duration, publisher, language="fi"):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Speech,DejaVu Sans,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,1,2,0,2,70,70,72,1
Style: Label,DejaVu Sans,25,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,7,0,8,25,25,18,1
Style: Credit,DejaVu Sans,19,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,2,25,25,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    credit_line = (f"Videopohja: {publisher} — huulisynkronointia muokattu" if language == "fi" else
                   f"Source video: {publisher} — lip synchronization modified")
    events = [f"Dialogue: 1,0:00:00.00,{ass_time(duration)},Label,,0,0,0,,{LABELS[language]}",
              f"Dialogue: 1,0:00:00.00,{ass_time(duration)},Credit,,0,0,0,,{ass_escape(credit_line)}"]
    if not isinstance(cues, list) or not cues:
        raise ValueError("At least one caption cue is required")
    previous_end = 0.0
    for cue in cues:
        start, end, text = cue["start"], cue["end"], cue["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Caption text must be nonempty")
        if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (start, end)):
            raise ValueError("Caption times must be finite numbers")
        if not 0 <= start < end <= duration + 1e-6 or start < previous_end - 1e-6:
            raise ValueError(f"Caption outside the video or overlapping a previous cue: {cue}")
        if ass_time(start) == ass_time(end):
            raise ValueError("Caption is too short for ASS centisecond timing")
        previous_end = end
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Speech,,0,0,0,,{ass_escape(text)}")
    return header + "\n".join(events) + "\n"


def decoded_audio_details(path):
    # Decode float samples so lossy-codec overshoot is measurable, not clipped
    # away by a PCM16 decoder. Both exports use the same pre-encoding signal.
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                   "-ac", "1", "-c:a", "pcm_f32le", "-f", "f32le", "-"])
    samples = array("f", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples or not all(math.isfinite(x) for x in samples):
        raise ValueError(f"Decoded audio is empty or nonfinite: {path}")
    peak = max(abs(x) for x in samples)
    if peak >= 1:
        raise ValueError(f"Lossy export exceeds full scale ({peak:.6f}); use a new basename with a lower --gain-db")
    return {"decoded_samples": len(samples), "peak": peak,
            "rms": math.sqrt(sum(x * x for x in samples) / len(samples)),
            "finite": True, "unclipped": True}


def build_command(lipsync, audio, ass_name, video, mp3, frames, audio_frames, gain, title, language="fi"):
    # The ASS filename is a validated simple basename and cwd is the output
    # directory. No source path is interpolated into FFmpeg filter syntax.
    filters = (f"[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS,subtitles={ass_name},format=yuv420p[v];"
               f"[1:a]volume={gain:.17g}:precision=double,apad=whole_len={audio_frames},"
               f"atrim=end_sample={audio_frames},asetpts=PTS-STARTPTS,asplit=2[av][am]")
    common_metadata = ["-metadata", f"title={title}", "-metadata", f"comment={LABELS[language]}; fictional generated greeting, not an authentic statement."]
    return ["ffmpeg", "-v", "error", "-n", "-i", str(lipsync), "-i", str(audio),
            "-filter_complex_threads", "4", "-filter_complex", filters,
            "-map", "[v]", "-map", "[av]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-threads", "4", "-fps_mode:v", "passthrough",
            "-c:a", "aac", "-b:a", "128k", "-metadata:s:a:0", "language=" + ("fin" if language == "fi" else "eng"),
            "-movflags", "+faststart", *common_metadata, str(video),
            "-map", "[am]", "-c:a", "libmp3lame", "-q:a", "2", *common_metadata, str(mp3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-metadata", type=Path, required=True, help="Greeting JSON with output WAV path and SHA256")
    parser.add_argument("--caption-cues", type=Path, required=True)
    parser.add_argument("--caption-alignment", type=Path, help="Defaults to caption-alignment.json beside the cues; verifies their source WAV hash")
    parser.add_argument("--lipsync-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", required=True)
    parser.add_argument("--source-metadata", type=Path, help="Prepared inputs.json for automatic source attribution")
    parser.add_argument("--source-url", help="HTTP(S) URL or URN; overrides source metadata when supplied")
    parser.add_argument("--source-title")
    parser.add_argument("--source-publisher")
    parser.add_argument("--clip-start", type=float, help="Original source timeline, seconds")
    parser.add_argument("--clip-end", type=float, help="Original source timeline, seconds")
    parser.add_argument("--recipient", required=True)
    parser.add_argument("--language", choices=("fi", "en"), default="fi")
    parser.add_argument("--gain-db", type=float, default=0.0, help="One fixed gain shared by MP4 and MP3; default 0 dB")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.basename):
        parser.error("basename must be a simple filename stem")
    source_record = None
    if args.source_metadata:
        source_metadata = args.source_metadata.resolve()
        source_record = json.loads(source_metadata.read_text(encoding="utf-8"))
        args.source_url = args.source_url or source_record.get("source_url")
        args.source_title = args.source_title or source_record.get("source_title")
        args.source_publisher = args.source_publisher or source_record.get("uploader", source_record.get("publisher"))
        intervals = source_record.get("used_intervals_seconds", [])
        if args.clip_start is None:
            args.clip_start = source_record.get("downloaded_source_offset_seconds", intervals[0][0] if intervals else None)
        if args.clip_end is None:
            source_duration = source_record.get("source_clip_duration_seconds")
            args.clip_end = (args.clip_start + source_duration if source_duration is not None and args.clip_start is not None else
                             max((end for _, end in intervals), default=None))
        if not re.fullmatch(r"[a-f0-9]{64}", source_record.get("sha256", "")):
            parser.error("source-metadata must hash the prepared input video")
        if source_record.get("output_video"):
            source_video_path = Path(source_record["output_video"])
            if not source_video_path.is_absolute():
                source_video_path = ROOT / source_video_path
            if sha256(source_video_path) != source_record["sha256"]:
                raise ValueError("Prepared source video differs from source-metadata")
    for name in ("recipient", "source_title", "source_publisher"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
            parser.error(f"{name} must be supplied as a nonempty single line")
    if not valid_source_identifier(args.source_url):
        parser.error("source-url must be an HTTP(S) URL or URN source identifier")
    if args.clip_start is None or args.clip_end is None:
        parser.error("Supply source metadata intervals or explicit --clip-start and --clip-end")
    if not all(math.isfinite(x) for x in (args.clip_start, args.clip_end, args.gain_db)) or not 0 <= args.clip_start < args.clip_end:
        parser.error("Source interval and gain must be finite; 0 <= clip-start < clip-end")
    if not -60 <= args.gain_db <= 12:
        parser.error("Use a fixed gain between -60 and +12 dB")
    out = args.output_dir.resolve()
    video, mp3, metadata, credits, ass = [out / (args.basename + extension)
                                          for extension in (".mp4", ".mp3", ".json", ".CREDITS.md", ".ass")]
    for path in (video, mp3, metadata, credits, ass):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}")
    audio_metadata, cues_path, lipsync = (p.resolve() for p in (args.audio_metadata, args.caption_cues, args.lipsync_input))
    lipsync_sidecar = lipsync.with_suffix(".json")
    lipsync_record = None
    if lipsync_sidecar.exists():
        lipsync_record = json.loads(lipsync_sidecar.read_text(encoding="utf-8"))
        if lipsync_record.get("output_sha256") != sha256(lipsync):
            raise ValueError("Rendered video differs from its provenance sidecar")
        if source_record is not None and lipsync_record.get("input_video_sha256") != source_record["sha256"]:
            raise ValueError("Rendered video used a different prepared source")
    audio_record = json.loads(audio_metadata.read_text(encoding="utf-8"))
    audio = Path(audio_record["output"])
    audio = audio.resolve() if audio.is_absolute() else (ROOT / audio).resolve()
    digest = sha256(audio)
    recorded_hashes = [audio_record[k] for k in ("sha256", "output_sha256") if k in audio_record]
    if not recorded_hashes or any(value != digest for value in recorded_hashes):
        raise ValueError("Source audio SHA256 differs from its provenance metadata")
    audio_info = pcm_details(audio)
    if not math.isclose(audio_info["duration_seconds"], audio_record["duration_seconds"], abs_tol=1 / audio_info["sample_rate"]):
        raise ValueError("Source audio duration differs from its metadata")
    before = probe(lipsync)
    stream = next(s for s in before["streams"] if s["codec_type"] == "video")
    if (stream["width"], stream["height"], stream["r_frame_rate"], stream["avg_frame_rate"]) != (1280, 720, "25/1", "25/1"):
        raise ValueError("Prepare the MuseTalk video at 1280x720 and constant 25 fps before export")
    rendered_frames = int(stream["nb_frames"])
    # Integer arithmetic gives ceil(audio_duration * 25) without float drift.
    frames = (audio_info["frames"] * 25 + audio_info["sample_rate"] - 1) // audio_info["sample_rate"]
    extra_frames = rendered_frames - frames
    if not 0 <= extra_frames <= 3:
        raise ValueError(f"MuseTalk has {rendered_frames} frames; need {frames}, with at most three extra trailing frames")
    duration = frames / 25
    target_audio_frames = frames * (audio_info["sample_rate"] // 25)
    padding_samples = target_audio_frames - audio_info["frames"]
    padding = padding_samples / audio_info["sample_rate"]
    gain = 10 ** (args.gain_db / 20)
    projected_peak = audio_info["peak"] * gain
    if projected_peak > 0.95 + 1 / 32768:
        maximum = 20 * math.log10(0.95 / audio_info["peak"])
        raise ValueError(f"Requested gain lacks export headroom; use --gain-db <= {maximum:.3f}")
    cue_record = json.loads(cues_path.read_text(encoding="utf-8"))
    cues = cue_record["cues"] if isinstance(cue_record, dict) else cue_record
    alignment_path = (args.caption_alignment.resolve() if args.caption_alignment else
                      cues_path.with_name("caption-alignment.json"))
    alignment_record = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment_record.get("audio_sha256") != digest:
        raise ValueError("Caption alignment belongs to a different source WAV")
    if isinstance(cue_record, dict) and cue_record.get("audio_sha256", digest) != digest:
        raise ValueError("Caption cues belong to a different source WAV")
    if " ".join(" ".join(cue["text"].split()) for cue in cues) != " ".join(audio_record["script"].split()):
        raise ValueError("Caption words and punctuation differ from the source display script")
    ass_text = subtitles(cues, duration, args.source_publisher, args.language)
    title = (f"Tervehdys: {args.recipient} — tekoälyparodia" if args.language == "fi" else
             f"Greeting for {args.recipient} — AI-generated parody")
    command = build_command(lipsync, audio, ass.name, video, mp3, frames, target_audio_frames, gain, title, args.language)
    out.mkdir(parents=True, exist_ok=True)
    with ass.open("x", encoding="utf-8") as stream:
        stream.write(ass_text)
    print(json.dumps({"output": str(video), "mp3": str(mp3), "gain_db": args.gain_db,
                      "gain_linear": gain, "audio_padding_seconds": padding,
                      "trailing_video_frames_removed": extra_frames}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=out, check=True)
    for path in (video, mp3):
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
    after, mp3_probe = probe(video), probe(mp3)
    stream = next(s for s in after["streams"] if s["codec_type"] == "video")
    if (stream["width"], stream["height"], stream["r_frame_rate"], int(stream["nb_frames"])) != (1280, 720, "25/1", frames):
        raise RuntimeError("Export changed the rendered video dimensions, frame rate or frame count")
    decoded_checks = {"mp4": decoded_audio_details(video), "mp3": decoded_audio_details(mp3)}
    model_variant = audio_record.get("model_variant", "unspecified")
    model_name = {"multilingual-v2": "Chatterbox Multilingual V2", "multilingual-v3": "Chatterbox Multilingual V3",
                  "turbo": "Chatterbox Turbo"}.get(model_variant, audio_record.get("model", "Model recorded in audio metadata"))
    source = {"url": args.source_url, "title": args.source_title, "publisher": args.source_publisher,
              "clip_start_seconds": args.clip_start, "clip_end_seconds": args.clip_end,
              "used_intervals_seconds": source_record.get("used_intervals_seconds") if source_record else None,
              "metadata": str(source_metadata) if source_record else None,
              "metadata_sha256": sha256(source_metadata) if source_record else None,
              "provenance": source_record}
    record = {"label": "Fictional AI-generated greeting; not an authentic statement", "visible_label": LABELS[args.language],
              "created_utc": datetime.now(timezone.utc).isoformat(), "recipient": args.recipient, "language": args.language,
              "output": str(video), "sha256": sha256(video), "mp3": str(mp3), "mp3_sha256": sha256(mp3),
              "duration_seconds": duration, "frames": frames, "source_video": source,
              "rendered_input_frames": rendered_frames, "trailing_video_frames_removed": extra_frames,
              "lipsync_input": str(lipsync), "lipsync_input_sha256": sha256(lipsync),
              "lipsync_metadata": str(lipsync_sidecar) if lipsync_record else None,
              "lipsync_metadata_sha256": sha256(lipsync_sidecar) if lipsync_record else None,
              "audio": str(audio), "audio_sha256": digest, "audio_metadata": str(audio_metadata),
              "audio_metadata_sha256": sha256(audio_metadata), "source_audio_checks": audio_info,
              "caption_cues": str(cues_path), "caption_cues_sha256": sha256(cues_path), "captions_sha256": sha256(ass),
              "caption_alignment": str(alignment_path), "caption_alignment_sha256": sha256(alignment_path),
              "caption_source_audio_sha256_verified": True,
              "model_variant": model_variant, "speech_model": model_name,
              "voice_reference_source": audio_record.get("reference_source"),
              "voice_reference_sha256": audio_record.get("reference_sha256"),
              "lipsync_metadata": str(lipsync.with_suffix(".json")) if lipsync.with_suffix(".json").is_file() else None,
              "lipsync_metadata_sha256": sha256(lipsync.with_suffix(".json")) if lipsync.with_suffix(".json").is_file() else None,
              "watermark_provenance": audio_record.get("watermark"),
              "gain_db": args.gain_db, "gain_linear": gain, "projected_pcm_peak": projected_peak,
              "audio_padding_seconds": padding, "audio_padding_samples": padding_samples,
              "pre_encoding_audio_samples": target_audio_frames,
              "audio_treatment": "One shared volume/padding filter split to AAC and MP3 encoders; identical pre-encoding signal, no time stretching or speech trimming",
              "video_treatment": "MuseTalk frames and timing with burned captions/labels; only optional trailing surplus frames removed at the frame boundary after all source speech; no portrait animation, pause blend, frame loop, reversal or rescaling applied by this exporter",
              "decoded_audio_checks": decoded_checks, "ffprobe": after, "mp3_ffprobe": mp3_probe,
              "full_decode": "MP4 and MP3 passed", "ffmpeg_command": command, "working_directory": str(out),
              "generator_sha256": sha256(Path(__file__)),
              "assessment": "Technical export checks passed; full listening and visual performance review remain separate"}
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    source_link = (f"[{args.source_title}]({args.source_url})" if urlparse(args.source_url).scheme in ("http", "https") else
                   f"{args.source_title} — `{args.source_url}`")
    used_intervals = source.get("used_intervals_seconds") or [[args.clip_start, args.clip_end]]
    interval_text = ", ".join(f"{start:g}–{end:g} s" for start, end in used_intervals)
    if args.language == "fi":
        credit_text = f"""# {title}

Tämä on kuvitteellinen tekoälyllä tehty tervehdys, ei aito lausunto.

Videopohja: **{args.source_publisher}**, {source_link}.
Lähteen aikavälit: {interval_text}. Julkaisija ja lähdetiedot tulevat käyttäjän
antamista tiedoista; työkalu ei vahvista tekijänoikeuksia tai henkilöiden identiteettiä.

Puhe on korvattu synteettisellä äänellä ({model_name}) ja huulisynkronointi
muokattu MuseTalkilla. Pohjavideon liike tulee lähdetallenteesta; muunnettu
kasvoalue voi muuttua. Tauoissa ei palata alkuperäiseen puhuvaan suuhun.
Tekstitys säilyttää annetun käsikirjoituksen sanat ja nimien kirjoitusasun.

MP4 ja MP3 käyttävät samaa lähdeääntä ja kiinteää {args.gain_db:g} dB:n
vahvistusta. Lisätty loppuhiljaisuus: {padding:.6f} s; poistettuja ylimääräisiä
loppuruutuja: {extra_frames}. Puhetta ei ole leikattu tai nopeutettu.
Äänen alkuperä- ja vesileimatiedot: `{audio_metadata}`.
"""
    else:
        credit_text = f"""# {title}

This is a fictional AI-generated greeting, not an authentic statement.

Source video: **{args.source_publisher}**, {source_link}.
Source intervals: {interval_text}. Attribution comes from supplied metadata;
the tool does not verify recording rights or the identity of people shown.

Speech was replaced with synthetic audio ({model_name}); lip synchronization
was modified using MuseTalk. Base movement comes from the recording; the
processed face region can change. Pauses retain rendered frames rather than
restoring the original speaking mouth. Captions preserve the supplied script.

MP4 and MP3 share the same source audio and fixed {args.gain_db:g} dB gain.
Trailing silence: {padding:.6f} s. Surplus trailing video frames removed:
{extra_frames}. No speech was trimmed or sped up.
Audio provenance and watermark record: `{audio_metadata}`.
"""
    with credits.open("x", encoding="utf-8") as stream:
        stream.write(credit_text)
    print(json.dumps({"output": str(video), "mp3": str(mp3), "duration_seconds": duration,
                      "frames": frames, "full_decode": "passed"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
