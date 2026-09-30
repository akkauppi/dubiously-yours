"""Small real FFmpeg test; synthetic tones/colours, no face or neural models."""

from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and FFprobe required")
class MediaSmokeTests(unittest.TestCase):
    def test_forward_preparation_and_captioned_export(self):
        with tempfile.TemporaryDirectory(prefix="dub media smoke ") as temporary:
            root = Path(temporary)
            source, audio = root / "colours.mp4", root / "tone.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i", "color=red:s=1280x720:r=25:d=1",
                            "-f", "lavfi", "-i", "color=blue:s=1280x720:r=25:d=1", "-filter_complex_threads", "2",
                            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-an",
                            "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(source)], check=True)
            pcm = array("h", (round(4500 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(24240)))
            if sys.byteorder != "little":
                pcm.byteswap()
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(pcm.tobytes())
            metadata = {"output": str(audio), "sha256": digest(audio), "duration_seconds": 1.01,
                        "script": "Hello, friend!", "model_variant": "test-tone", "reference_source": "urn:recording:test-tone"}
            meta = root / "audio.json"
            meta.write_text(json.dumps(metadata))
            info = root / "source.json"
            info.write_text(json.dumps({"webpage_url": "urn:recording:test-colours", "title": "Generated colours", "uploader": "Test fixture"}))
            prepared = root / "prepared video"
            command = [sys.executable, str(ROOT / "scripts/prepare_interview_video.py"), "--source-video", str(source),
                       "--source-info", str(info), "--source-offset", "0", "--audio-metadata", str(meta),
                       "--ranges", "[[0,0.52],[1,1.60]]", "--output-dir", str(prepared)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            inputs = json.loads((prepared / "inputs.json").read_text())
            self.assertEqual((inputs["target_frames"], inputs["inference_frames"]), (26, 28))
            with wave.open(str(prepared / "inference-audio.wav"), "rb") as stream:
                padded = stream.readframes(stream.getnframes())
            self.assertEqual(padded[:len(pcm) * 2], pcm.tobytes())
            self.assertEqual(padded[len(pcm) * 2:], bytes(2640 * 2))
            # The edit must be red followed by blue, with no reverse fill.
            for time, channel in ((0.0, 0), (0.56, 2)):
                pixel = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(time), "-i", str(prepared / "source-video.mp4"),
                    "-frames:v", "1", "-vf", "scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
                self.assertEqual(len(pixel), 3)
                self.assertGreater(pixel[channel], 180)
            cues = root / "caption-cues.json"
            cues.write_text(json.dumps([{"start": 0.1, "end": 0.9, "text": "Hello, friend!"}]))
            alignment = root / "caption-alignment.json"
            alignment.write_text(json.dumps({"audio_sha256": digest(audio)}))
            export = root / "export with spaces"
            base = [sys.executable, str(ROOT / "scripts/export_video_greeting.py"), "--audio-metadata", str(meta),
                    "--caption-cues", str(cues), "--lipsync-input", str(prepared / "source-video.mp4"),
                    "--source-metadata", str(prepared / "inputs.json"), "--output-dir", str(export),
                    "--recipient", "A fictional friend", "--gain-db", "0"]
            for language, label in (("fi", "TEKOÄLYLLÄ TEHTY PARODIA"), ("en", "AI-GENERATED PARODY")):
                subprocess.run(base + ["--basename", language, "--language", language], check=True, capture_output=True, text=True)
                record = json.loads((export / f"{language}.json").read_text())
                self.assertEqual(record["frames"], 26)
                self.assertEqual(record["trailing_video_frames_removed"], 2)
                self.assertEqual(record["audio_padding_samples"], 720)
                self.assertEqual(record["visible_label"], label)
                self.assertIn(label, (export / f"{language}.ass").read_text())
                self.assertTrue(all(c["unclipped"] for c in record["decoded_audio_checks"].values()))
            before = digest(export / "fi.mp4")
            repeated = subprocess.run(base + ["--basename", "fi"], capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(digest(export / "fi.mp4"), before)
            alignment.write_text(json.dumps({"audio_sha256": "0" * 64}))
            wrong = subprocess.run(base + ["--basename", "wrong-audio"], capture_output=True, text=True)
            self.assertNotEqual(wrong.returncode, 0)
            self.assertFalse((export / "wrong-audio.mp4").exists())


if __name__ == "__main__":
    unittest.main()
