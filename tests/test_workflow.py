import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dub


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "examples").mkdir()
        for language in ("fi", "en"):
            (self.root / f"examples/demo-{language}.txt").write_text("Hello, friend.\n\nThis is a fictional greeting.\n")
        self.config = dub.init_job("demo", "fi", self.root)
        self.job = dub.load_job(self.config, self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def save_and_load(self):
        self.config.write_text(json.dumps(self.job))
        return dub.load_job(self.config, self.root)

    def test_init_preserves_existing_project(self):
        before = self.config.read_bytes()
        with self.assertRaises(FileExistsError):
            dub.init_job("demo", "en", self.root)
        self.assertEqual(self.config.read_bytes(), before)

    def test_init_rejects_escaping_parent_symlink_before_writing(self):
        isolated = self.root / "isolated"
        isolated.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (isolated / "projects").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            dub.init_job("escape", "fi", isolated)
        self.assertEqual(list(outside.iterdir()), [])

    def test_stage_log_rejects_symlink_without_touching_target(self):
        path = dub.stage_log_path(self.job, self.root)
        path.parent.mkdir(parents=True)
        target = self.root / "keep.txt"
        target.write_text("unchanged")
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            dub.stage_log_path(self.job, self.root)
        self.assertEqual(target.read_text(), "unchanged")

    def test_rejects_path_escape_and_symlink_escape(self):
        for value in ("../outside.wav", str(Path(self.tmp.name).parent / "outside.wav")):
            self.job["voice"]["media"] = value
            with self.assertRaises(ValueError):
                self.save_and_load()
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        self.job["voice"]["media"] = "escape/outside.wav"
        with self.assertRaises(ValueError):
            self.save_and_load()

    def test_rejects_output_symlink(self):
        (self.root / "outputs").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            dub.paths(self.job, self.root)

    def test_invalid_interval_and_gain_rejected(self):
        for ranges in ([[0, 0]], [[2, 1]], [[0, float("nan")]], [[False, 1]]):
            with self.assertRaises(ValueError):
                dub.intervals(ranges)
        self.job["export"]["gain_db"] = float("inf")
        with self.assertRaises(ValueError):
            self.save_and_load()

    def test_video_fit_preserves_forward_order_and_never_loops(self):
        self.assertEqual(dub.fit_video_ranges([[1, 2], [5, 7]], 40), [[1, 2.0], [5, 5.6]])
        with self.assertRaisesRegex(ValueError, "Not enough video"):
            dub.fit_video_ranges([[1, 2]], 26)
        with self.assertRaisesRegex(ValueError, "0.04"):
            dub.fit_video_ranges([[0.01, 2]], 25)
        for ranges in ([[2, 3], [0, 1]], [[0, 2], [1, 3]]):
            with self.assertRaisesRegex(ValueError, "chronological"):
                dub.fit_video_ranges(ranges, 25)

    def test_plans_preserve_paragraphs_and_language(self):
        for language in ("fi", "en"):
            self.job["language"] = language
            plan = dub.make_plan(self.job, dub.paths(self.job, self.root), self.root)
            self.assertEqual(plan["sampling"]["language_id"], language)
            self.assertEqual([g["paragraphs"] for g in plan["groups"]], [[0], [1]])
        self.job["tts"]["groups"] = [{"id": "missing", "paragraphs": [1], "pause_after": 0.5}]
        with self.assertRaisesRegex(ValueError, "every paragraph"):
            dub.make_plan(self.job, dub.paths(self.job, self.root), self.root)

    def test_dry_run_does_not_write_or_import_models(self):
        before = set(self.root.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()):
            for stage in dub.STAGES:
                dub.run_stage(self.job, stage, dry_run=True, root=self.root)
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("chatterbox", sys.modules)

    def test_stale_reference_script_and_settings_stop_downstream_work(self):
        p = dub.paths(self.job, self.root)
        p["work"].mkdir(parents=True)
        source = self.root / self.job["voice"]["media"]
        source.parent.mkdir()
        source.write_bytes(b"source")
        p["reference"].write_bytes(b"reference")
        voice = self.job["voice"]
        reference = {"source_media": str(source), "source_url": voice["source_url"],
                     "source_title": voice["title"], "credit": voice["credit"],
                     "source_excerpts_seconds": voice["intervals_seconds"],
                     "sha256": dub.sha256(p["reference"]), "source_sha256": dub.sha256(source)}
        p["reference_meta"].write_text(json.dumps(reference))
        dub.validate_dependencies(self.job, p, "audio", self.root)
        voice["intervals_seconds"] = [[1, 15]]
        with self.assertRaisesRegex(ValueError, "Voice settings changed"):
            dub.validate_dependencies(self.job, p, "audio", self.root)
        voice["intervals_seconds"] = [[0, 15]]
        source.write_bytes(b"different recording")
        with self.assertRaisesRegex(ValueError, "voice source"):
            dub.validate_dependencies(self.job, p, "audio", self.root)
        source.write_bytes(b"source")
        p["plan"].write_text(json.dumps(dub.make_plan(self.job, p, self.root)))
        p["audio"].parent.mkdir(parents=True)
        p["audio"].write_bytes(b"generated audio")
        script = self.root / self.job["script"]
        metadata = {"sha256": dub.sha256(p["audio"]), "script_sha256": dub.sha256(script),
                    "reference_metadata_sha256": dub.sha256(p["reference_meta"]),
                    "reference_sha256": dub.sha256(p["reference"]), "plan_sha256": dub.sha256(p["plan"])}
        p["audio_meta"].write_text(json.dumps(metadata))
        dub.validate_dependencies(self.job, p, "align", self.root)
        original = script.read_bytes()
        script.write_text("Changed wording")
        with self.assertRaisesRegex(ValueError, "script"):
            dub.validate_dependencies(self.job, p, "align", self.root)
        script.write_bytes(original)
        self.job["tts"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "Speech settings changed"):
            dub.validate_dependencies(self.job, p, "align", self.root)

    def test_verify_checks_each_recorded_sidecar_before_decoding(self):
        p = dub.paths(self.job, self.root)
        p["out"].mkdir(parents=True)
        record = {"recipient": self.job["recipient"], "language": "fi", "gain_db": 0,
                  "frames": 25, "duration_seconds": 1}
        sidecars = []
        for name, hash_key in (("output", "sha256"), ("mp3", "mp3_sha256"), ("audio", "audio_sha256"),
                               ("caption_cues", "caption_cues_sha256"), ("lipsync_input", "lipsync_input_sha256"),
                               ("audio_metadata", "audio_metadata_sha256"), ("caption_alignment", "caption_alignment_sha256"),
                               ("lipsync_metadata", "lipsync_metadata_sha256")):
            path = p["export"] if name == "output" else p["out"] / name
            path.write_text(name)
            record[name], record[hash_key] = str(path), dub.sha256(path)
            sidecars.append(path)
        captions = p["export"].with_suffix(".ass")
        captions.write_text("captions")
        record["captions_sha256"] = dub.sha256(captions)
        source = p["out"] / "source.json"
        source.write_text("source metadata")
        record["source_video"] = {"metadata": str(source), "metadata_sha256": dub.sha256(source)}
        sidecars.extend((captions, source))
        p["export"].with_suffix(".json").write_text(json.dumps(record))
        media = {"streams": [{"codec_type": "video", "width": 1280, "height": 720,
                              "avg_frame_rate": "25/1", "nb_frames": "25"}]}
        with patch.object(dub, "execute") as decode, patch.object(dub, "probe", return_value=media):
            with contextlib.redirect_stdout(io.StringIO()):
                dub.verify_export(self.job, p)
            self.assertEqual(decode.call_count, 2)
            for path in sidecars:
                with self.subTest(file=path.name):
                    original = path.read_bytes()
                    path.write_text("tampered")
                    decode.reset_mock()
                    with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                        dub.verify_export(self.job, p)
                    decode.assert_not_called()
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
