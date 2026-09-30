"""Offline stdlib checks for media setup and timed-download planning."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_clip
import setup_media


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="download unit ")
        self.root = Path(self.temporary.name)
        self.output = self.root / "clip.mp4"
        self.url = "https://example.test/video?v=example&list=ignored"

    def tearDown(self):
        self.temporary.cleanup()

    def test_rejects_non_http_urls_and_invalid_ranges(self):
        for url in ("file:///tmp/video", "ftp://example.test/video", "https:///missing-host",
                    "https://user:password@example.test/video", "https://example.test/a b"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                download_clip.validate_request(url, 0, 1, self.output)
        for start, end in ((-1, 1), (2, 1), (1, 1), (float("nan"), 2), (0, float("inf"))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                download_clip.validate_request(self.url, start, end, self.output)
        self.assertEqual(download_clip.validate_request("http://example.test/video", 0, 1, self.output), self.output)

    def test_requires_mp4_and_rejects_control_characters(self):
        for output in (self.root / "clip.mkv", self.root / "clip\n.mp4", self.root / ".mp4"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                download_clip.validate_request(self.url, 0, 1, output)

    def test_refuses_output_sidecars_and_partial_files(self):
        for filename in ("clip.mp4", "clip.info.json", "clip.source.json", "clip.mp4.part", "clip.f137.mp4"):
            path = self.root / filename
            path.write_bytes(b"preserve this")
            with self.subTest(filename=filename), self.assertRaises(FileExistsError):
                download_clip.validate_request(self.url, 0, 1, self.output)
            self.assertEqual(path.read_bytes(), b"preserve this")
            path.unlink()
        self.output.symlink_to(self.root / "missing-target")
        with self.assertRaises(FileExistsError):
            download_clip.validate_request(self.url, 0, 1, self.output)
        self.assertTrue(self.output.is_symlink())

    def test_argv_preserves_timing_url_and_output_template(self):
        command = download_clip.build_command(self.url, 60.125, 61.75,
            self.root / "100% source.mp4", Path("/local tools/node"), python=Path("/local tools/python"))
        self.assertEqual(command[:3], ["/local tools/python", "-m", "yt_dlp"])
        self.assertEqual(command[-2:], ["--", self.url])
        self.assertEqual(command[command.index("--download-sections") + 1], "*60.125-61.75")
        self.assertEqual(command[command.index("--js-runtimes") + 1], "node:/local tools/node")
        self.assertTrue(command[command.index("--output") + 1].endswith("100%% source.mp4"))
        for flag in ("--no-playlist", "--no-overwrites", "--no-continue", "--ignore-config",
                     "--no-plugin-dirs", "--no-remote-components", "--force-keyframes-at-cuts", "--write-info-json"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--downloader-args") + 1],
                         "ffmpeg_o:-fps_mode passthrough -threads 4")
        self.assertNotIn("-r", command)
        self.assertNotIn("--recode-video", command)
        self.assertNotIn("height<=", command[command.index("--format") + 1])

    def test_dry_run_creates_no_files_or_network_calls(self):
        output = self.root / "not-created" / "clip.mp4"
        argv = ["download_clip.py", "--url", self.url, "--start", "10", "--end", "11", "--output", str(output), "--dry-run"]
        before = list(self.root.rglob("*"))
        with patch.object(sys, "argv", argv), patch.object(download_clip, "find_node", side_effect=FileNotFoundError), \
                patch.object(download_clip.subprocess, "run", side_effect=AssertionError("Executed a download")), \
                patch.object(setup_media.urllib.request, "urlopen", side_effect=AssertionError("Used network")), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            download_clip.main()
        record = json.loads(stream.getvalue())
        self.assertTrue(record["dry_run"])
        self.assertEqual(record["source_interval_seconds"], [10.0, 11.0])
        self.assertEqual(record["info"], str(output.with_suffix(".info.json")))
        self.assertEqual(record["provenance"], str(output.with_suffix(".source.json")))
        self.assertEqual(list(self.root.rglob("*")), before)

    def test_inspection_rejects_extractor_rebased_interval_before_decode(self):
        with patch.object(download_clip.subprocess, "check_output", side_effect=AssertionError("Probed invalid interval")):
            with self.assertRaisesRegex(ValueError, "section_start"):
                download_clip.inspect_clip(self.output, {"section_start": 0, "section_end": 1}, 10, 11)

    def test_setup_default_only_prints_a_plan(self):
        with patch.object(sys, "argv", ["setup_media.py"]), \
                patch.object(setup_media.subprocess, "run", side_effect=AssertionError("Executed setup")), \
                patch.object(setup_media.urllib.request, "urlopen", side_effect=AssertionError("Used network")), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            setup_media.main()
        record = json.loads(stream.getvalue())
        self.assertEqual(record["python"], "3.12.3")
        self.assertEqual(record["node"], "22.13.0")
        self.assertIn("--install", record["network"])

    def test_setup_check_reads_local_versions_without_writes(self):
        environment = self.root / "venv"
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python").touch()
        lock = self.root / "requirements.txt"
        lock.write_text("yt-dlp[default]==2026.8.19\nyt-dlp-ejs==0.8.0\n")
        expected = {"python": "3.12.3", "packages": {"yt-dlp": "2026.8.19", "yt-dlp-ejs": "0.8.0"}}
        before = sorted(str(p) for p in self.root.rglob("*"))
        commands = []
        def local_check(command, **kwargs):
            commands.append(command)
            if command[-1] == "--version":
                return "v22.13.0\n"
            self.assertIn("-B", command)
            self.assertEqual(json.loads(command[-1]), expected["packages"])
            return json.dumps(expected)
        with patch.object(setup_media, "ENVIRONMENT", environment), patch.object(setup_media, "LOCK", lock), \
                patch.object(setup_media, "find_node", return_value=Path("/local/node")), \
                patch.object(setup_media.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), \
                patch.object(setup_media.subprocess, "check_output", side_effect=local_check), \
                patch.object(setup_media.urllib.request, "urlopen", side_effect=AssertionError("Used network")):
            record = setup_media.check_environment()
        self.assertEqual(record["node_version"], "v22.13.0")
        self.assertEqual(record["packages"], expected["packages"])
        self.assertEqual(len(commands), 2)
        self.assertEqual(sorted(str(p) for p in self.root.rglob("*")), before)


if __name__ == "__main__":
    unittest.main()
