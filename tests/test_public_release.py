from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_release as release


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "source"
        self.root.mkdir()
        self.names = [".gitignore", "PUBLIC_FILES.txt", "README.md", "tool.py"]
        self.set_manifest(self.names)
        (self.root / "README.md").write_text("# Generic demo\n")
        (self.root / "tool.py").write_text("print('hello')\n")

    def tearDown(self):
        self.tmp.cleanup()

    def set_manifest(self, names):
        (self.root / "PUBLIC_FILES.txt").write_text("\n".join(names) + "\n")
        (self.root / ".gitignore").write_text(release.ignore_text(names))

    def test_only_allowlisted_files_are_copied(self):
        (self.root / "input").mkdir()
        (self.root / "input/private.wav").write_bytes(b"private media")
        out = self.root.parent / "public"
        report = release.build(self.root, out)
        self.assertTrue(report.is_file())
        self.assertEqual({str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}, set(self.names))
        with self.assertRaises(FileExistsError):
            release.build(self.root, out)

    def test_report_collision_is_rejected_before_creating_release(self):
        for kind in ("file", "dangling-symlink"):
            with self.subTest(kind=kind):
                out = self.root.parent / f"public-{kind}"
                report = out.with_name(out.name + "-manifest.json")
                target = self.root.parent / "missing-report-target.json"
                if kind == "file":
                    report.write_bytes(b"preserve existing report\n")
                else:
                    report.symlink_to(target)
                with self.assertRaisesRegex(FileExistsError, "report already exists"):
                    release.build(self.root, out)
                self.assertFalse(out.exists())
                if kind == "file":
                    self.assertEqual(report.read_bytes(), b"preserve existing report\n")
                else:
                    self.assertTrue(report.is_symlink())
                    self.assertFalse(target.exists())

    def test_private_paths_and_traversal_cannot_be_allowlisted(self):
        for name in ("input/recording.wav", "../secret.txt", "models/weights.pt", "local/history.md"):
            self.set_manifest(self.names + [name])
            with self.assertRaises(ValueError):
                release.audit(self.root, check_git=False)

    def test_symlinks_cannot_smuggle_private_files(self):
        (self.root / "tool.py").unlink()
        private = self.root.parent / "private.py"
        private.write_text("print('not public')\n")
        (self.root / "tool.py").symlink_to(private)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            release.audit(self.root, check_git=False)

    def test_absolute_paths_and_tokens_are_flagged_without_printing_values(self):
        for secret in ("/" + "home" + "/fictional/work/file.txt", "ghp_" + "a" * 40):
            (self.root / "README.md").write_text(secret)
            with self.assertRaises(ValueError) as context:
                release.audit(self.root, check_git=False)
            self.assertNotIn(secret, str(context.exception))

    def test_private_document_links_are_rejected(self):
        (self.root / "README.md").write_text("[private](input/greeting.mp4)\n")
        with self.assertRaisesRegex(ValueError, "link leaves"):
            release.audit(self.root, check_git=False)

    def test_ignore_rules_must_match_manifest(self):
        (self.root / ".gitignore").write_text("\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            release.audit(self.root, check_git=False)

    def test_update_ignore_preserves_external_symlink_targets(self):
        ignore = self.root / ".gitignore"
        for kind in ("existing", "missing"):
            with self.subTest(target=kind):
                ignore.unlink()
                target = self.root.parent / f"{kind}-ignore-target"
                if kind == "existing":
                    target.write_bytes(b"must remain unchanged\n")
                ignore.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "symlinked"):
                    release.update_ignore(self.root)
                self.assertTrue(ignore.is_symlink())
                if kind == "existing":
                    self.assertEqual(target.read_bytes(), b"must remain unchanged\n")
                else:
                    self.assertFalse(target.exists())
                self.assertEqual(list(self.root.glob(".gitignore-*")), [])

    def test_update_ignore_regenerates_regular_file(self):
        ignore = self.root / ".gitignore"
        ignore.write_text("outdated rules\n")
        ignore.chmod(0o640)
        release.update_ignore(self.root)
        self.assertEqual(ignore.read_text(), release.ignore_text(self.names))
        self.assertEqual(ignore.stat().st_mode & 0o777, 0o640)
        self.assertEqual(list(self.root.glob(".gitignore-*")), [])

    @unittest.skipUnless(shutil.which("git"), "Git required")
    def test_git_sees_only_the_allowlist_in_a_mixed_workspace(self):
        names = self.names + ["scripts/public.py", "docs/guide.md", ".github/workflows/check.yml"]
        self.set_manifest(names)
        for name in names[4:] + ["scripts/private.py", "notes.md", ".env", "outputs/private.wav", ".aws/credentials"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test fixture\n")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        visible = subprocess.check_output(["git", "-C", str(self.root), "ls-files", "--others", "--exclude-standard", "-z"])
        self.assertEqual(set(filter(None, visible.decode().split("\0"))), set(names))


if __name__ == "__main__":
    unittest.main()
