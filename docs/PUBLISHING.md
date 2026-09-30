# Publishing the code

The public project is **Dubiously Yours**; the intended repository name is `dubiously-yours`. Its own code and neutral examples use the [MIT licence](../LICENSE).

Publish a clean bundle, not the media-production workspace. Local recordings, scripts for personal greetings, generated files, model weights, environment directories, caches and historical notes stay outside the public release.

## Build and inspect the bundle

`PUBLIC_FILES.txt` is an explicit allowlist. The release helper exports only listed files; the deny-by-default ignore rules provide an additional boundary. Existing historical files can remain locally without entering the release.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/public_release.py check
python3 scripts/public_release.py build --output dist/dubiously-yours
```

Build into a fresh directory. The helper does not push, create a remote repository or upload media. Inspect the resulting files, documentation links, licence and changes before publishing. Rebuild after substantive changes instead of manually copying private material into the staging directory.

For an existing personal workspace, an optional ignored text file can list one private phrase per line for additional scanning. Keep that list local; it is never copied into the bundle:

```bash
python3 scripts/public_release.py check --deny-terms-file local/private-terms.txt
python3 scripts/public_release.py build --output dist/dubiously-yours \
  --deny-terms-file local/private-terms.txt
```

Create the terms file before using these optional commands. Use complete identities or distinctive phrases so ordinary words do not trigger accidental matches.

The bundle contains project code, dependency pins, public docs, neutral text examples and lightweight tests. It excludes:

- `input/`, `outputs/`, `projects/` and `sample/` personal work.
- `local/history/` and other private documentation.
- `models/`, `vendor/`, caches and virtual environments.
- Recorded or generated audio/video, source downloads and credentials.

Pinned source locations in fetch helpers describe dependencies; they do not bundle those dependencies' model weights or relicense them.

## Review the Git commit

The following commands operate inside the clean staging directory, not the original media workspace:

```bash
cd dist/dubiously-yours
git init -b main
git add .
git diff --cached --stat
git diff --cached
git status --short
```

Inspect the staged content before committing. Ensure there are no personal names, source-media URLs, absolute workstation paths, tokens, logs or generated files. Generic upstream software/model URLs are expected.

When the reviewed content is ready:

```bash
git commit -m "Prepare Dubiously Yours local parody workflow"
```

Creating the public remote and pushing is a separate user action. For example, after creating an empty repository in your own GitHub account:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/dubiously-yours.git
git push -u origin main
```

These commands are instructions for the person publishing; project preparation does not execute them automatically.

## What release checks establish

Tests and release checks can catch path, command, metadata and packaging errors. Model-free FFmpeg tests can exercise preparation/export integrity without downloading speech models or requiring a GPU. They do not establish a successful clean model installation, realistic pronunciation, source-media rights or full lip-sync quality.

State those limits in release notes. Link reproduction commands and relevant checks, and report new platform support only after testing it. Keep third-party notices separate from the project's MIT licence; see [credits](../CREDITS.md).
