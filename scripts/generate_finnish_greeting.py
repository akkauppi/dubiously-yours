#!/usr/bin/env python3
"""Generate or safely resume speech from a paragraph-preserving local plan."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import re
import shutil
import time
import unicodedata

if __package__:
    from .generate_finnish_sample import ROOT, REPO, REQUIRED_MODELS, configure_offline, sha256, validate_audio
    from .fetch_finnish_models import verify_manifest
else:
    from generate_finnish_sample import ROOT, REPO, REQUIRED_MODELS, configure_offline, sha256, validate_audio
    from fetch_finnish_models import verify_manifest

SAMPLING = {"language_id": "fi", "exaggeration": 0.5, "cfg_weight": 0.5,
            "temperature": 0.8, "top_p": 1.0, "min_p": 0.05, "repetition_penalty": 2.0}
RATE = 24000
LABEL = "AI-generated synthetic speech; not an authentic recording"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def relative_path(value, project_root=ROOT):
    project_root = Path(project_root).resolve()
    path = Path(value)
    if path.is_absolute() or not (project_root / path).resolve().is_relative_to(project_root):
        raise ValueError(f"Plan paths must be relative to the project: {value}")
    return (project_root / path).resolve()


def content_tokens(text):
    return [token for token in re.findall(r"\w+|[^\w\s]", text)
            if not all(unicodedata.category(c).startswith("P") for c in token)]


def planned_groups(plan, original, project_root=ROOT):
    paragraphs = re.split(r"\n[ \t]*\n(?:[ \t]*\n)*", original.strip())
    if not original.strip() or not isinstance(plan.get("groups"), list) or not plan["groups"]:
        raise ValueError("Script and plan groups must be nonempty")
    groups, coverage, ids = [], [], set()
    for group in plan["groups"]:
        identity = group["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identity) or identity == "greeting" or identity in ids:
            raise ValueError(f"Duplicate, reserved or unsafe segment id: {identity}")
        ids.add(identity)
        indexes = group["paragraphs"]
        if not isinstance(indexes, list) or not indexes or any(type(i) is not int or not 0 <= i < len(paragraphs) for i in indexes):
            raise ValueError(f"Invalid paragraph indexes: {identity}")
        coverage.extend(indexes)
        display = "\n\n".join(paragraphs[i] for i in indexes)
        tts = " ".join(paragraphs[i].strip() for i in indexes)
        if "tts_text" in group:
            override = group["tts_text"]
            if not isinstance(override, str) or not override.strip() or content_tokens(override) != content_tokens(tts):
                raise ValueError(f"TTS override may change punctuation/spacing only: {identity}")
            tts = override
        seed = group.get("seed", plan.get("seed", 43))
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError(f"Invalid seed: {identity}")
        pause = group["pause_after"]
        if isinstance(pause, bool) or not isinstance(pause, (int, float)) or not math.isfinite(pause) or pause < 0:
            raise ValueError(f"Invalid pause_after: {identity}")
        groups.append({"id": identity, "paragraphs": indexes, "display_text": display,
                       "tts_text": tts, "seed": seed, "pause_after": pause,
                       "reuse": str(relative_path(group["reuse"], project_root)) if group.get("reuse") else None})
    if coverage != list(range(len(paragraphs))):
        raise ValueError("Plan must preserve every paragraph exactly once and in original order")
    return groups


def read_audio(path, np, sf):
    info = sf.info(path)
    if info.subtype != "PCM_16" or info.channels != 1 or info.samplerate != RATE:
        raise ValueError(f"Expected mono 24 kHz PCM16 WAV: {path}")
    pcm, rate = sf.read(path, dtype="int16")
    checks = validate_audio(pcm.astype(np.float32) / 32768, rate, np)
    return pcm, checks


def validate_sidecar(path, record, expected, np, sf):
    """Check actual bytes and generation identity, including legacy pilot JSON."""
    if record.get("script") != expected["tts_text"]:
        raise ValueError(f"Segment script differs: {path}")
    hashes = [record[k] for k in ("sha256", "output_sha256") if k in record]
    if not hashes or any(value != sha256(path) for value in hashes):
        raise ValueError(f"Segment audio hash differs or is absent: {path}")
    for key in ("seed", "model", "model_variant", "model_revision", "model_file_sha256",
                "reference_sha256", "sampling"):
        if record.get(key) != expected[key]:
            raise ValueError(f"Segment {key} differs from plan: {path}")
    if record.get("device") != "cpu" or not record.get("watermark"):
        raise ValueError(f"Missing CPU/watermark provenance: {path}")
    pcm, checks = read_audio(path, np, sf)
    saved = record.get("exported_wav_checks")
    if not isinstance(saved, dict):
        raise ValueError(f"Missing exported WAV checks: {path}")
    for key in ("frames", "channels", "sample_rate", "finite", "unclipped"):
        if saved.get(key) != checks[key]:
            raise ValueError(f"Segment numeric metadata differs ({key}): {path}")
    for key in ("peak", "rms", "duration_seconds"):
        if not math.isclose(saved.get(key, -1), checks[key], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"Segment numeric metadata differs ({key}): {path}")
    return pcm, checks


def validate_resume(path, metadata, expected, signature, np, sf):
    if not path.is_file() or not metadata.is_file():
        raise ValueError(f"Incomplete segment WAV/JSON pair; refusing to replace: {path}")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    if record.get("group_signature") != signature:
        raise ValueError(f"Existing segment no longer matches its plan/group signature: {path}")
    pcm, checks = validate_sidecar(path, record, expected, np, sf)
    return record, pcm, checks


def exists(path):
    return path.exists() or path.is_symlink()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Base for all relative plan paths; defaults to the repository root")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    plan_path, output_dir = args.plan.resolve(), args.output_dir.resolve()
    final, final_json = output_dir / "greeting.wav", output_dir / "greeting.json"
    if exists(final) or exists(final_json):
        raise FileExistsError(f"Refusing to replace a completed or incomplete final assembly: {final}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    script_path = relative_path(plan["script"], args.project_root)
    original = script_path.read_text(encoding="utf-8")
    groups = planned_groups(plan, original, args.project_root)
    requested = plan.get("sampling")
    if not isinstance(requested, dict):
        raise ValueError("Plan sampling must be an object")
    language = requested.get("language_id", "fi")
    sampling = {**SAMPLING, "language_id": language}
    if language not in ("fi", "en") or {**sampling, **requested} != sampling:
        raise ValueError("Plan requires supported multilingual sampling settings and language_id fi or en")
    model_dir, reference, provenance_path = (relative_path(plan[k], args.project_root) for k in ("model_dir", "reference", "reference_metadata"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    reference_hash = sha256(reference)
    if provenance.get("sha256") != reference_hash or not isinstance(provenance.get("source_url"), str) or not provenance["source_url"].strip():
        raise ValueError("Reference provenance needs a matching SHA256 and source_url")
    manifest_path = model_dir / "download-manifest.json"
    required = [*REQUIRED_MODELS, "Cangjie5_TC.json"]
    if (model_dir / "conds.pt").is_file():
        required.append("conds.pt")
    manifest = verify_manifest(model_dir, repo=REPO, required=required)
    if manifest.get("repo") != REPO or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")):
        raise ValueError("Model manifest must identify a pinned ResembleAI/chatterbox revision")
    model_hashes = {}
    for name in (*REQUIRED_MODELS, "conds.pt", "Cangjie5_TC.json"):
        path = model_dir / name
        if name in REQUIRED_MODELS or path.is_file():
            model_hashes[name] = sha256(path)
            saved = manifest.get("files", {}).get(name, {})
            if isinstance(saved, dict) and saved.get("sha256") and saved["sha256"] != model_hashes[name]:
                raise ValueError(f"Model differs from manifest: {name}")
    cache_dir = configure_offline(4, args.cache_dir, model_dir)
    import numpy as np
    import soundfile as sf
    packages = {n: importlib.metadata.version(n) for n in ("chatterbox-tts", "torch", "torchaudio", "transformers",
                 "diffusers", "resemble-perth", "numpy", "soundfile")}
    common = {"model": REPO, "model_variant": "multilingual-v2", "model_revision": manifest["revision"],
              "model_file_sha256": model_hashes, "reference_sha256": reference_hash, "sampling": sampling}
    audit = {"label": LABEL, **common, "model_snapshot": str(model_dir), "model_manifest_sha256": sha256(manifest_path),
             "reference": str(reference), "reference_source": provenance["source_url"], "reference_details": provenance,
             "reference_metadata_sha256": sha256(provenance_path), "script_path": plan["script"],
             "script_sha256": sha256(script_path), "plan": str(plan_path), "plan_sha256": sha256(plan_path),
             "generator_sha256": sha256(Path(__file__)), "package_versions": packages,
             "device": "cpu", "threads": 4, "interop_threads": 1, "offline": True,
             "cache_directory": str(cache_dir),
             "watermark": "Built-in Chatterbox Perth watermark retained; detector not independently run",
             "max_new_speech_tokens": 1000, "reference_conditioning": "Prepared once from the same reference/exaggeration; equivalent cached conditioning reused for each generated group",
             "assessment": "Numerically checked; performance, pronunciation and completeness pending listening/ASR"}

    # Validate every existing and explicitly reused segment before loading a model.
    prepared = []
    for group in groups:
        output = output_dir / (group["id"] + ".wav")
        sidecar = output.with_suffix(".json")
        expected = {**common, "tts_text": group["tts_text"], "seed": group["seed"]}
        reuse_identity = None
        source_meta = None
        if group["reuse"]:
            source = Path(group["reuse"])
            source_sidecar = source.with_suffix(".json")
            source_meta = json.loads(source_sidecar.read_text(encoding="utf-8"))
            validate_sidecar(source, source_meta, expected, np, sf)
            reuse_identity = {"path": str(source), "sha256": sha256(source),
                              "metadata_sha256": sha256(source_sidecar)}
        signature = fingerprint({"version": 1, **expected, "id": group["id"], "paragraphs": group["paragraphs"],
                                 "display_text": group["display_text"], "package_versions": packages,
                                 "reuse": reuse_identity})
        state = {"group": group, "output": output, "sidecar": sidecar, "expected": expected, "signature": signature}
        if exists(output) or exists(sidecar):
            state["resumed"] = validate_resume(output, sidecar, expected, signature, np, sf)
        elif group["reuse"]:
            state["reuse_metadata"] = source_meta
            state["reuse_metadata_sha256"] = reuse_identity["metadata_sha256"]
        prepared.append(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    pieces, records, cursor = [], [], 0
    load_seconds = conditioning_seconds = None
    for state in prepared:
        group, output, sidecar = state["group"], state["output"], state["sidecar"]
        if "resumed" in state:
            # Recheck hashes at use time in case files changed during earlier work.
            record, pcm, checks = validate_resume(output, sidecar, state["expected"], state["signature"], np, sf)
        else:
            if exists(output) or exists(sidecar):
                raise FileExistsError(f"Segment appeared after preflight: {output}")
            started = time.monotonic()
            details = {}
            if group["reuse"]:
                source = Path(group["reuse"])
                if sha256(source.with_suffix(".json")) != state["reuse_metadata_sha256"]:
                    raise ValueError(f"Reuse metadata changed after preflight: {source}")
                validate_sidecar(source, state["reuse_metadata"], state["expected"], np, sf)
                with source.open("rb") as src, output.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                details = {"reused_selected_take": str(source), "reuse_source_sha256": sha256(source),
                           "reuse_metadata_sha256": state["reuse_metadata_sha256"],
                           "reuse_source_metadata": state["reuse_metadata"], "additional_export_gain_linear": 1.0}
                for key in ("raw_float_checks", "raw_float_peak", "raw_float_rms", "export_gain_linear", "export_gain_db", "export_ceiling_peak", "export_adjustment"):
                    if key in state["reuse_metadata"]:
                        details[key] = state["reuse_metadata"][key]
            else:
                import torch
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                if model is None:
                    if not re.search(r"max_new_tokens\s*=\s*1000\b", inspect.getsource(ChatterboxMultilingualTTS.generate)):
                        raise RuntimeError("Upstream 1000-token cap changed; review before generating")
                    torch.set_num_threads(4)
                    torch.set_num_interop_threads(1)
                    print("Loading local Multilingual V2 once on CPU", flush=True)
                    tick = time.monotonic()
                    model = ChatterboxMultilingualTTS.from_local(model_dir, device="cpu")
                    load_seconds = time.monotonic() - tick
                    if model.sr != RATE or not callable(getattr(model.watermarker, "apply_watermark", None)):
                        raise RuntimeError("Expected 24 kHz model and real Perth watermarker")
                    tick = time.monotonic()
                    with torch.inference_mode():
                        model.prepare_conditionals(str(reference), exaggeration=sampling["exaggeration"])
                    conditioning_seconds = time.monotonic() - tick
                random.seed(group["seed"])
                np.random.seed(group["seed"])
                torch.manual_seed(group["seed"])
                tick = time.monotonic()
                with torch.inference_mode():
                    wav = model.generate(group["tts_text"], **sampling)
                render_seconds = time.monotonic() - tick
                audio = wav.detach().cpu().numpy()
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio[0]
                raw = validate_audio(audio, RATE, np, require_unclipped=False)
                gain = min(1.0, 0.95 / raw["peak"])
                audio = audio * gain
                validate_audio(audio, RATE, np)
                with output.open("xb") as stream:
                    sf.write(stream, audio, RATE, subtype="PCM_16", format="WAV")
                details = {"render_seconds": render_seconds, "raw_float_checks": raw, "raw_float_peak": raw["peak"],
                           "raw_float_rms": raw["rms"], "export_gain_linear": gain, "export_gain_db": 20 * math.log10(gain),
                           "export_ceiling_peak": 0.95, "export_adjustment": "gain=min(1,0.95/raw_peak), once after watermarking; no amplification"}
            pcm, checks = read_audio(output, np, sf)
            record = {**audit, **details, "created_utc": datetime.now(timezone.utc).isoformat(), "id": group["id"],
                      "display_text": group["display_text"], "tts_text": group["tts_text"], "script": group["tts_text"],
                      "paragraphs": group["paragraphs"], "seed": group["seed"], "group_signature": state["signature"],
                      "generation_seconds": time.monotonic() - started, "exported_wav_checks": checks,
                      "duration_seconds": checks["duration_seconds"], "sample_rate": RATE, "subtype": "PCM_16",
                      "sha256": sha256(output), "output_sha256": sha256(output), "output": str(output)}
            validate_sidecar(output, record, state["expected"], np, sf)
            write_json(sidecar, record)
        pause_frames = round(group["pause_after"] * RATE)
        start = cursor / RATE
        speech_end = (cursor + len(pcm)) / RATE
        pieces.extend([pcm, np.zeros(pause_frames, dtype=np.int16)])
        cursor += len(pcm) + pause_frames
        records.append({**record, "path": str(output), "start": start, "speech_end": speech_end,
                        "end_with_pause": cursor / RATE, "pause_after_seconds": pause_frames / RATE,
                        "requested_pause_after_seconds": group["pause_after"]})
        print(json.dumps({"segment": group["id"], "speech_seconds": len(pcm) / RATE,
                          "assembled_seconds": cursor / RATE, "resumed": "resumed" in state}), flush=True)
    assembled = np.concatenate(pieces)
    with final.open("xb") as stream:
        sf.write(stream, assembled, RATE, subtype="PCM_16", format="WAV")
    decoded, checks = read_audio(final, np, sf)
    if not np.array_equal(decoded, assembled):
        raise RuntimeError("Final PCM differs from concatenated segments and silence")
    final_record = {**audit, "created_utc": datetime.now(timezone.utc).isoformat(), "script": original,
                    "seed": plan.get("seed", 43), "segments": records, "sample_rate": RATE,
                    "duration_seconds": len(assembled) / RATE, "output": str(final), "sha256": sha256(final),
                    "model_load_seconds": load_seconds, "conditioning_seconds": conditioning_seconds,
                    "exported_wav_checks": checks, "assembly": "Original PCM16 segments plus digital silence; no additional gain or speed change"}
    write_json(final_json, final_record)
    print(json.dumps({"output": str(final), "duration_seconds": final_record["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
