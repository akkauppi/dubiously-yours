#!/usr/bin/env python3
"""Read-only resource planning for the pinned local workflow; never installs or loads models.

Exit codes: 0 ready_to_try, 1 needs_attention, 2 insufficient_resources or invalid
CLI arguments. A size match is NOT a checksum or environment compatibility check.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MIB = 1024 ** 2
# Measured 2026-09-30, matching the fetch helpers' pinned revisions. These are
# inventory sizes, not a second trust/checksum manifest. Update when pins change.
AUDIO_FILES = {
    'models/chatterbox-multilingual-v2/ve.pt': 5698626,
    'models/chatterbox-multilingual-v2/t3_mtl23ls_v2.safetensors': 2143989752,
    'models/chatterbox-multilingual-v2/s3gen.pt': 1057165844,
    'models/chatterbox-multilingual-v2/grapheme_mtl_merged_expanded_v1.json': 69989,
    'models/chatterbox-multilingual-v2/Cangjie5_TC.json': 1920163,
    'models/chatterbox-multilingual-v2/README.md': 10081,
    'models/faster-whisper-small/config.json': 2370,
    'models/faster-whisper-small/model.bin': 483546902,
    'models/faster-whisper-small/tokenizer.json': 2203239,
    'models/faster-whisper-small/vocabulary.txt': 459861,
    'models/faster-whisper-small/README.md': 1998,
    '.cache/pkuseg/spacy_ontonotes.zip': 34567143,
}
VIDEO_FILES = {
    'models/musetalk/musetalkV15/musetalk.json': 748,
    'models/musetalk/musetalkV15/unet.pth': 3400074924,
    'models/musetalk/sd-vae/config.json': 547,
    'models/musetalk/sd-vae/diffusion_pytorch_model.bin': 334707217,
    'models/musetalk/whisper/config.json': 1983,
    'models/musetalk/whisper/pytorch_model.bin': 151095027,
    'models/musetalk/whisper/preprocessor_config.json': 184990,
    'models/musetalk/dwpose/dw-ll_ucoco_384.pth': 406878486,
    'models/musetalk/face-parse-bisent/79999_iter.pth': 53289463,
    'models/musetalk/face-parse-bisent/resnet18-5c106cde.pth': 46827520,
    'models/musetalk/s3fd/s3fd-619a316812.pth': 89843225,
}
TOKENIZER_EXPANDED = {
    '.cache/pkuseg/spacy_ontonotes/features.msgpack': 22685181,
    '.cache/pkuseg/spacy_ontonotes/weights.npz': 37508754,
}
ENVIRONMENT_ALLOWANCES = {'.venv-audio': int(1.9 * GIB), '.venv-video': int(5.4 * GIB), '.venv-media': 63 * MIB}
RENDERER_FREE_FLOOR = 4 * GIB


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError('duration must be positive and finite')
    return number


def file_inventory(root, files):
    records = []
    for relative, size in files.items():
        path = root / relative
        error = None
        actual = None
        try:
            if path.is_file():
                actual = path.stat().st_size
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        matches = actual == size
        records.append({'path': str(path), 'expected_bytes': size, 'observed_bytes': actual,
                        'size_matches': matches, 'remaining_payload_bytes': 0 if matches else size,
                        'error': error})
    return records


def filesystem_for(path):
    """Resolve symlinked destinations and inspect the closest existing ancestor."""
    resolved = Path(path).resolve()
    anchor = resolved
    while not anchor.exists():
        if anchor.parent == anchor:
            raise OSError(f'No existing ancestor for {path}')
        anchor = anchor.parent
    if anchor.is_file():
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    return {'device': str(anchor.stat().st_dev), 'anchor': str(anchor),
            'resolved_path': str(resolved), 'free_bytes': usage.free, 'total_bytes': usage.total}


def group_disk_allocations(allocations, inspect=filesystem_for):
    groups = {}
    unknown = []
    for entry in allocations:
        try:
            probe = inspect(entry['path'])
        except (OSError, RuntimeError) as exc:
            unknown.append({**entry, 'error': str(exc)})
            continue
        group = groups.setdefault(probe['device'], {
            'device': probe['device'], 'free_bytes': probe['free_bytes'],
            'total_bytes': probe['total_bytes'], 'known_remaining_asset_bytes': 0,
            'estimated_additional_bytes': 0, 'allocations': []})
        # One free-space pool per filesystem, never sum aliases of the same pool.
        group['free_bytes'] = min(group['free_bytes'], probe['free_bytes'])
        key = 'known_remaining_asset_bytes' if entry['kind'] == 'known_asset' else 'estimated_additional_bytes'
        group[key] += entry['bytes']
        group['allocations'].append({**entry, 'resolved_path': probe['resolved_path'], 'existing_anchor': probe['anchor']})
    for group in groups.values():
        group['planned_additional_bytes'] = group['known_remaining_asset_bytes'] + group['estimated_additional_bytes']
        group['estimated_margin_bytes'] = group['free_bytes'] - group['planned_additional_bytes']
    return list(groups.values()), unknown


def cgroup_memory(cgroup_file=Path('/proc/self/cgroup'), mount=Path('/sys/fs/cgroup')):
    """Read cgroup-v2 memory bounds where mounted; never alter controller limits."""
    result = {'status': 'not_observed', 'limits': [], 'headroom_bytes': None, 'total_limit_bytes': None}
    try:
        unified = next((line[3:] for line in cgroup_file.read_text().splitlines() if line.startswith('0::')), None)
        if unified is None:
            return result
        mount = mount.resolve()
        candidate = (mount / unified.lstrip('/')).resolve()
        # A container may expose only its own cgroup at the mount root. Never
        # follow a namespace-relative ../ path outside the controller mount.
        current = candidate if candidate.is_relative_to(mount) and candidate.is_dir() else mount
        while current.is_relative_to(mount):
            maximum_path = current / 'memory.max'
            if maximum_path.is_file():
                maximum_text = maximum_path.read_text().strip()
                used = int((current / 'memory.current').read_text().strip())
                maximum = None if maximum_text == 'max' else int(maximum_text)
                if used < 0 or (maximum is not None and maximum <= 0):
                    raise ValueError('Invalid cgroup memory values')
                result['limits'].append({'path': str(current), 'max_bytes': maximum,
                                         'current_bytes': used,
                                         'headroom_bytes': None if maximum is None else max(0, maximum - used)})
            if current == mount:
                break
            current = current.parent
        finite = [entry for entry in result['limits'] if entry['max_bytes'] is not None]
        if finite:
            result['headroom_bytes'] = min(entry['headroom_bytes'] for entry in finite)
            result['total_limit_bytes'] = min(entry['max_bytes'] for entry in finite)
        if result['limits']:
            result['status'] = 'observed'
    except (OSError, ValueError, RuntimeError) as exc:
        result['status'] = 'unknown'
        result['error'] = str(exc)
    return result


def host_resources():
    memory = {}
    cpu_name = platform.processor() or None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu_name = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = None
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpus = None
    cgroup = cgroup_memory()
    available_bounds = [value for value in (memory.get('MemAvailable'), cgroup['headroom_bytes']) if value is not None]
    total_bounds = [value for value in (memory.get('MemTotal'), cgroup['total_limit_bytes']) if value is not None]
    return {'system': platform.system(), 'machine': platform.machine(), 'cpu': cpu_name,
            'logical_cpus': os.cpu_count(), 'affinity_cpus': affinity_cpus,
            'effective_logical_cpus': affinity_cpus or os.cpu_count(),
            'load_average_1_5_15_minutes': load,
            'host_ram_total_bytes': memory.get('MemTotal'), 'host_ram_available_bytes': memory.get('MemAvailable'),
            'ram_total_bytes': min(total_bounds) if total_bounds else None,
            'ram_available_bytes': min(available_bounds) if available_bounds else None,
            'ram_basis': 'Minimum of host MemAvailable and readable cgroup-v2 memory.max minus memory.current; cgroup cache reclamation is not assumed.',
            'cgroup_v2_memory': cgroup}


def gpu_resources(run=subprocess.run, which=shutil.which, environ=None):
    environ = os.environ if environ is None else environ
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    result = {'status': 'unknown', 'devices': [], 'selected_device': None,
              'cuda_visible_devices': visible, 'selection_note': '', 'error': None}
    binary = which('nvidia-smi')
    if not binary and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        binary = '/usr/lib/wsl/lib/nvidia-smi'
    if not binary:
        result['error'] = 'nvidia-smi is unavailable; video CUDA capability has not been established'
        return result
    try:
        process = run([binary, '--query-gpu=index,uuid,name,memory.total,memory.free',
                       '--format=csv,noheader,nounits'], capture_output=True, text=True,
                      timeout=5, check=True)
        for row in csv.reader(process.stdout.splitlines()):
            if len(row) != 5:
                raise ValueError('Unexpected nvidia-smi row')
            index, uuid, name, total, free = (part.strip() for part in row)
            device = {'index': int(index), 'uuid': uuid, 'name': name,
                      'total_bytes': int(total) * MIB, 'free_bytes': int(free) * MIB}
            if device['total_bytes'] <= 0 or not 0 <= device['free_bytes'] <= device['total_bytes']:
                raise ValueError('Invalid nvidia-smi memory values')
            result['devices'].append(device)
        if not result['devices']:
            raise ValueError('nvidia-smi returned no GPU devices')
        result['status'] = 'observed'
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result['error'] = f'GPU probe unavailable: {exc}'
        return result
    # CUDA enumeration can differ from NVML/nvidia-smi on multi-GPU machines.
    # Only a unique UUID or a single-GPU numeric selection gives certainty here.
    if visible is not None:
        first = visible.split(',')[0].strip()
        matches = [d for d in result['devices'] if first.startswith('GPU-') and d['uuid'].startswith(first)]
        if len(matches) == 1:
            result['selected_device'] = matches[0]
        elif len(result['devices']) == 1 and first == '0':
            result['selected_device'] = result['devices'][0]
        result['selection_note'] = 'CUDA_VISIBLE_DEVICES is respected; ambiguous numeric multi-GPU/MIG ordering needs a runtime check.'
    elif len(result['devices']) == 1:
        result['selected_device'] = result['devices'][0]
        result['selection_note'] = 'Single observed GPU; CUDA compatibility still requires the setup/runtime check.'
    else:
        result['selection_note'] = 'Multiple GPUs: the renderer CUDA default cannot be proven from nvidia-smi index ordering.'
    return result


def build_report(root, mode='video', duration=10, *, disk_inspect=filesystem_for, host=None, gpu=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('root must be an existing project directory')
    if mode not in ('audio', 'video'):
        raise ValueError('mode must be audio or video')
    duration = positive_seconds(duration)
    inventory = file_inventory(root, AUDIO_FILES | (VIDEO_FILES if mode == 'video' else {}))
    expanded = file_inventory(root, TOKENIZER_EXPANDED)
    allocations = [{'path': item['path'], 'bytes': item['remaining_payload_bytes'],
                    'kind': 'known_asset', 'reason': 'Pinned file absent or not the expected size'}
                   for item in inventory + expanded]
    envs = []
    missing_env_bytes = 0
    for name, allowance in ENVIRONMENT_ALLOWANCES.items():
        if name == '.venv-video' and mode == 'audio':
            continue
        present = (root / name / 'pyvenv.cfg').is_file() and (root / name / 'bin/python').is_file()
        estimated = 0 if present else allowance
        missing_env_bytes += estimated
        envs.append({'path': str(root / name), 'markers_present': present,
                     'estimated_remaining_install_bytes': estimated,
                     'integrity': 'not checked; markers alone do not establish compatibility'})
        allocations.append({'path': str(root / name), 'bytes': estimated, 'kind': 'estimate',
                            'reason': 'Environment allowance from observed installed footprint'})
    # A separate staging budget for compressed package archives, builds and
    # managed Python runtimes. Existing cache contents do not prove reusability.
    cache = math.ceil(missing_env_bytes * 0.5) + (GIB if missing_env_bytes else 256 * MIB)
    allocations.append({'path': str(root / '.cache/uv'), 'bytes': cache, 'kind': 'estimate',
                        'reason': 'Package/build/runtime cache: half missing environment allowance plus 1 GiB (256 MiB if environments present)'})
    allocations.append({'path': str(root / 'input'), 'bytes': math.ceil(duration * 12_500_000), 'kind': 'estimate',
                        'reason': 'Source-media allowance at 100 Mbit/s for the requested duration; longer or larger downloads need extra space'})
    frames = math.ceil(duration * 25) + 2
    frame_bytes = frames * 1280 * 720 * 3 * 3 if mode == 'video' else 0
    audio_work = math.ceil(duration * 24_000 * 4 * 10)
    allocations.append({'path': str(root / 'projects'), 'bytes': audio_work + math.ceil(duration * (12_500_000 if mode == 'video' else 0)) + 256 * MIB,
                        'kind': 'estimate', 'reason': 'Prepared compressed video at 100 Mbit/s, audio intermediates and 256 MiB workspace'})
    allocations.append({'path': str(root / 'outputs'), 'bytes': frame_bytes + math.ceil(duration * (25_000_000 if mode == 'video' else 1_000_000)) + 128 * MIB,
                        'kind': 'estimate', 'reason': 'Video: three raw RGB frame sets at 1280x720/25fps in the renderer output directory, plus encoded export allowance; bitrate and multiple takes can require more'})
    groups, unknown_disk = group_disk_allocations(allocations, disk_inspect)
    host = host_resources() if host is None else host
    gpu = gpu_resources() if gpu is None else gpu
    findings = []
    def finding(severity, code, message, action):
        findings.append({'severity': severity, 'code': code, 'message': message, 'next_action': action})
    for group in groups:
        if group['estimated_margin_bytes'] < 0:
            finding('insufficient', 'disk_budget', f"Filesystem {group['device']} falls short of the planning budget by {-group['estimated_margin_bytes']} bytes (estimates included).",
                    'Choose a filesystem with more free space, reduce the clip length, or explicitly choose files to remove; rerun this check.')
    if unknown_disk:
        finding('attention', 'disk_unknown', 'Some destination filesystems could not be inspected.', 'Inspect listed paths and permissions before installation or rendering.')
    if any(item['error'] for item in inventory + expanded):
        finding('attention', 'asset_probe_unknown', 'Some existing asset paths could not be inspected.', 'Inspect listed file permissions and symlinks before asset checks or setup.')
    if any(item['observed_bytes'] is not None and not item['size_matches'] for item in inventory + expanded):
        finding('attention', 'asset_size_mismatch', 'Some existing assets differ from pinned sizes; they have not been credited toward remaining payload.', 'Run the existing offline asset checks; preserve unexpected files for inspection.')
    if host.get('system') != 'Linux' or host.get('machine') not in ('x86_64', 'AMD64'):
        finding('attention', 'platform', 'The documented pinned setup is Linux x86_64/WSL2; this platform is not established compatible.', 'Use the documented Linux/WSL2 setup or assess platform support before installing.')
    ram = host.get('ram_available_bytes')
    # The upstream renderer retains decoded-frame lists. The two-copy estimate
    # gives duration a planning effect without presenting unprofiled peak usage
    # as a measured requirement; the GPU working set is not estimated here.
    ram_allowance = max(16 * GIB, 8 * GIB + frames * 1280 * 720 * 3 * 2) if mode == 'video' else 8 * GIB
    if host.get('cgroup_v2_memory', {}).get('status') == 'unknown':
        finding('attention', 'cgroup_memory_unknown', 'A cgroup-v2 memory bound could not be inspected completely.', 'Check container memory limits before relying on host RAM totals.')
    if ram is None:
        finding('attention', 'ram_unknown', 'Available system RAM could not be read.', 'Check free/available memory on the host and WSL limit before generation.')
    elif ram < ram_allowance:
        finding('attention', 'ram_headroom', f'Effective available RAM is below the {ram_allowance / GIB:.2f} GiB estimated planning allowance; this is not a measured minimum.', 'Reduce the clip duration, close other workloads yourself if appropriate, or use a short pilot while monitoring memory.')
    if not host.get('effective_logical_cpus', host.get('logical_cpus')):
        finding('attention', 'cpu_unknown', 'Logical CPU count is unknown.', 'Check host CPU resources before the CPU speech pilot.')
    if mode == 'video':
        selected = gpu.get('selected_device')
        if gpu.get('status') != 'observed' or selected is None:
            finding('attention', 'gpu_unknown', 'The renderer GPU or its available memory could not be established.', 'Check NVIDIA/WSL driver visibility and CUDA_VISIBLE_DEVICES; do not launch video rendering until the selected CUDA device is known.')
        elif selected['free_bytes'] < RENDERER_FREE_FLOOR:
            finding('insufficient', 'gpu_renderer_guard', 'The selected GPU has less than the renderer-enforced 4 GiB free-memory floor.', 'Ask the user to release another GPU workload or choose another device, then rerun; never terminate workloads automatically.')
        elif selected['free_bytes'] < 8 * GIB:
            finding('attention', 'gpu_headroom', 'The renderer 4 GiB free-memory guard clears, but free VRAM is below an 8 GiB planning allowance; model fit is unproven.', 'Use a short pilot after the offline setup check; monitor the actual CUDA device and stop on out-of-memory errors.')
    severity = {item['severity'] for item in findings}
    status = 'insufficient_resources' if 'insufficient' in severity else 'needs_attention' if severity else 'ready_to_try'
    return {'schema_version': 1, 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'root': str(root), 'mode': mode, 'duration_seconds': duration, 'status': status,
            'scope': 'Read-only capacity planning. Does not establish installed readiness, checksum integrity, CUDA compatibility, visual quality or runtime.',
            'inventory_measurement_date': '2026-09-30',
            'model_payload': {'expected_bytes': sum(item['expected_bytes'] for item in inventory),
                              'size_matched_bytes': sum(item['expected_bytes'] for item in inventory if item['size_matches']),
                              'remaining_bytes': sum(item['remaining_payload_bytes'] for item in inventory),
                              'integrity': 'not checked; size matches only', 'files': inventory},
            'tokenizer_expanded_files': expanded, 'environments': envs,
            'disk': {'filesystems': groups, 'unknown_allocations': unknown_disk,
                     'estimate_basis': 'One new experiment, sequential stages, 1280x720/25fps for video. Estimates are extra free-space allowances, not measured minima. Source downloads can fetch more than the selected interval; multiple takes, higher resolution, caches and existing partial installs can require more.'},
            'memory_planning': {'estimated_ram_allowance_bytes': ram_allowance,
                                'basis': 'Audio: 8 GiB. Video: max(16 GiB, 8 GiB plus two raw RGB frame sets at 1280x720/25fps). Conservative duration-aware estimate, not a measured peak or proven minimum.'},
            'host': host, 'gpu': gpu, 'findings': findings,
            'next_steps': ['Check dependency/asset integrity with the existing offline setup and model verification commands.',
                           'If capacity findings are resolved, generate and review a short audio pilot before video rendering.',
                           'Recheck resources immediately before each expensive stage; free RAM/VRAM/disk can change.']}


def human_report(report):
    lines = [f"Resource check: {report['status']} ({report['mode']}, {report['duration_seconds']:g} seconds)", report['scope'],
             f"Pinned model payload still absent by size: {report['model_payload']['remaining_bytes'] / 1e9:.3f} GB; integrity not checked."]
    for group in report['disk']['filesystems']:
        lines.append(f"Filesystem {group['device']}: {group['free_bytes'] / GIB:.2f} GiB free; {group['planned_additional_bytes'] / GIB:.2f} GiB additional planning allowance (includes estimates).")
        paths = sorted({item['existing_anchor'] for item in group['allocations']})
        lines.append('  Paths: ' + ', '.join(paths))
    host = report['host']
    lines.append(f"CPU: {host.get('cpu') or 'unknown'}; effective logical CPUs: {host.get('effective_logical_cpus', host.get('logical_cpus')) or 'unknown'}")
    ram = host.get('ram_available_bytes')
    lines.append(f"Effective available RAM: {ram / GIB:.2f} GiB" if ram is not None else 'Effective available RAM: unknown')
    for gpu in report['gpu']['devices']:
        lines.append(f"GPU {gpu['index']}: {gpu['name']}; {gpu['free_bytes'] / GIB:.2f}/{gpu['total_bytes'] / GIB:.2f} GiB free/total")
    if report['gpu'].get('error'):
        prefix = 'Optional GPU check (audio uses CPU): ' if report['mode'] == 'audio' else ''
        lines.append(prefix + report['gpu']['error'])
    for finding in report['findings']:
        lines.extend([f"{finding['severity']}: {finding['message']}", f"  Next: {finding['next_action']}"])
    lines.append('Use --json for per-path budgets, asset inventory and GPU selection details.')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root; no paths are created')
    parser.add_argument('--mode', choices=('audio', 'video'), default='video')
    parser.add_argument('--duration', type=float, default=10, help='Planned generated clip seconds, default 10')
    parser.add_argument('--json', action='store_true', help='Emit the full schema_version=1 report')
    args = parser.parse_args(argv)
    try:
        report = build_report(args.root, args.mode, args.duration)
    except (ValueError, OSError, OverflowError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2) if args.json else human_report(report))
    return {'ready_to_try': 0, 'needs_attention': 1, 'insufficient_resources': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
