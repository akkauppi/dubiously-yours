import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_resources as resources


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.host = {'system': 'Linux', 'machine': 'x86_64', 'cpu': 'Fixture CPU', 'logical_cpus': 16,
                     'ram_total_bytes': 32 * resources.GIB, 'ram_available_bytes': 24 * resources.GIB}
        self.device = {'index': 0, 'uuid': 'GPU-test', 'name': 'Fixture GPU',
                       'total_bytes': 12 * resources.GIB, 'free_bytes': 11 * resources.GIB}
        self.gpu = {'status': 'observed', 'devices': [self.device], 'selected_device': self.device,
                    'cuda_visible_devices': None, 'selection_note': 'fixture', 'error': None}

    def tearDown(self):
        self.temporary.cleanup()

    def disk(self, path):
        return {'device': 'fixture-volume', 'anchor': str(self.root), 'resolved_path': str(path),
                'free_bytes': 100 * resources.GIB, 'total_bytes': 200 * resources.GIB}

    def report(self, **kwargs):
        return resources.build_report(self.root, disk_inspect=self.disk, host=self.host,
                                      gpu=kwargs.pop('gpu', self.gpu), **kwargs)

    def test_fresh_setup_payload_matches_recorded_pins_and_writes_nothing(self):
        before = list(self.root.rglob('*'))
        report = self.report()
        snapshot = json.loads((ROOT / 'docs/performance-example.json').read_text())
        self.assertEqual(report['model_payload']['expected_bytes'], snapshot['total_payload_bytes'])
        self.assertEqual(report['model_payload']['remaining_bytes'], 8212540098)
        self.assertEqual(report['status'], 'ready_to_try')
        self.assertEqual(list(self.root.rglob('*')), before)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('chatterbox', sys.modules)
        self.assertIn('size matches only', report['model_payload']['integrity'])

    def test_missing_gpu_requires_attention_for_video_but_not_cpu_audio(self):
        unknown = {'status': 'unknown', 'devices': [], 'selected_device': None, 'error': 'unavailable'}
        report = self.report(gpu=unknown)
        self.assertEqual(report['status'], 'needs_attention')
        self.assertIn('gpu_unknown', {item['code'] for item in report['findings']})
        self.assertEqual(self.report(mode='audio', gpu=unknown)['status'], 'ready_to_try')

    def test_small_disk_blocks_even_with_ample_memory(self):
        def little_disk(path):
            return {**self.disk(path), 'free_bytes': resources.GIB}
        report = resources.build_report(self.root, disk_inspect=little_disk, host=self.host, gpu=self.gpu)
        self.assertEqual(report['status'], 'insufficient_resources')
        self.assertEqual(len(report['disk']['filesystems']), 1)
        self.assertLess(report['disk']['filesystems'][0]['estimated_margin_bytes'], 0)
        self.assertIn('estimates included', report['findings'][0]['message'])

    def test_same_filesystem_shares_one_free_pool_and_sums_allocations(self):
        allocations = [{'path': str(self.root / 'models'), 'bytes': 40, 'kind': 'known_asset'},
                       {'path': str(self.root / 'outputs'), 'bytes': 70, 'kind': 'estimate'}]
        def inspect(path):
            return {**self.disk(path), 'free_bytes': 100}
        groups, unknown = resources.group_disk_allocations(allocations, inspect)
        self.assertEqual(unknown, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['free_bytes'], 100)
        self.assertEqual(groups[0]['planned_additional_bytes'], 110)
        self.assertEqual(groups[0]['estimated_margin_bytes'], -10)

    def test_separate_filesystem_budget_is_not_hidden_by_another_volume(self):
        allocations = [{'path': 'models', 'bytes': 40, 'kind': 'known_asset'},
                       {'path': 'outputs', 'bytes': 70, 'kind': 'estimate'}]
        def inspect(path):
            return {**self.disk(path), 'device': path, 'free_bytes': 1000 if path == 'models' else 50}
        groups, _ = resources.group_disk_allocations(allocations, inspect)
        self.assertEqual({g['device']: g['estimated_margin_bytes'] for g in groups}, {'models': 960, 'outputs': -20})

    def test_decoded_frame_budget_is_charged_to_renderer_outputs_volume(self):
        def inspect(path):
            if Path(path) == self.root / 'outputs':
                return {**self.disk(path), 'device': 'small-outputs', 'free_bytes': 512 * resources.MIB}
            return self.disk(path)
        report = resources.build_report(self.root, disk_inspect=inspect, host=self.host, gpu=self.gpu)
        volumes = {group['device']: group for group in report['disk']['filesystems']}
        self.assertLess(volumes['small-outputs']['estimated_margin_bytes'], 0)
        self.assertGreater(volumes['fixture-volume']['estimated_margin_bytes'], 0)
        self.assertEqual(report['status'], 'insufficient_resources')
        self.assertIn('raw RGB frame sets', volumes['small-outputs']['allocations'][0]['reason'])

    def test_resolved_destination_follows_existing_directory_symlink(self):
        actual = self.root / 'actual'
        actual.mkdir()
        (self.root / 'models').symlink_to(actual, target_is_directory=True)
        result = resources.filesystem_for(self.root / 'models/new-model/weights.bin')
        self.assertEqual(result['resolved_path'], str(actual / 'new-model/weights.bin'))
        self.assertEqual(result['anchor'], str(actual))
        self.assertEqual(result['device'], str(actual.stat().st_dev))

    def test_installed_asset_reduces_budget_without_claiming_integrity(self):
        before = self.report(mode='audio')
        relative = 'models/chatterbox-multilingual-v2/t3_mtl23ls_v2.safetensors'
        path = self.root / relative
        path.parent.mkdir(parents=True)
        # Sparse fixture: allocated disk stays small; only size is being tested.
        with path.open('wb') as stream:
            stream.truncate(resources.AUDIO_FILES[relative])
        after = self.report(mode='audio')
        self.assertEqual(before['model_payload']['remaining_bytes'] - after['model_payload']['remaining_bytes'], resources.AUDIO_FILES[relative])
        self.assertEqual(after['model_payload']['integrity'], 'not checked; size matches only')
        self.assertLess(after['disk']['filesystems'][0]['planned_additional_bytes'], before['disk']['filesystems'][0]['planned_additional_bytes'])

    def test_partial_asset_is_not_credited_as_ready(self):
        relative = 'models/faster-whisper-small/config.json'
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_text('partial')
        report = self.report(mode='audio')
        item = next(item for item in report['model_payload']['files'] if item['path'] == str(path))
        self.assertFalse(item['size_matches'])
        self.assertEqual(item['remaining_payload_bytes'], resources.AUDIO_FILES[relative])
        self.assertIn('asset_size_mismatch', {item['code'] for item in report['findings']})

    def test_duration_increases_video_workspace_and_invalid_values_fail(self):
        short = self.report(duration=10)
        longer = self.report(duration=60)
        self.assertGreater(longer['disk']['filesystems'][0]['planned_additional_bytes'], short['disk']['filesystems'][0]['planned_additional_bytes'])
        self.assertEqual(short['model_payload']['remaining_bytes'], longer['model_payload']['remaining_bytes'])
        for duration in (0, -1, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, 'positive and finite'):
                self.report(duration=duration)

    def test_long_video_has_duration_dependent_ram_caution(self):
        short = self.report(duration=10)
        long = self.report(duration=300)
        self.assertNotIn('ram_headroom', {item['code'] for item in short['findings']})
        self.assertIn('ram_headroom', {item['code'] for item in long['findings']})
        self.assertGreater(long['memory_planning']['estimated_ram_allowance_bytes'], short['memory_planning']['estimated_ram_allowance_bytes'])

    def test_cgroup_uses_leaf_and_parent_headroom_without_counting_host_ram_twice(self):
        mount = self.root / 'cgroup'
        leaf = mount / 'job'
        leaf.mkdir(parents=True)
        cgroup_file = self.root / 'self-cgroup'
        cgroup_file.write_text('0::/job\n')
        for path, maximum, used in ((mount, 16, 10), (leaf, 8, 7)):
            (path / 'memory.max').write_text(str(maximum * resources.GIB))
            (path / 'memory.current').write_text(str(used * resources.GIB))
        result = resources.cgroup_memory(cgroup_file, mount)
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['headroom_bytes'], resources.GIB)
        self.assertEqual(result['total_limit_bytes'], 8 * resources.GIB)
        (leaf / 'memory.max').write_text('max')
        result = resources.cgroup_memory(cgroup_file, mount)
        self.assertEqual(result['headroom_bytes'], 6 * resources.GIB)

    def test_host_report_applies_cgroup_bound_and_cpu_affinity(self):
        cgroup = {'status': 'observed', 'limits': [], 'headroom_bytes': resources.GIB,
                  'total_limit_bytes': 4 * resources.GIB}
        def read(path, *args, **kwargs):
            if str(path) == '/proc/meminfo':
                return 'MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n'
            return 'model name: Fixture CPU\n'
        with patch.object(resources, 'cgroup_memory', return_value=cgroup), \
                patch.object(Path, 'read_text', read), \
                patch.object(resources.os, 'cpu_count', return_value=16), \
                patch.object(resources.os, 'sched_getaffinity', return_value={0, 1}):
            host = resources.host_resources()
        self.assertEqual(host['host_ram_available_bytes'], 24 * resources.GIB)
        self.assertEqual(host['ram_available_bytes'], resources.GIB)
        self.assertEqual(host['ram_total_bytes'], 4 * resources.GIB)
        self.assertEqual(host['logical_cpus'], 16)
        self.assertEqual(host['effective_logical_cpus'], 2)

    def test_asset_probe_error_prevents_ready_status(self):
        actual_inventory = resources.file_inventory
        def failed(root, files):
            records = actual_inventory(root, files)
            records[0]['error'] = 'fixture permission denied'
            return records
        with patch.object(resources, 'file_inventory', side_effect=failed):
            report = self.report()
        self.assertEqual(report['status'], 'needs_attention')
        self.assertIn('asset_probe_unknown', {item['code'] for item in report['findings']})

    def test_renderer_floor_differs_from_advisory_headroom(self):
        self.device['free_bytes'] = 3 * resources.GIB
        self.assertEqual(self.report()['status'], 'insufficient_resources')
        self.device['free_bytes'] = 5 * resources.GIB
        report = self.report()
        self.assertEqual(report['status'], 'needs_attention')
        self.assertIn('gpu_headroom', {item['code'] for item in report['findings']})
        self.device['free_bytes'] = 11 * resources.GIB
        self.assertEqual(self.report()['status'], 'ready_to_try')

    def test_gpu_probe_is_bounded_and_timeout_is_unknown(self):
        def timeout(command, **kwargs):
            self.assertEqual(kwargs['timeout'], 5)
            self.assertNotIn('shell', kwargs)
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        result = resources.gpu_resources(run=timeout, which=lambda _: '/fixture/nvidia-smi', environ={})
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['selected_device'])
        self.assertIn('GPU probe unavailable', result['error'])
        with patch.object(Path, 'is_file', return_value=False):
            self.assertEqual(resources.gpu_resources(which=lambda _: None, environ={})['status'], 'unknown')

    def test_gpu_selection_does_not_choose_arbitrary_free_device(self):
        def run(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, '0, GPU-alpha, GPU A, 12288, 1024\n1, GPU-beta, GPU B, 24576, 22000\n', '')
        result = resources.gpu_resources(run=run, which=lambda _: '/fixture/nvidia-smi', environ={})
        self.assertIsNone(result['selected_device'])
        explicit = resources.gpu_resources(run=run, which=lambda _: '/fixture/nvidia-smi', environ={'CUDA_VISIBLE_DEVICES': 'GPU-alpha'})
        self.assertEqual(explicit['selected_device']['index'], 0)
        self.assertEqual(explicit['selected_device']['free_bytes'], resources.GIB)
        numeric = resources.gpu_resources(run=run, which=lambda _: '/fixture/nvidia-smi', environ={'CUDA_VISIBLE_DEVICES': '1'})
        self.assertIsNone(numeric['selected_device'])

    def test_unknown_disk_and_ram_require_attention(self):
        def unknown(path):
            raise PermissionError('fixture unreadable')
        self.host['ram_available_bytes'] = None
        report = resources.build_report(self.root, disk_inspect=unknown, host=self.host, gpu=self.gpu)
        self.assertEqual(report['status'], 'needs_attention')
        codes = {item['code'] for item in report['findings']}
        self.assertTrue({'disk_unknown', 'ram_unknown'}.issubset(codes))


if __name__ == '__main__':
    unittest.main()
