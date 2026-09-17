from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from data_toolkit.pipeline import resources
from data_toolkit.pipeline.config import load_config
from data_toolkit.pipeline.runtime import (
    NoFollowTelemetryWriter,
    initialize_project_accounting,
)
from data_toolkit.pipeline.resources import (
    GpuMetric,
    ProjectStorageAccounting,
    ResourceMonitor,
    ResourceSampler,
    ResourceSnapshot,
)


def _record_accounting_delta_after_release(
    config_path, target, delta, ready, release
):
    config = load_config(Path(config_path))
    accounting = initialize_project_accounting(
        config,
        directory_size=lambda _root: 0,
    )
    ready.put(True)
    if not release.wait(timeout=10):
        raise RuntimeError("accounting test release timed out")
    accounting.record_registry_delta(Path(target), delta)


def test_resource_monitor_serializes_parallel_chunk_sampling():
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(5)

    def sampler():
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        return sample(datetime.now(timezone.utc), monotonic_seconds=time.monotonic())

    class Writer:
        def write(self, *_args):
            pass

    monitor = ResourceMonitor(
        sampler,
        Writer(),
    )

    def worker(index):
        barrier.wait()
        monitor.record("ABO-00000", f"chunk-{index}")

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert state["peak"] == 1


def sample(now, **changes):
    base = ResourceSnapshot(
        now,
        20.0,
        10.0,
        1.0,
        400.0,
        0,
        300.0,
        40.0,
        1.0,
        20.0,
        1.0,
        30.0,
    )
    return replace(base, **changes)


class FakePsutil:
    def __init__(self, roots, *, local_free_gib=300.0):
        self.roots = roots
        self.local_free_gib = local_free_gib
        self.swap_values = iter((10_000, 14_096))

    def cpu_percent(self, interval=None):
        assert interval is None
        return 37.5

    def cpu_times_percent(self, interval=None):
        assert interval is None
        return SimpleNamespace(iowait=2.5)

    def getloadavg(self):
        return (12.0, 10.0, 8.0)

    def virtual_memory(self):
        return SimpleNamespace(available=200 * 1024**3)

    def swap_memory(self):
        return SimpleNamespace(sin=next(self.swap_values))

    def disk_usage(self, path):
        if Path(path) == self.roots.local_root:
            return SimpleNamespace(
                total=1_000 * 1024**3,
                free=int(self.local_free_gib * 1024**3),
            )
        if Path(path) == self.roots.data2_root:
            return SimpleNamespace(total=30 * 1024**4, free=20 * 1024**4)
        if Path(path) == self.roots.data3_root:
            return SimpleNamespace(total=40 * 1024**4, free=30 * 1024**4)
        raise AssertionError(path)


def test_sampler_uses_cached_project_bytes_and_swap_delta(config, tmp_path):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    walk_calls = []

    def directory_size(root):
        walk_calls.append(root)
        return 7 if root == roots.data2_root else 11

    accounting = ProjectStorageAccounting(
        roots.data2_root,
        roots.data3_root,
        data2_bytes=1 * 1024**4,
        data3_bytes=2 * 1024**4,
        directory_size=directory_size,
    )
    runner_calls = []

    def gpu_runner(argv, **kwargs):
        runner_calls.append((argv, kwargs))
        return SimpleNamespace(stdout="0, 75, 1234, 97887, 55, 210.5\n")

    now = datetime(2026, 7, 16, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    monotonic_values = iter((100.0, 105.0))
    sampler = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots),
        gpu_runner=gpu_runner,
        clock=lambda: now,
        monotonic_clock=lambda: next(monotonic_values),
    )

    first = sampler()
    second = sampler()

    assert first.swap_in_bytes == 0
    assert second.swap_in_bytes == 4096
    assert first.timestamp == datetime(2026, 7, 15, 18, 30, tzinfo=timezone.utc)
    assert first.monotonic_seconds == 100.0
    assert second.monotonic_seconds == 105.0
    assert first.data2_project_tib == 1.0
    assert first.data3_project_tib == 2.0
    assert first.gpu_metrics[0].power_watts == 210.5
    assert first.gpu_metrics[0].memory_total_mib == 97887
    assert first.gpu_query_error is None
    assert walk_calls == []
    assert runner_calls[0] == (
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        {"capture_output": True, "text": True, "check": True, "timeout": 5.0},
    )

    assert sampler.reconcile_at_shard_boundary() == (7, 11)
    assert walk_calls == [roots.data2_root, roots.data3_root]


def test_project_accounting_accepts_registry_deltas_without_walking(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=lambda root: pytest.fail("unexpected directory walk"),
    )

    accounting.record_registry_delta(data2 / "prepared" / "pack.tar", 25)
    accounting.record_registry_delta(data3 / "archive" / "raw.tar", -50)

    assert accounting.current_bytes() == (125, 150)
    with pytest.raises(ValueError, match="outside configured project roots"):
        accounting.record_registry_delta(tmp_path / "other" / "file", 1)
    with pytest.raises(ValueError, match="negative project accounting"):
        accounting.record_registry_delta(data2 / "prepared" / "pack.tar", -126)


def test_project_accounting_reuses_persisted_totals_without_walking(tmp_config):
    config = load_config(tmp_config)
    config.paths.data2_root.mkdir(parents=True)
    config.paths.data3_root.mkdir(parents=True)
    path = config.paths.data2_root / "control/accounting.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"artifact_type":"project_accounting",'
        f'"config_hash":"{config.config_hash()}",'
        '"data2_bytes":123,"data3_bytes":456,"schema_version":1}'
    )

    accounting = initialize_project_accounting(
        config,
        directory_size=lambda root: pytest.fail(f"unexpected walk: {root}"),
    )

    assert accounting.current_bytes() == (123, 456)


def test_persistent_project_accounting_merges_concurrent_process_deltas(
    tmp_config,
):
    config = load_config(tmp_config)
    config.paths.data2_root.mkdir(parents=True)
    config.paths.data3_root.mkdir(parents=True)
    accounting = initialize_project_accounting(
        config,
        directory_size=lambda _root: 0,
    )
    assert accounting.current_bytes() == (0, 0)
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    release = context.Event()
    processes = [
        context.Process(
            target=_record_accounting_delta_after_release,
            args=(
                tmp_config,
                config.paths.data2_root / f"pack-{index}.tar",
                delta,
                ready,
                release,
            ),
        )
        for index, delta in enumerate((11, 17))
    ]
    for process in processes:
        process.start()
    for _ in processes:
        assert ready.get(timeout=10) is True
    release.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    persisted = json.loads(
        (config.paths.data2_root / "control/accounting.json").read_text()
    )
    assert persisted["data2_bytes"] == 28
    assert persisted["data3_bytes"] == 0


def test_project_accounting_scans_once_when_persisted_totals_are_missing(
    tmp_config,
):
    config = load_config(tmp_config)
    config.paths.data2_root.mkdir(parents=True)
    config.paths.data3_root.mkdir(parents=True)
    sizes = {
        config.paths.data2_root.resolve(): 123,
        config.paths.data3_root.resolve(): 456,
    }
    walked = []

    def directory_size(root):
        walked.append(root)
        return sizes[root]

    accounting = initialize_project_accounting(
        config, directory_size=directory_size
    )

    assert accounting.current_bytes() == (123, 456)
    assert walked == [
        config.paths.data2_root.resolve(),
        config.paths.data3_root.resolve(),
    ]


@pytest.mark.parametrize(
    ("data2_bytes", "data3_bytes"), [(None, None), (0, None), (None, 0)]
)
def test_project_accounting_requires_initialized_registry_totals(
    tmp_path, data2_bytes, data3_bytes
):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()

    with pytest.raises(RuntimeError, match="registry totals must be initialized"):
        ProjectStorageAccounting(
            data2, data3, data2_bytes=data2_bytes, data3_bytes=data3_bytes
        )


def test_project_accounting_rejects_missing_or_non_directory_roots(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.write_text("not a directory")

    with pytest.raises(RuntimeError, match="project root is not a directory"):
        ProjectStorageAccounting(data2, data3, data2_bytes=0, data3_bytes=0)
    with pytest.raises(RuntimeError, match="project root is not a directory"):
        ProjectStorageAccounting(
            tmp_path / "missing", data2, data2_bytes=0, data3_bytes=0
        )


def test_directory_size_reraises_walk_errors(tmp_path, monkeypatch):
    root = tmp_path / "data2"
    root.mkdir()

    def failing_walk(path, *, onerror):
        onerror(PermissionError("walk denied"))
        return ()

    monkeypatch.setattr(resources.os, "walk", failing_walk)

    with pytest.raises(PermissionError, match="walk denied"):
        resources._directory_size(root)


def test_reconciliation_failure_does_not_partially_publish_cache(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()

    def directory_size(root):
        if root == data2:
            return 1_000
        raise OSError("data3 walk failed")

    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=directory_size,
    )

    with pytest.raises(OSError, match="data3 walk failed"):
        accounting.reconcile_at_shard_boundary()

    assert accounting.current_bytes() == (100, 200)


def test_reconciliation_walks_outside_lock_and_rejects_concurrent_delta(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    walk_started = threading.Event()
    release_walk = threading.Event()

    def directory_size(root):
        if root == data2:
            walk_started.set()
            assert release_walk.wait(timeout=2)
            return 1_000
        return 2_000

    accounting = ProjectStorageAccounting(
        data2,
        data3,
        data2_bytes=100,
        data3_bytes=200,
        directory_size=directory_size,
    )
    errors = []

    def reconcile():
        try:
            accounting.reconcile_at_shard_boundary()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=reconcile)
    worker.start()
    assert walk_started.wait(timeout=2)
    accounting.record_registry_delta(data2 / "prepared" / "new.tar", 25)
    release_walk.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "accounting changed during reconciliation" in str(errors[0])
    assert accounting.current_bytes() == (125, 200)


def test_parallel_registry_deltas_are_not_lost(tmp_path):
    data2 = tmp_path / "data2"
    data3 = tmp_path / "data3"
    data2.mkdir()
    data3.mkdir()
    accounting = ProjectStorageAccounting(
        data2, data3, data2_bytes=0, data3_bytes=0
    )
    start = threading.Barrier(9)

    def add_deltas():
        start.wait()
        for _ in range(1_000):
            accounting.record_registry_delta(data2 / "prepared" / "pack.tar", 1)

    workers = [threading.Thread(target=add_deltas) for _ in range(8)]
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for worker in workers:
            worker.start()
        start.wait()
        for worker in workers:
            worker.join(timeout=5)
    finally:
        sys.setswitchinterval(previous_interval)

    assert all(not worker.is_alive() for worker in workers)
    assert accounting.current_bytes() == (8_000, 0)


def test_gpu_query_failure_is_reported(config, tmp_path):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    accounting = ProjectStorageAccounting(
        roots.data2_root, roots.data3_root, data2_bytes=0, data3_bytes=0
    )

    def gpu_runner(argv, **kwargs):
        raise OSError("driver unavailable")

    snapshot = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots, local_free_gib=119.0),
        gpu_runner=gpu_runner,
        clock=lambda: datetime(2026, 7, 16, tzinfo=timezone.utc),
    )()

    assert snapshot.gpu_metrics == ()
    assert snapshot.gpu_query_error == "driver unavailable"


def test_gpu_query_timeout_is_bounded(config, tmp_path):
    roots = replace(
        config.paths,
        data2_root=tmp_path / "data2",
        data3_root=tmp_path / "data3",
        local_root=tmp_path / "local",
    )
    config = replace(config, paths=roots)
    roots.data2_root.mkdir()
    roots.data3_root.mkdir()
    accounting = ProjectStorageAccounting(
        roots.data2_root, roots.data3_root, data2_bytes=0, data3_bytes=0
    )
    calls = []

    def gpu_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    snapshot = ResourceSampler(
        config,
        accounting,
        psutil_api=FakePsutil(roots, local_free_gib=119.0),
        gpu_runner=gpu_runner,
        clock=lambda: datetime(2026, 7, 16, tzinfo=timezone.utc),
    )()

    assert calls[0][1]["timeout"] == 5.0
    assert "timed out after 5.0 seconds" in snapshot.gpu_query_error


def test_no_follow_telemetry_publishes_each_shared_append_immediately(tmp_path):
    now = datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc)
    path = tmp_path / "telemetry.jsonl"
    first = NoFollowTelemetryWriter(
        path, clock=lambda: 0.0, sync_interval=3600
    )
    second = NoFollowTelemetryWriter(
        path, clock=lambda: 0.0, sync_interval=3600
    )

    first.write(sample(now), "ABO-00000", "download")
    second.write(sample(now), "HSSD-00000", "download")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["shard_id"] for record in records] == [
        "ABO-00000",
        "HSSD-00000",
    ]
    assert {record["action"] for record in records} == {"run"}
    assert all(not record["reasons"] for record in records)
    first.close()
    second.close()


def test_resource_monitor_history_is_bounded_and_json_serializable():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    samples = iter(
        sample(start, monotonic_seconds=float(index))
        for index in range(65)
    )

    class RecordingTelemetry:
        def write(self, snapshot, shard_id, command):
            return None

    monitor = ResourceMonitor(lambda: next(samples), RecordingTelemetry())

    for _ in range(65):
        monitor.record("shard", "command")

    history = monitor.last_five_minutes()
    assert len(history) == 60
    assert "monotonic_seconds" not in history[0]
    json.dumps(history)
