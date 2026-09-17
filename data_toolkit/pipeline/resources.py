from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Protocol

import psutil

from .config import PipelineConfig


GIB = 1024**3
TIB = 1024**4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class GpuMetric:
    index: int
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_celsius: float
    power_watts: float


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: datetime
    cpu_percent: float
    load_1m: float
    io_wait_percent: float
    available_ram_gib: float
    swap_in_bytes: int
    local_free_gib: float
    local_free_percent: float
    data2_project_tib: float
    data2_fs_free_tib: float
    data3_project_tib: float
    data3_fs_free_tib: float
    gpu_metrics: tuple[GpuMetric, ...] = ()
    gpu_query_error: str | None = None
    monotonic_seconds: float | None = None
    cpu_max_temperature_celsius: float | None = None


class ResourceAccountingError(RuntimeError):
    pass


class ResourceAccountingConflict(ResourceAccountingError):
    pass


def _raise_walk_error(error: OSError) -> None:
    raise error


def _directory_size(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root, onerror=_raise_walk_error):
        for filename in filenames:
            total += (Path(directory) / filename).stat().st_size
    return total


def _validated_project_root(root: Path) -> Path:
    try:
        resolved = Path(root).resolve(strict=True)
    except OSError as error:
        raise ResourceAccountingError(
            f"project root is not a directory: {root}"
        ) from error
    if not resolved.is_dir():
        raise ResourceAccountingError(f"project root is not a directory: {root}")
    return resolved


class ProjectStorageAccounting:
    """Cached registry totals, reconciled against disk only at shard boundaries."""

    def __init__(
        self,
        data2_root: Path,
        data3_root: Path,
        *,
        data2_bytes: int | None = None,
        data3_bytes: int | None = None,
        directory_size: Callable[[Path], int] = _directory_size,
    ):
        if data2_bytes is None or data3_bytes is None:
            raise ResourceAccountingError("registry totals must be initialized")
        if data2_bytes < 0 or data3_bytes < 0:
            raise ResourceAccountingError("negative project accounting")
        self.data2_root = _validated_project_root(data2_root)
        self.data3_root = _validated_project_root(data3_root)
        if self.data2_root == self.data3_root:
            raise ResourceAccountingError("project roots must be distinct")
        self._data2_bytes = data2_bytes
        self._data3_bytes = data3_bytes
        self._directory_size = directory_size
        self._lock = threading.Lock()
        self._version = 0

    def current_bytes(self) -> tuple[int, int]:
        with self._lock:
            return self._data2_bytes, self._data3_bytes

    def replace_current_bytes(self, data2_bytes: int, data3_bytes: int) -> None:
        if data2_bytes < 0 or data3_bytes < 0:
            raise ValueError("negative project accounting")
        with self._lock:
            self._data2_bytes = data2_bytes
            self._data3_bytes = data3_bytes
            self._version += 1

    def record_registry_delta(self, path: Path, delta_bytes: int) -> None:
        candidate = Path(path).resolve()
        for attribute, root in (
            ("_data2_bytes", self.data2_root),
            ("_data3_bytes", self.data3_root),
        ):
            if candidate == root or root in candidate.parents:
                with self._lock:
                    updated = getattr(self, attribute) + delta_bytes
                    if updated < 0:
                        raise ValueError("negative project accounting")
                    setattr(self, attribute, updated)
                    self._version += 1
                return
        raise ValueError(f"path outside configured project roots: {path}")

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        with self._lock:
            version = self._version
        data2_bytes = self._directory_size(self.data2_root)
        data3_bytes = self._directory_size(self.data3_root)
        if data2_bytes < 0 or data3_bytes < 0:
            raise ResourceAccountingError("negative reconciliation result")
        with self._lock:
            if self._version != version:
                raise ResourceAccountingConflict(
                    "accounting changed during reconciliation"
                )
            self._data2_bytes = data2_bytes
            self._data3_bytes = data3_bytes
            self._version += 1
            return self._data2_bytes, self._data3_bytes


class ResourceSampler:
    GPU_QUERY_TIMEOUT_SECONDS = 5.0
    GPU_QUERY = (
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )

    def __init__(
        self,
        config: PipelineConfig,
        project_accounting: ProjectStorageAccounting,
        *,
        psutil_api=psutil,
        gpu_runner=subprocess.run,
        clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.project_accounting = project_accounting
        self.psutil = psutil_api
        self.gpu_runner = gpu_runner
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self._previous_swap_in: int | None = None

    def _gpu_metrics(self) -> tuple[tuple[GpuMetric, ...], str | None]:
        try:
            completed = self.gpu_runner(
                list(self.GPU_QUERY),
                capture_output=True,
                text=True,
                check=True,
                timeout=self.GPU_QUERY_TIMEOUT_SECONDS,
            )
            metrics = []
            for line in completed.stdout.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 6:
                    raise ValueError(f"unexpected nvidia-smi row: {line!r}")
                metrics.append(
                    GpuMetric(
                        index=int(fields[0]),
                        utilization_percent=float(fields[1]),
                        memory_used_mib=float(fields[2]),
                        memory_total_mib=float(fields[3]),
                        temperature_celsius=float(fields[4]),
                        power_watts=float(fields[5]),
                    )
                )
            return tuple(metrics), None
        except Exception as error:
            return (), str(error) or type(error).__name__

    def _cpu_max_temperature(self) -> float | None:
        sensors = getattr(self.psutil, "sensors_temperatures", None)
        if sensors is None:
            return None
        try:
            readings = sensors()
        except Exception:
            return None
        values = []
        for entries in readings.values():
            for entry in entries:
                current = getattr(entry, "current", None)
                if isinstance(current, (int, float)) and current == current:
                    values.append(float(current))
        return max(values) if values else None

    def reconcile_at_shard_boundary(self) -> tuple[int, int]:
        return self.project_accounting.reconcile_at_shard_boundary()

    def __call__(self) -> ResourceSnapshot:
        paths = self.config.paths
        cpu_times = self.psutil.cpu_times_percent(interval=None)
        memory = self.psutil.virtual_memory()
        swap_in = self.psutil.swap_memory().sin
        swap_delta = (
            0
            if self._previous_swap_in is None
            else max(0, swap_in - self._previous_swap_in)
        )
        self._previous_swap_in = swap_in
        local_usage = self.psutil.disk_usage(paths.local_root)
        data2_usage = self.psutil.disk_usage(paths.data2_root)
        data3_usage = self.psutil.disk_usage(paths.data3_root)
        data2_bytes, data3_bytes = self.project_accounting.current_bytes()
        gpu_metrics, gpu_error = self._gpu_metrics()
        timestamp = self.clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("resource timestamp must be timezone-aware")
        local_free_percent = (
            100.0 * local_usage.free / local_usage.total if local_usage.total else 0.0
        )
        return ResourceSnapshot(
            timestamp=timestamp.astimezone(timezone.utc),
            cpu_percent=self.psutil.cpu_percent(interval=None),
            load_1m=self.psutil.getloadavg()[0],
            io_wait_percent=getattr(cpu_times, "iowait", 0.0),
            available_ram_gib=memory.available / GIB,
            swap_in_bytes=swap_delta,
            local_free_gib=local_usage.free / GIB,
            local_free_percent=local_free_percent,
            data2_project_tib=data2_bytes / TIB,
            data2_fs_free_tib=data2_usage.free / TIB,
            data3_project_tib=data3_bytes / TIB,
            data3_fs_free_tib=data3_usage.free / TIB,
            gpu_metrics=gpu_metrics,
            gpu_query_error=gpu_error,
            monotonic_seconds=self.monotonic_clock(),
            cpu_max_temperature_celsius=self._cpu_max_temperature(),
        )


def _snapshot_payload(snapshot: ResourceSnapshot) -> dict:
    payload = asdict(snapshot)
    timestamp = snapshot.timestamp
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("resource timestamp must be timezone-aware")
    payload["timestamp"] = timestamp.astimezone(timezone.utc).isoformat()
    payload.pop("monotonic_seconds", None)
    return payload


class ResourceTelemetryWriter(Protocol):
    def write(
        self,
        snapshot: ResourceSnapshot,
        shard_id: str,
        command: str,
    ) -> None: ...


class ResourceMonitor:
    """Serializes passive resource samples without controlling execution."""

    def __init__(
        self,
        sample: Callable[[], ResourceSnapshot],
        telemetry_writer: ResourceTelemetryWriter,
    ) -> None:
        self.sample = sample
        self.telemetry_writer = telemetry_writer
        self._snapshots = deque(maxlen=60)
        self._lock = threading.RLock()

    def record(self, shard_id: str, command: str) -> ResourceSnapshot:
        with self._lock:
            snapshot = self.sample()
            self._snapshots.append(snapshot)
            self.telemetry_writer.write(snapshot, shard_id, command)
            return snapshot

    def last_five_minutes(self) -> tuple[dict, ...]:
        with self._lock:
            return tuple(
                _snapshot_payload(snapshot) for snapshot in self._snapshots
            )
