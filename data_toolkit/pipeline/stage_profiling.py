"""Opt-in timing records for pipeline stages."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Final, ParamSpec, TypeVar

if TYPE_CHECKING:
    import torch


_PROFILE_ENV: Final = "PIXAL3D_PROFILE_STAGES"
_TRUTHY_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_PROFILE_PREFIX: Final = "[PIXAL3D_PROFILE] "
_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


def _profiling_enabled() -> bool:
    return os.environ.get(_PROFILE_ENV, "").strip().lower() in _TRUTHY_VALUES


def _milliseconds(elapsed_nanoseconds: int) -> float:
    return elapsed_nanoseconds / 1_000_000


def _emit(record: dict[str, float | str]) -> None:
    print(
        f"{_PROFILE_PREFIX}{json.dumps(record, sort_keys=True, separators=(',', ':'))}",
        flush=True,
    )


@contextmanager
def profile_stage(
    name: str, device: str | int | torch.device | None = None
) -> Generator[None, None, None]:
    """Emit timing data for one stage when ``PIXAL3D_PROFILE_STAGES`` is truthy."""
    if not _profiling_enabled():
        yield
        return

    if device is None:
        with _profile_cpu_stage(name):
            yield
        return

    with _profile_cuda_stage(name, device):
        yield


def profiled_stage(
    name: str,
) -> Callable[[Callable[_Parameters, _Result]], Callable[_Parameters, _Result]]:
    """Decorate a CPU function to emit an opt-in stage timing record."""

    def decorate(
        function: Callable[_Parameters, _Result],
    ) -> Callable[_Parameters, _Result]:
        @wraps(function)
        def wrapper(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            with profile_stage(name):
                return function(*args, **kwargs)

        return wrapper

    return decorate


def profile_call(
    name: str,
    function: Callable[_Parameters, _Result],
    /,
    *args: _Parameters.args,
    **kwargs: _Parameters.kwargs,
) -> _Result:
    """Call a CPU function inside an opt-in profiling stage."""
    with profile_stage(name):
        return function(*args, **kwargs)


@contextmanager
def _profile_cpu_stage(name: str) -> Generator[None, None, None]:
    wall_started_at = time.perf_counter_ns()
    process_started_at = time.process_time_ns()
    thread_started_at = time.thread_time_ns()
    try:
        yield
    finally:
        _emit(
            {
                "name": name,
                "process_cpu_ms": _milliseconds(
                    time.process_time_ns() - process_started_at
                ),
                "thread_cpu_ms": _milliseconds(
                    time.thread_time_ns() - thread_started_at
                ),
                "wall_ms": _milliseconds(time.perf_counter_ns() - wall_started_at),
            }
        )


@contextmanager
def _profile_cuda_stage(
    name: str, device: str | int | torch.device
) -> Generator[None, None, None]:
    from torch import cuda

    wall_started_at = time.perf_counter_ns()
    process_started_at = time.process_time_ns()
    thread_started_at = time.thread_time_ns()

    pre_sync_started_at = time.perf_counter_ns()
    cuda.synchronize(device)
    pre_sync_ms = _milliseconds(time.perf_counter_ns() - pre_sync_started_at)

    started = cuda.Event(enable_timing=True)
    finished = cuda.Event(enable_timing=True)
    started.record()
    host_submit_started_at = time.perf_counter_ns()
    try:
        yield
    finally:
        host_submit_ms = _milliseconds(time.perf_counter_ns() - host_submit_started_at)
        finished.record()
        sync_wait_started_at = time.perf_counter_ns()
        cuda.synchronize(device)
        sync_wait_ms = _milliseconds(time.perf_counter_ns() - sync_wait_started_at)
        _emit(
            {
                "cuda_elapsed_ms": started.elapsed_time(finished),
                "host_submit_ms": host_submit_ms,
                "name": name,
                "pre_sync_ms": pre_sync_ms,
                "process_cpu_ms": _milliseconds(
                    time.process_time_ns() - process_started_at
                ),
                "sync_wait_ms": sync_wait_ms,
                "thread_cpu_ms": _milliseconds(
                    time.thread_time_ns() - thread_started_at
                ),
                "wall_ms": _milliseconds(time.perf_counter_ns() - wall_started_at),
            }
        )
