"""Bounded subprocess fan-out with fail-fast sibling cleanup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import subprocess
import time
from typing import Protocol


class Process(Protocol):
    @property
    def args(self) -> object: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


Command = tuple[Sequence[str], Mapping[str, str] | None]


def terminate_and_reap(processes: Sequence[Process]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    survivors: list[Process] = []
    for process in running:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            survivors.append(process)
    for process in survivors:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    kill_deadline = time.monotonic() + 5.0
    for process in survivors:
        process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))


def wait_until(
    processes: Sequence[Process], required: Sequence[Process]
) -> None:
    while True:
        for process in processes:
            status = process.poll()
            if status not in (None, 0):
                terminate_and_reap(processes)
                raise subprocess.CalledProcessError(status, str(process.args))
        if all(process.poll() == 0 for process in required):
            return
        time.sleep(0.1)


def wait_all(processes: Sequence[Process]) -> None:
    wait_until(processes, processes)


def run_bounded(commands: Sequence[Command], max_processes: int) -> None:
    if max_processes <= 0:
        raise ValueError("max_processes must be positive")
    queued = list(commands)
    active: list[Process] = []
    try:
        while queued or active:
            while queued and len(active) < max_processes:
                argv, environment = queued.pop(0)
                active.append(subprocess.Popen(argv, env=environment))
            for process in tuple(active):
                status = process.poll()
                if status is None:
                    continue
                active.remove(process)
                if status != 0:
                    raise subprocess.CalledProcessError(status, str(process.args))
            if active:
                time.sleep(0.1)
    except BaseException:
        terminate_and_reap(active)
        raise
