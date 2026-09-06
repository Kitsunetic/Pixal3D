"""Opt-in wait used when encoders are pipelined with geometry producers."""

from __future__ import annotations

import os
from pathlib import Path
import time


def wait_for_geometry(vxz: Path, scale: Path, cancel_event) -> None:
    raw_timeout = os.environ.get("PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS")
    if raw_timeout is None:
        return
    timeout = float(raw_timeout)
    if timeout <= 0:
        raise ValueError("PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS must be positive")
    deadline = time.monotonic() + timeout
    while not (vxz.is_file() and scale.is_file()):
        if cancel_event.is_set():
            raise TimeoutError("geometry wait cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"geometry input timed out: {vxz}")
        time.sleep(0.05)
