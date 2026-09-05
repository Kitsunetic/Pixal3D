"""Process-safe ephemeral caches for geometry inputs shared by view workers."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable, TypeVar


T = TypeVar("T")
_CACHE_ROOT = Path(os.environ.get(
    "PIXAL3D_GEOMETRY_INPUT_CACHE_DIR", "/tmp/pixal3d_geometry_input_cache"
))


def _source_signature(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_size, status.st_mtime_ns


def load_or_prepare(
    kind: str,
    sha256: str,
    source: Path,
    prepare: Callable[[], T],
) -> T:
    """Load one prepared input or create it once across concurrent view workers."""
    directory = _CACHE_ROOT / kind
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / f"{sha256}.pickle"
    lock_path = directory / f"{sha256}.lock"
    signature = _source_signature(source)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            try:
                with cache_path.open("rb") as cache_file:
                    cached_signature, value = pickle.load(cache_file)
                if cached_signature == signature:
                    return value
            except (EOFError, OSError, pickle.PickleError, ValueError):
                pass

            value = prepare()
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=f".{sha256}.",
                suffix=".pickle",
                delete=False,
            ) as cache_file:
                temporary = Path(cache_file.name)
                pickle.dump((signature, value), cache_file, protocol=pickle.HIGHEST_PROTOCOL)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temporary, cache_path)
            return value
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
