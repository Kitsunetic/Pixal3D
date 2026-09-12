"""Keep securely opened asset descriptors alive across Blender exec."""

from __future__ import annotations

import os
import re
from pathlib import Path

_PROC_FD = re.compile(r"/proc/self/fd/([0-9]+)")


def pass_fds_for_path(path: str | os.PathLike[str]) -> tuple[int, ...]:
    """Return the live descriptor referenced by a private asset alias."""
    candidate = Path(path)
    try:
        target = os.readlink(candidate) if candidate.is_symlink() else str(candidate)
    except OSError:
        return ()
    match = _PROC_FD.fullmatch(target)
    if match is None:
        return ()
    descriptor = int(match.group(1))
    os.fstat(descriptor)
    return (descriptor,)
