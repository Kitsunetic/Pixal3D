"""Strict readers for runner-owned instance lists and family manifests."""

from __future__ import annotations

from collections.abc import Collection
import errno
import json
import os
from pathlib import Path
import stat


def _open_directory_nofollow(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def read_regular_bytes_nofollow(path: Path) -> bytes:
    path = Path(path)
    directory_fd = _open_directory_nofollow(path.parent)
    try:
        file_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        try:
            opened = os.fstat(file_fd)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError(errno.ELOOP, f"not a regular file: {path}")
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                value = stream.read()
            current = os.stat(
                path.name, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)
            ):
                raise OSError(
                    errno.ELOOP,
                    f"file identity changed while reading: {path}",
                )
            return value
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def read_asset_ids(path: Path) -> tuple[str, ...]:
    try:
        lines = read_regular_bytes_nofollow(path).decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"instance list must be ASCII: {path}") from error
    assets = tuple(lines)
    if any(
        len(asset) != 64
        or any(character not in "0123456789abcdef" for character in asset)
        for asset in assets
    ):
        raise ValueError(f"instance list contains an invalid SHA-256: {path}")
    if assets != tuple(sorted(set(assets))):
        raise ValueError(f"instance list must be sorted and unique: {path}")
    return assets


def read_family_instance_paths(
    path: Path | None,
    expected_families: Collection[str],
) -> dict[str, Path]:
    if path is None:
        return {}
    manifest = Path(os.path.abspath(path))
    try:
        value: object = json.loads(
            read_regular_bytes_nofollow(manifest).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid family instance manifest: {manifest}") from error
    if not isinstance(value, dict) or set(value) != set(expected_families):
        raise ValueError("family instance manifest has unexpected family keys")
    result: dict[str, Path] = {}
    for family, raw_instance_path in value.items():
        if not isinstance(family, str) or not isinstance(raw_instance_path, str):
            raise ValueError(
                "family instance manifest must map families to paths"
            )
        instance_path = Path(os.path.abspath(raw_instance_path))
        if (
            not Path(raw_instance_path).is_absolute()
            or instance_path.parent != manifest.parent
        ):
            raise ValueError(
                "family instance files must be in the manifest directory"
            )
        read_asset_ids(instance_path)
        result[family] = instance_path
    return result
