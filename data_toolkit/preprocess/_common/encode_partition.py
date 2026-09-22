"""Select disjoint 05 encoder asset sets from cheap VXZ header metadata."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Iterable, Mapping


def parse_view_indices(value: str) -> tuple[int, ...]:
    """Parse the compact comma/range view syntax used by preprocess YAML."""
    parsed: list[int] = []
    for item in value.split(","):
        start_text, separator, end_text = item.strip().partition("-")
        if not start_text:
            raise ValueError("view index cannot be empty")
        start = int(start_text)
        if separator:
            if not end_text:
                raise ValueError("view range end cannot be empty")
            end = int(end_text)
            if end < start:
                raise ValueError("view range must be ascending")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(start)
    if not parsed or any(index < 0 for index in parsed):
        raise ValueError("view indices must be non-negative")
    return tuple(parsed)


def read_vxz_num_voxels(path: Path) -> int:
    """Read only a VXZ header and return its declared sparse voxel count."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8 or prefix[:3] != b"VXZ" or prefix[3] != 0:
            raise ValueError(f"invalid VXZ header: {path}")
        header_end = struct.unpack(">I", prefix[4:8])[0]
        if header_end < 8:
            raise ValueError(f"invalid VXZ header size: {path}")
        raw_header = stream.read(header_end - 8)
    if len(raw_header) != header_end - 8:
        raise ValueError(f"truncated VXZ header: {path}")
    try:
        value = json.loads(raw_header.decode("utf-8"))
        count = value["num_voxel"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"invalid VXZ metadata: {path}") from error
    if type(count) is not int or count < 0:
        raise ValueError(f"invalid VXZ num_voxel: {path}")
    return count


def max_shape_1024_voxels(
    asset_id: str,
    voxel_root: Path,
    *,
    view_indices: tuple[int, ...],
) -> int:
    """Return the largest 1024-shape VXZ count across required views."""
    if not view_indices:
        raise ValueError("view indices must not be empty")
    values = [
        read_vxz_num_voxels(
            voxel_root
            / "dual_grid_view_1024"
            / asset_id
            / f"view{view_index:02d}.vxz"
        )
        for view_index in view_indices
    ]
    return max(values)


def select_records_by_shape_1024_voxels(
    records: Iterable[Mapping[str, str]],
    voxel_root: Path,
    *,
    view_indices: tuple[int, ...],
    minimum: int | None = None,
    maximum: int | None = None,
) -> list[Mapping[str, str]]:
    """Keep records whose largest 1024 shape view is in the inclusive range."""
    if minimum is not None and (type(minimum) is not int or minimum < 0):
        raise ValueError("minimum voxel count must be a non-negative integer")
    if maximum is not None and (type(maximum) is not int or maximum < 0):
        raise ValueError("maximum voxel count must be a non-negative integer")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum voxel count must not exceed maximum voxel count")
    selected: list[Mapping[str, str]] = []
    for record in records:
        asset_id = record.get("asset_id")
        if asset_id is None:
            raise ValueError("05 encode record has no asset_id")
        count = max_shape_1024_voxels(
            asset_id, voxel_root, view_indices=view_indices,
        )
        if (minimum is None or count >= minimum) and (
            maximum is None or count <= maximum
        ):
            selected.append(record)
    return selected


def expected_latent_paths(
    records: Iterable[Mapping[str, str]],
    encode_root: Path,
    *,
    resolutions: tuple[int, ...],
    ss_resolution: int,
    view_indices: tuple[int, ...],
) -> tuple[Path, ...]:
    """Return the latent files required to finalize all successful 04 assets."""
    paths: list[Path] = []
    for record in records:
        asset_id = record.get("asset_id")
        if asset_id is None:
            raise ValueError("06 finalize record has no asset_id")
        for view_index in view_indices:
            view_name = f"view{view_index:02d}.npz"
            for resolution in resolutions:
                paths.append(
                    encode_root
                    / "shape"
                    / "shape_latents"
                    / f"shape_enc_next_dc_f16c32_fp16_{resolution}_view"
                    / asset_id
                    / view_name
                )
                paths.append(
                    encode_root
                    / "pbr"
                    / "pbr_latents"
                    / f"tex_enc_next_dc_f16c32_fp16_{resolution}_view_fix"
                    / asset_id
                    / view_name
                )
            paths.append(
                encode_root
                / "ss"
                / "ss_latents"
                / f"ss_enc_conv3d_16l8_fp16_{ss_resolution}_view"
                / asset_id
                / view_name
            )
    return tuple(paths)
