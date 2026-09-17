"""Profiled CPU-side payload loading shared by production encoders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, TypeAlias

import numpy as np
import torch

from data_toolkit.pipeline.stage_profiling import profile_stage


ShapePayload: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
PbrPayload: TypeAlias = tuple[torch.Tensor, tuple[torch.Tensor, ...]]
_PBR_ATTRIBUTE_NAMES: Final = ("base_color", "metallic", "roughness", "alpha")


class VxzReader(Protocol):
    """The narrow production contract provided by ``o_voxel.io.read_vxz``."""

    def __call__(
        self,
        path: str,
        *,
        num_threads: int,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]: ...


def load_shape_vxz(path: Path, read_vxz: VxzReader) -> ShapePayload:
    """Read a Shape VXZ file and select its existing encoder payload."""
    with profile_stage("shape.loader.vxz.read_decode"):
        coordinates, attributes = read_vxz(str(path), num_threads=1)
    with profile_stage("shape.loader.payload.select"):
        return (
            coordinates,
            attributes["vertices"],
            attributes["intersected"],
        )


def load_pbr_vxz(path: Path, read_vxz: VxzReader) -> PbrPayload:
    """Read a PBR VXZ file and select its encoder attributes in legacy order."""
    with profile_stage("pbr.loader.vxz.read_decode"):
        coordinates, attributes = read_vxz(str(path), num_threads=1)
    with profile_stage("pbr.loader.payload.select"):
        return coordinates, tuple(
            attributes[name] for name in _PBR_ATTRIBUTE_NAMES
        )


def load_ss_coordinates(path: Path) -> torch.Tensor:
    """Materialize sparse-latent coordinates while preserving their source dtype."""
    with profile_stage("ss.loader.npz.open"):
        archive = np.load(path, allow_pickle=False)
    try:
        with profile_stage("ss.loader.npz.coords_materialize"):
            coordinates = np.asarray(archive["coords"])
    finally:
        archive.close()
    with profile_stage("ss.loader.numpy_to_torch"):
        return torch.from_numpy(coordinates)
