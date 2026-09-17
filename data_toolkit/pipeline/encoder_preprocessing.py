from __future__ import annotations

from collections.abc import Sequence

import torch

import pixal3d.modules.sparse as sp
from data_toolkit.pipeline.stage_profiling import profile_stage


def coordinates_to_uint8_tensor(
    coords: torch.Tensor,
    grid_resolution: int,
) -> torch.Tensor:
    """Validate and losslessly narrow latent coordinates on their device."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"invalid coordinate shape: {tuple(coords.shape)}")
    if coords.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        raise ValueError(f"invalid coordinate dtype: {coords.dtype}")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError("coordinates must be finite")
    if coords.is_floating_point() and not torch.equal(coords, torch.trunc(coords)):
        raise ValueError("coordinates must be integral")
    if bool((coords < 0).any()):
        raise ValueError("coordinates must be non-negative")
    if bool((coords >= grid_resolution).any()):
        raise ValueError("coordinates outside grid resolution")
    if bool((coords > torch.iinfo(torch.uint8).max).any()):
        raise ValueError("coordinates exceed uint8 storage range")
    narrowed = coords.to(torch.uint8)
    if not torch.equal(narrowed.to(torch.float64), coords.to(torch.float64)):
        raise ValueError("coordinate narrowing to uint8 was not lossless")
    return narrowed


ShapePayload = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
PbrPayload = tuple[torch.Tensor, Sequence[torch.Tensor]]


def dense_ss_batch(
    coordinate_sets: Sequence[torch.Tensor],
    resolution: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Create float occupancy volumes directly on the encoder device."""
    target_device = torch.device(device)
    with profile_stage("ss.volume.allocate", target_device):
        batch = torch.zeros(
            len(coordinate_sets),
            1,
            resolution,
            resolution,
            resolution,
            dtype=torch.float32,
            device=target_device,
        )
    for batch_index, coordinates in enumerate(coordinate_sets):
        with profile_stage("ss.coordinates.transfer", target_device):
            device_coordinates = coordinates.to(
                device=target_device,
                dtype=torch.long,
                non_blocking=True,
            )
        with profile_stage("ss.volume.scatter", target_device):
            batch[
                batch_index,
                0,
                device_coordinates[:, 0],
                device_coordinates[:, 1],
                device_coordinates[:, 2],
            ] = 1.0
    return batch


def _tensors_to_device(
    tensors: Sequence[torch.Tensor],
    device: str | torch.device,
    stage_name: str,
) -> list[torch.Tensor]:
    values = list(tensors)
    if not values:
        raise ValueError("cannot transfer zero tensors")
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("raw payloads must be tensors")
    target_device = torch.device(device)
    use_non_blocking_transfer = target_device.type == "cuda"
    if use_non_blocking_transfer and all(
        value.device.type == "cpu" for value in values
    ):
        with profile_stage(f"{stage_name}.pin"):
            pinned = [value.pin_memory() for value in values]
        with profile_stage(f"{stage_name}.h2d", target_device):
            return [
                value.to(target_device, non_blocking=True) for value in pinned
            ]
    with profile_stage(
        f"{stage_name}.transfer",
        target_device if use_non_blocking_transfer else None,
    ):
        return [value.to(target_device, non_blocking=False) for value in values]


def _batch_sparse_coordinates(
    coordinates: Sequence[torch.Tensor],
) -> torch.Tensor:
    values = list(coordinates)
    if not values:
        raise ValueError("cannot batch zero coordinate tensors")
    return torch.cat(
        [
            torch.cat(
                [torch.full_like(coords[:, :1], batch_index), coords],
                dim=-1,
            )
            for batch_index, coords in enumerate(values)
        ],
        dim=0,
    )


def prepare_shape_sparse_batch(
    payloads: Sequence[ShapePayload],
    device: str | torch.device,
) -> tuple[sp.SparseTensor, sp.SparseTensor]:
    """Build normalized shape encoder inputs after transferring raw VXZ tensors."""
    target_device = torch.device(device)
    coords = _tensors_to_device(
        [payload[0] for payload in payloads], target_device, "shape.coordinates"
    )
    vertex_features = _tensors_to_device(
        [payload[1] for payload in payloads], target_device, "shape.vertices"
    )
    intersected_features = _tensors_to_device(
        [payload[2] for payload in payloads],
        target_device,
        "shape.intersections",
    )
    with profile_stage("shape.coordinates.batch", target_device):
        batch_coords = _batch_sparse_coordinates(coords)
    with profile_stage("shape.features.concatenate", target_device):
        vertex_batch = torch.cat(vertex_features, dim=0)
        intersection_batch = torch.cat(intersected_features, dim=0)
    with profile_stage("shape.sparse.construct", target_device):
        vertices = sp.SparseTensor(vertex_batch, batch_coords)
        intersected = vertices.replace(intersection_batch)
    with profile_stage("shape.vertices.normalize", target_device):
        vertices = vertices.replace(vertices.feats.float().div(255.0))
    with profile_stage("shape.intersections.unpack", target_device):
        intersected = intersected.replace(
            torch.cat(
                [
                    intersected.feats % 2,
                    intersected.feats // 2 % 2,
                    intersected.feats // 4 % 2,
                ],
                dim=-1,
            ).bool()
        )
    return vertices, intersected


def prepare_pbr_sparse_batch(
    payloads: Sequence[PbrPayload],
    device: str | torch.device,
) -> sp.SparseTensor:
    """Build normalized PBR encoder input after transferring raw VXZ tensors."""
    target_device = torch.device(device)
    coords = _tensors_to_device(
        [payload[0] for payload in payloads], target_device, "pbr.coordinates"
    )
    attributes = [
        _tensors_to_device(
            [payload[1][attribute_index] for payload in payloads],
            target_device,
            f"pbr.attribute_{attribute_index}",
        )
        for attribute_index in range(len(payloads[0][1]))
    ]
    with profile_stage("pbr.features.concatenate", target_device):
        features = torch.cat(
            [torch.cat(attribute, dim=0) for attribute in attributes],
            dim=-1,
        )
    with profile_stage("pbr.coordinates.batch", target_device):
        batch_coords = _batch_sparse_coordinates(coords)
    with profile_stage("pbr.sparse.construct", target_device):
        voxels = sp.SparseTensor(features, batch_coords)
    with profile_stage("pbr.features.normalize", target_device):
        return voxels.replace(
            voxels.feats.float().div(255.0).mul(2.0).sub(1.0)
        )
