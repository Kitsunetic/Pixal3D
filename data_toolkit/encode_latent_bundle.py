"""Run production latent encoders while caching models across resolutions."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import gc
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Protocol

from data_toolkit.pipeline.instance_manifest import (
    read_asset_ids,
    read_family_instance_paths,
)


class Encoder(Protocol):
    def eval(self) -> "Encoder": ...
    def cuda(self) -> "Encoder": ...
    def requires_grad_(self, requires_grad: bool = True) -> "Encoder": ...


def _parse_resolutions(value: str) -> tuple[int, ...]:
    resolutions = tuple(int(item) for item in value.split(","))
    if not resolutions or any(resolution <= 0 for resolution in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive integers")
    return resolutions


def _parse_micro_batches(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    try:
        for item in value.split(","):
            resolution, size = (int(part) for part in item.split(":", 1))
            if resolution <= 0 or size <= 0:
                raise ValueError
            result[resolution] = size
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "micro batches must use RESOLUTION:SIZE pairs"
        ) from error
    return result


def _run_leaf(script: Path, arguments: Sequence[str]) -> None:
    previous = sys.argv
    previous_path = sys.path.copy()
    sys.argv = [str(script), *arguments]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous
        sys.path[:] = previous_path


def _run_fake_leaf(script: Path, arguments: Sequence[str], override: str) -> None:
    command = [
        sys.executable, override, "--original-script", script.name, *arguments
    ]
    subprocess.run(command, check=True)


def _wait_for_shape_ranks(
    barrier_root: Path,
    rank: int,
    world_size: int,
    timeout_seconds: int,
) -> None:
    marker = barrier_root / f"rank-{rank}.ready"
    marker.write_bytes(b"ready\n")
    deadline = time.monotonic() + timeout_seconds
    expected = tuple(
        barrier_root / f"rank-{other_rank}.ready"
        for other_rank in range(world_size)
    )
    while not all(path.is_file() for path in expected):
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for all shape encoder ranks")
        time.sleep(0.05)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--family_instances_file")
    parser.add_argument("--dual_grid_root", required=True)
    parser.add_argument("--pbr_voxel_root", required=True)
    parser.add_argument("--shape_latent_root", required=True)
    parser.add_argument("--pbr_latent_root", required=True)
    parser.add_argument("--ss_latent_root", required=True)
    parser.add_argument("--resolutions", type=_parse_resolutions, required=True)
    parser.add_argument("--ss_resolution", type=int, required=True)
    parser.add_argument("--view_indices", default="0-1")
    parser.add_argument("--loader_workers", type=int, required=True)
    parser.add_argument("--saver_workers", type=int, required=True)
    parser.add_argument("--latent_dtype", choices=("float32", "float16"), required=True)
    parser.add_argument("--micro_batch_sizes", type=_parse_micro_batches, required=True)
    parser.add_argument("--gpu_memory_target_percent", type=float, required=True)
    parser.add_argument("--timeout_seconds", type=int, default=900)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--barrier_root")
    arguments = parser.parse_args(argv)
    required_resolutions = {*arguments.resolutions, arguments.ss_resolution}
    if not required_resolutions.issubset(arguments.micro_batch_sizes):
        parser.error("micro batch size is missing for a configured resolution")
    if arguments.world_size > 1 and arguments.barrier_root is None:
        parser.error("multi-rank encoding requires --barrier_root")

    override = os.environ.get("PIXAL3D_LEAF_WORKER")
    default_instances = Path(arguments.instances)
    read_asset_ids(default_instances)
    expected_families = {
        *(f"shape-{resolution}" for resolution in arguments.resolutions),
        *(f"PBR-{resolution}" for resolution in arguments.resolutions),
        f"SS-{arguments.ss_resolution}",
    }
    family_instances = read_family_instance_paths(
        Path(arguments.family_instances_file)
        if arguments.family_instances_file is not None
        else None,
        expected_families,
    )
    common = [
        "--root",
        arguments.root,
        "--view_indices",
        arguments.view_indices,
        "--loader_workers",
        str(arguments.loader_workers),
        "--saver_workers",
        str(arguments.saver_workers),
        "--gpu_memory_target_percent",
        str(arguments.gpu_memory_target_percent),
        "--timeout_seconds",
        str(arguments.timeout_seconds),
        "--rank",
        str(arguments.rank),
        "--world_size",
        str(arguments.world_size),
    ]
    toolkit = Path(__file__).parent
    shape_invocations: list[tuple[Path, list[str]]] = []
    pbr_invocations: list[tuple[Path, list[str]]] = []

    def family_arguments(family: str) -> list[str] | None:
        instances = family_instances.get(family, default_instances)
        if not read_asset_ids(instances):
            return None
        return [*common, "--instances", str(instances)]

    for resolution in arguments.resolutions:
        leaf_common = family_arguments(f"shape-{resolution}")
        if leaf_common is None:
            continue
        shape_invocations.append(
            (
                toolkit / "encode_shape_latent_view.py",
                [
                    *leaf_common,
                    "--dual_grid_root",
                    arguments.dual_grid_root,
                    "--shape_latent_root",
                    arguments.shape_latent_root,
                    "--resolution",
                    str(resolution),
                    "--latent_dtype",
                    arguments.latent_dtype,
                    "--micro_batch_size",
                    str(arguments.micro_batch_sizes[resolution]),
                ],
            )
        )
    shape_resolution = max(arguments.resolutions)
    ss_common = family_arguments(f"SS-{arguments.ss_resolution}")
    ss_invocations = (
        [
            (
                toolkit / "encode_ss_latent_view.py",
                [
                    *ss_common,
                    "--shape_latent_root",
                    arguments.shape_latent_root,
                    "--ss_latent_root",
                    arguments.ss_latent_root,
                    "--shape_latent_name",
                    f"shape_enc_next_dc_f16c32_fp16_{shape_resolution}",
                    "--resolution",
                    str(arguments.ss_resolution),
                    "--micro_batch_size",
                    str(arguments.micro_batch_sizes[arguments.ss_resolution]),
                ],
            )
        ]
        if ss_common is not None
        else []
    )
    for resolution in arguments.resolutions:
        leaf_common = family_arguments(f"PBR-{resolution}")
        if leaf_common is None:
            continue
        pbr_invocations.append(
            (
                toolkit / "encode_pbr_latent_view.py",
                [
                    *leaf_common,
                    "--pbr_voxel_root",
                    arguments.pbr_voxel_root,
                    "--pbr_latent_root",
                    arguments.pbr_latent_root,
                    "--resolution",
                    str(resolution),
                    "--latent_dtype",
                    arguments.latent_dtype,
                    "--micro_batch_size",
                    str(arguments.micro_batch_sizes[resolution]),
                ],
            )
        )
    invocations = [*shape_invocations, *ss_invocations, *pbr_invocations]

    if override is not None:
        for script, leaf_arguments in invocations:
            _run_fake_leaf(script, leaf_arguments, override)
        return 0

    import pixal3d.models as models

    original_loader: Callable[[str], Encoder] = models.from_pretrained
    encoders: dict[str, Encoder] = {}

    def cached_loader(model_path: str) -> Encoder:
        encoder = encoders.get(model_path)
        if encoder is None:
            encoder = original_loader(model_path).requires_grad_(False)
            encoders[model_path] = encoder
        return encoder

    def clear_cached_encoder() -> None:
        encoders.clear()
        gc.collect()

    models.from_pretrained = cached_loader
    try:
        for script, leaf_arguments in shape_invocations:
            _run_leaf(script, leaf_arguments)
        if arguments.barrier_root is not None:
            _wait_for_shape_ranks(
                Path(arguments.barrier_root),
                arguments.rank,
                arguments.world_size,
                arguments.timeout_seconds,
            )
        clear_cached_encoder()
        for script, leaf_arguments in ss_invocations:
            _run_leaf(script, leaf_arguments)
        if ss_invocations:
            clear_cached_encoder()
        for script, leaf_arguments in pbr_invocations:
            _run_leaf(script, leaf_arguments)
    finally:
        models.from_pretrained = original_loader
        clear_cached_encoder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
