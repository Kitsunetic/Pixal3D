"""Pipeline concurrent geometry producers into cached production encoders."""

from __future__ import annotations

import argparse
from math import ceil
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

from data_toolkit.geometry_bundle import GEOMETRY_SCRIPTS, parse_views
from data_toolkit.pipeline.instance_manifest import (
    read_asset_ids,
    read_family_instance_paths,
)
from data_toolkit.pipeline.subprocess_control import terminate_and_reap, wait_all


def _gpu_indices(count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("GPU count must be positive")
    raw = os.environ.get("PIXAL3D_GPU_INDICES")
    if raw is None:
        return tuple(range(count))
    try:
        indices = tuple(int(value) for value in raw.split(","))
    except ValueError as error:
        raise ValueError(
            "PIXAL3D_GPU_INDICES must be comma-separated integers"
        ) from error
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(
            "PIXAL3D_GPU_INDICES must contain unique nonnegative GPU indices"
        )
    if len(indices) != count:
        raise ValueError(
            f"PIXAL3D_GPU_INDICES must contain exactly {count} GPU indices"
        )
    return indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--source")
    parser.add_argument("--family_instances_file")
    for name in (
        "root", "instances", "mesh_dump_root", "pbr_dump_root",
        "transform_root", "voxel_root", "shape_latent_root",
        "pbr_latent_root", "ss_latent_root", "resolutions",
        "view_indices", "latent_dtype", "micro_batch_sizes",
    ):
        parser.add_argument(f"--{name}", required=True)
    for name in (
        "ss_resolution", "max_workers", "native_threads", "loader_workers",
        "saver_workers", "encoder_ranks", "gpu_count", "timeout_seconds",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--gpu_memory_target_percent", type=float, required=True)
    parser.add_argument("--record_prefix", default="")
    args = parser.parse_args(argv)
    resolutions = tuple(int(value) for value in args.resolutions.split(","))
    expected_families = {
        *(f"shape-{resolution}" for resolution in resolutions),
        *(f"PBR-{resolution}" for resolution in resolutions),
        f"SS-{args.ss_resolution}",
    }
    family_instances = read_family_instance_paths(
        Path(args.family_instances_file)
        if args.family_instances_file is not None
        else None,
        expected_families,
    )
    default_instances = Path(args.instances)
    if not read_asset_ids(default_instances):
        return 0
    scheduled_instances: dict[tuple[str, Path], int] = {}
    for resolution in resolutions:
        for script, _input_flag, _output_flag in GEOMETRY_SCRIPTS:
            family = (
                f"shape-{resolution}"
                if script == "dual_grid_view.py"
                else f"PBR-{resolution}"
            )
            instances = family_instances.get(family, default_instances)
            asset_count = len(read_asset_ids(instances))
            if asset_count:
                scheduled_instances[(script, instances)] = asset_count
    producer_waves = 1
    if scheduled_instances:
        geometry_job_count = len(parse_views(args.view_indices)) * len(
            scheduled_instances
        )
        workers_per_job = max(1, args.max_workers // geometry_job_count)
        producer_waves = max(
            ceil(asset_count / workers_per_job)
            for asset_count in scheduled_instances.values()
        )
    encoder_wait_seconds = args.timeout_seconds * producer_waves
    gpu_indices = _gpu_indices(args.gpu_count)
    rank_count = min(args.encoder_ranks, len(gpu_indices))
    if rank_count <= 0:
        parser.error("encoder rank and GPU counts must be positive")
    dataset = [args.dataset]
    if args.source is not None:
        dataset.extend(("--source", args.source))
    geometry = [
        sys.executable, "-m", "data_toolkit.geometry_bundle", *dataset,
        "--root", args.root, "--instances", args.instances,
        "--mesh_dump_root", args.mesh_dump_root,
        "--pbr_dump_root", args.pbr_dump_root,
        "--transform_root", args.transform_root, "--voxel_root", args.voxel_root,
        "--resolutions", args.resolutions, "--view_indices", args.view_indices,
        "--ss_resolution", str(args.ss_resolution),
        "--max_workers", str(args.max_workers),
        "--native_threads", str(args.native_threads),
        "--record_prefix", args.record_prefix,
    ]
    if args.family_instances_file is not None:
        geometry.extend(("--family_instances_file", args.family_instances_file))
    encoder_base = [
        sys.executable, "-m", "data_toolkit.encode_latent_bundle",
        "--root", args.root, "--instances", args.instances,
        "--dual_grid_root", args.voxel_root,
        "--pbr_voxel_root", args.voxel_root,
        "--shape_latent_root", args.shape_latent_root,
        "--pbr_latent_root", args.pbr_latent_root,
        "--ss_latent_root", args.ss_latent_root,
        "--resolutions", args.resolutions,
        "--ss_resolution", str(args.ss_resolution),
        "--view_indices", args.view_indices,
        "--loader_workers", str(args.loader_workers),
        "--saver_workers", str(args.saver_workers),
        "--latent_dtype", args.latent_dtype,
        "--micro_batch_sizes", args.micro_batch_sizes,
        "--gpu_memory_target_percent", str(args.gpu_memory_target_percent),
        "--timeout_seconds", str(encoder_wait_seconds),
    ]
    if args.family_instances_file is not None:
        encoder_base.extend(
            ("--family_instances_file", args.family_instances_file)
        )
    barrier_root = Path(
        tempfile.mkdtemp(prefix=".encoder-barrier-", dir=args.voxel_root)
    )
    encoder_base.extend(("--barrier_root", str(barrier_root)))
    try:
        geometry_process = subprocess.Popen(geometry)
        encoders: list[subprocess.Popen[bytes]] = []
        try:
            for rank in range(rank_count):
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(
                    gpu_indices[rank % len(gpu_indices)]
                )
                environment["PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS"] = str(
                    encoder_wait_seconds
                )
                encoders.append(subprocess.Popen([
                    *encoder_base, "--rank", str(rank),
                    "--world_size", str(rank_count),
                ], env=environment))
        except BaseException:
            terminate_and_reap((geometry_process, *encoders))
            raise
        wait_all((geometry_process, *encoders))
    finally:
        shutil.rmtree(barrier_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
