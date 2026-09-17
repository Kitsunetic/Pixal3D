"""Overlap independent dump and render leaves without changing their outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Sequence

from data_toolkit.pipeline.instance_manifest import read_asset_ids
from data_toolkit.pipeline.subprocess_control import (
    terminate_and_reap,
    wait_all,
    wait_until,
)


def _leaf(script: str, arguments: Sequence[str]) -> list[str]:
    override = os.environ.get("PIXAL3D_LEAF_WORKER")
    if override is not None:
        return [sys.executable, override, "--original-script", script, *arguments]
    return [sys.executable, str(Path(__file__).parent / script), *arguments]


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


def _worker_budget(
    cpu_budget: int, requested_render_workers: int
) -> tuple[int, int, int]:
    if cpu_budget < 3 or requested_render_workers <= 0:
        raise ValueError("prepare bundle requires two dump slots and one render slot")
    render_workers = min(requested_render_workers, cpu_budget - 2)
    dump_workers = max(1, (cpu_budget - render_workers) // 2)
    stats_workers = max(1, cpu_budget - render_workers)
    return render_workers, dump_workers, stats_workers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--source")
    parser.add_argument("--root", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--download_root", required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_cond_views", type=int, required=True)
    parser.add_argument("--cond_resolution", type=int, required=True)
    parser.add_argument("--boundary_fit_resolution", type=int, required=True)
    parser.add_argument("--boundary_fit_engine", required=True)
    parser.add_argument("--boundary_fit_samples", type=int, required=True)
    parser.add_argument(
        "--renderer_mode", choices=("external", "native"), default="external"
    )
    parser.add_argument("--native_worker_max_assets", type=int, default=8)
    parser.add_argument("--blender_path", required=True)
    parser.add_argument("--cycles_device", required=True)
    parser.add_argument("--dump_workers", type=int, required=True)
    parser.add_argument("--render_workers", type=int, required=True)
    parser.add_argument("--render_workers_per_gpu", type=int, required=True)
    parser.add_argument("--gpu_count", type=int, required=True)
    parser.add_argument("--record_prefix", default="")
    parser.add_argument(
        "--phase", choices=("all", "dump", "render"), default="all"
    )
    args = parser.parse_args(argv)
    if not read_asset_ids(Path(args.instances)):
        return 0
    if args.native_worker_max_assets <= 0:
        parser.error("native worker max assets must be positive")
    gpu_indices: tuple[int, ...] = ()
    total_render_workers = 0
    if args.phase in {"all", "render"}:
        gpu_indices = _gpu_indices(min(args.gpu_count, args.render_workers))
        if args.renderer_mode == "native" and (
            args.gpu_count != 1
            or len(gpu_indices) != 1
        ):
            parser.error("native renderer requires exactly one visible GPU")
        requested_render_workers = (
            len(gpu_indices) * args.render_workers_per_gpu
        )
        if min(requested_render_workers, args.render_workers) <= 0:
            parser.error("render worker and GPU counts must be positive")
        total_render_workers = requested_render_workers
    if args.phase == "all":
        try:
            total_render_workers, dump_workers, stats_workers = _worker_budget(
                args.dump_workers, total_render_workers
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.phase == "dump":
        if args.dump_workers < 2:
            parser.error("dump phase requires at least two CPU workers")
        dump_workers = max(1, args.dump_workers // 2)
        stats_workers = args.dump_workers
    else:
        dump_workers = 0
        stats_workers = 0

    instances = args.instances
    dataset = [args.dataset]
    if args.source is not None:
        dataset.extend(("--source", args.source))
    base = [*dataset, "--root", args.root, "--instances", instances]
    dump_common = [
        *base, "--download_root", args.download_root,
        "--blender_path", args.blender_path, "--max_workers", str(dump_workers),
    ]
    mesh = _leaf("dump_mesh.py", [*dump_common, "--mesh_dump_root", args.work_root])
    pbr = _leaf("dump_pbr.py", [*dump_common, "--pbr_dump_root", args.work_root])
    record = ["--record_prefix", args.record_prefix] if args.record_prefix else []
    renders: list[tuple[list[str], dict[str, str]]] = []
    for rank in range(total_render_workers):
        command = _leaf(
            "render_cond.py",
            [
                *base, "--download_root", args.download_root,
                "--render_cond_root", args.output_root,
                "--num_cond_views", str(args.num_cond_views),
                "--cond_resolution", str(args.cond_resolution),
                "--boundary_fit_resolution", str(args.boundary_fit_resolution),
                "--boundary_fit_engine", args.boundary_fit_engine,
                "--boundary_fit_samples", str(args.boundary_fit_samples),
                "--renderer_mode", args.renderer_mode,
                "--native_worker_max_assets", str(args.native_worker_max_assets),
                "--blender_path", args.blender_path,
                "--cycles_device", args.cycles_device, "--max_workers", "1",
                *record, "--rank", str(rank),
                "--world_size", str(total_render_workers),
            ],
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(
            gpu_indices[rank % len(gpu_indices)]
        )
        renders.append((command, environment))

    processes: list[subprocess.Popen[bytes]] = []
    try:
        dump_processes: tuple[subprocess.Popen[bytes], ...] = ()
        if args.phase in {"all", "dump"}:
            mesh_process = subprocess.Popen(mesh)
            processes.append(mesh_process)
            pbr_process = subprocess.Popen(pbr)
            processes.append(pbr_process)
            dump_processes = (mesh_process, pbr_process)
        for command, environment in renders:
            processes.append(subprocess.Popen(command, env=environment))
        if dump_processes:
            wait_until(processes, dump_processes)
            stats = _leaf("asset_stats.py", [
                "--root", args.root, "--instances", instances,
                "--mesh_dump_root", args.work_root,
                "--pbr_dump_root", args.work_root,
                "--max_workers", str(stats_workers), *record,
            ])
            processes.append(subprocess.Popen(stats))
        wait_all(processes)
    except BaseException:
        terminate_and_reap(processes)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
