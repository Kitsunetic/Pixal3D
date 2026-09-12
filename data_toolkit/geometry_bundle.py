"""Run independent geometry resolutions concurrently within one CPU budget.

Run with the project environment, for example:
`python data_toolkit/geometry_bundle.py ABO --help`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from data_toolkit.pipeline.instance_manifest import (
    read_asset_ids,
    read_family_instance_paths,
)
from data_toolkit.pipeline.subprocess_control import run_bounded

GEOMETRY_SCRIPTS: Final = (
    ("dual_grid_view.py", "--mesh_dump_root", "--dual_grid_root"),
    ("voxelize_pbr_view.py", "--pbr_dump_root", "--pbr_voxel_root"),
)


def _parse_resolutions(value: str) -> tuple[int, ...]:
    resolutions = tuple(int(item) for item in value.split(","))
    if not resolutions or any(resolution <= 0 for resolution in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive integers")
    return resolutions


def parse_views(value: str) -> tuple[int, ...]:
    views: set[int] = set()
    for item in value.split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            views.update(range(start, end + 1))
        else:
            views.add(int(item))
    if not views or min(views) < 0:
        raise argparse.ArgumentTypeError("views must be non-negative integers")
    return tuple(sorted(views))


def _leaf_command(script: str, arguments: Sequence[str]) -> list[str]:
    override = os.environ.get("PIXAL3D_LEAF_WORKER")
    if override is not None:
        return [
            sys.executable,
            override,
            "--original-script",
            script,
            *arguments,
        ]
    return [sys.executable, str(Path(__file__).parent / script), *arguments]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--source")
    parser.add_argument("--root", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--family_instances_file")
    parser.add_argument("--mesh_dump_root", required=True)
    parser.add_argument("--pbr_dump_root", required=True)
    parser.add_argument("--transform_root", required=True)
    parser.add_argument("--voxel_root", required=True)
    parser.add_argument("--resolutions", type=_parse_resolutions, required=True)
    parser.add_argument("--ss_resolution", type=int)
    parser.add_argument("--view_indices", default="0-1")
    parser.add_argument("--max_workers", type=int, required=True)
    parser.add_argument("--native_threads", type=int, required=True)
    parser.add_argument("--record_prefix", default="")
    arguments = parser.parse_args(argv)
    if arguments.max_workers <= 0 or arguments.native_threads <= 0:
        parser.error("worker and native-thread counts must be positive")
    if arguments.max_workers > 44:
        parser.error("max_workers exceeds the 44-core affinity topology")

    views = parse_views(arguments.view_indices)
    default_instances = Path(arguments.instances)
    read_asset_ids(default_instances)
    expected_families = {
        f"{family}-{resolution}"
        for family in ("shape", "PBR")
        for resolution in arguments.resolutions
    }
    if arguments.family_instances_file is not None:
        if arguments.ss_resolution is None or arguments.ss_resolution <= 0:
            parser.error("family manifest requires a positive --ss_resolution")
        expected_families.add(f"SS-{arguments.ss_resolution}")
    family_instances = read_family_instance_paths(
        Path(arguments.family_instances_file)
        if arguments.family_instances_file is not None
        else None,
        expected_families,
    )
    dataset = [arguments.dataset]
    if arguments.source is not None:
        dataset.extend(("--source", arguments.source))

    def commands() -> tuple[tuple[list[str], dict[str, str]], ...]:
        result: list[tuple[list[str], dict[str, str]]] = []
        scheduled: dict[
            tuple[int, str, str, str, tuple[str, ...]],
            tuple[Path, list[int]],
        ] = {}
        for resolution in arguments.resolutions:
            for view in views:
                for script, input_flag, output_flag in GEOMETRY_SCRIPTS:
                    family = (
                        f"shape-{resolution}"
                        if script == "dual_grid_view.py"
                        else f"PBR-{resolution}"
                    )
                    instances = family_instances.get(family, default_instances)
                    assets = read_asset_ids(instances)
                    if not assets:
                        continue
                    key = (view, script, input_flag, output_flag, assets)
                    if key not in scheduled:
                        scheduled[key] = (instances, [])
                    scheduled[key][1].append(resolution)
        if not scheduled:
            return ()
        parallel_jobs = min(arguments.max_workers, len(scheduled))
        base_workers, extra_workers = divmod(
            arguments.max_workers, parallel_jobs
        )
        workers_by_slot = tuple(
            max(1, base_workers + (index < extra_workers))
            for index in range(parallel_jobs)
        )
        assigned_workers = sum(workers_by_slot)
        native_threads = min(
            arguments.native_threads,
            max(1, 44 // assigned_workers),
        )
        affinity_offsets: list[int] = []
        affinity_offset = 0
        for workers in workers_by_slot:
            affinity_offsets.append(affinity_offset)
            affinity_offset += workers
        for (
            (view, script, input_flag, output_flag, _assets),
            (instances, resolutions),
        ) in scheduled.items():
            slot = len(result) % parallel_jobs
            workers_per_job = workers_by_slot[slot]
            input_root = (
                arguments.mesh_dump_root
                if input_flag == "--mesh_dump_root"
                else arguments.pbr_dump_root
            )
            environment = os.environ.copy()
            environment["PIXAL3D_GEOMETRY_AFFINITY_OFFSET"] = str(
                affinity_offsets[slot]
            )
            result.append(
                (
                    _leaf_command(
                        script,
                        [
                            *dataset,
                            "--root",
                            arguments.root,
                            "--instances",
                            str(instances),
                            input_flag,
                            input_root,
                            "--transform_root",
                            arguments.transform_root,
                            output_flag,
                            arguments.voxel_root,
                            "--resolution",
                            ",".join(
                                str(resolution) for resolution in resolutions
                            ),
                            "--view_indices",
                            str(view),
                            "--max_workers",
                            str(workers_per_job),
                            "--native_threads",
                            str(native_threads),
                            "--record_prefix",
                            f"{arguments.record_prefix}view{view:02d}_",
                        ],
                    ),
                    environment,
                )
            )
        return tuple(result)

    launch = commands()
    if not launch:
        return 0

    parallel_jobs = min(len(launch), arguments.max_workers)
    for start in range(0, len(launch), parallel_jobs):
        wave = launch[start : start + parallel_jobs]
        run_bounded(wave, len(wave))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
