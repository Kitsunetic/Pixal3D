"""Run independent geometry resolutions concurrently within one CPU budget.

Run with the project environment, for example:
`python data_toolkit/geometry_bundle.py ABO --help`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys
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


def _parse_views(value: str) -> tuple[int, ...]:
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

    views = _parse_views(arguments.view_indices)
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

    def commands() -> tuple[list[str], ...]:
        result: list[list[str]] = []
        scheduled: dict[
            tuple[int, str, str, str, Path], list[int]
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
                    if not read_asset_ids(instances):
                        continue
                    key = (view, script, input_flag, output_flag, instances)
                    scheduled.setdefault(key, []).append(resolution)
        if not scheduled:
            return ()
        workers_per_job = max(
            1, arguments.max_workers // len(scheduled)
        )
        for (view, script, input_flag, output_flag, instances), resolutions in (
            scheduled.items()
        ):
            input_root = (
                arguments.mesh_dump_root
                if input_flag == "--mesh_dump_root"
                else arguments.pbr_dump_root
            )
            result.append(
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
                        ",".join(str(resolution) for resolution in resolutions),
                        "--view_indices",
                        str(view),
                        "--max_workers",
                        str(workers_per_job),
                        "--native_threads",
                        str(arguments.native_threads),
                        "--record_prefix",
                        f"{arguments.record_prefix}view{view:02d}_",
                    ],
                )
            )
        return tuple(result)

    launch = commands()
    if not launch:
        return 0

    run_bounded(
        tuple((command, None) for command in launch),
        min(len(launch), arguments.max_workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
