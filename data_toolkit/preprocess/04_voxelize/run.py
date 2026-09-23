#!/usr/bin/env python3
"""04_voxelize: 한 batch를 순차 실행하는 legacy voxel leaf adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_batch_location_arguments,
    add_config_argument,
    add_rank_arguments,
    apply_batch_defaults,
    atomic_write_jsonl,
    config_get,
    load_config,
    completed_batch_reason,
    read_jsonl,
    resolve_work_root,
    require_batch_ownership,
    stage_root,
    successful,
    write_legacy_metadata,
    write_stage_info,
)


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for item in value.split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(item))
    return sorted(set(indices))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    add_rank_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resolutions", default=None)
    parser.add_argument("--view-indices", default=None)
    parser.add_argument("--native-threads", type=int, default=None)
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    batch_index = require_batch_ownership(arguments, config)
    resolutions_arg = arguments.resolutions or str(config_get(config, "stages", "voxelize", "resolutions"))
    view_indices_arg = arguments.view_indices or str(config_get(config, "stages", "voxelize", "view_indices"))
    native_threads = arguments.native_threads if arguments.native_threads is not None else int(config_get(config, "stages", "voxelize", "native_threads"))
    # This adapter calls the legacy single-asset functions directly, bypassing
    # their legacy worker wrapper where the CPU thread cap was previously set.
    # Keep every rank bounded before importing numpy, the native voxel backend,
    # and torch.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(native_threads)
    from data_toolkit.pipeline.parallelism import configure_geometry_threads
    from data_toolkit.preprocess._common.cpu_ovoxel import import_cpu_ovoxel
    configure_geometry_threads(native_threads)
    import_cpu_ovoxel()

    work_root = resolve_work_root(arguments, config)
    output = stage_root(work_root, "04", "voxelize")
    skip_reason = completed_batch_reason(config, arguments.source, arguments.shard, arguments.batch)
    if skip_reason is not None:
        atomic_write_jsonl(output / "manifest.jsonl", [])
        write_stage_info(
            output, stage="04_voxelize", config=str(config_path), total=0,
            batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank,
            skipped_completed=True, skip_reason=skip_reason,
            skipped_existing_prepared=skip_reason == "legacy_prepared",
        )
        print(output / "manifest.jsonl")
        return 0
    manifest = arguments.manifest or stage_root(work_root, "03", "render") / "manifest.jsonl"
    records = successful(read_jsonl(manifest))
    if not records:
        parser.error("03_render의 성공 asset이 없습니다")
    instances = output / "instances.txt"
    instances.parent.mkdir(parents=True, exist_ok=True)
    instances.write_text("\n".join(record["asset_id"] for record in records) + "\n")
    write_legacy_metadata(output / "metadata.csv", records, "rendered")
    # The legacy leaf functions themselves contain the production-compatible voxel
    # math. Calling them directly avoids their scheduler-like bounded worker loop:
    # this run.py owns one batch and processes its assets sequentially.
    from data_toolkit.dual_grid_view import _dual_grid_mesh_view
    from data_toolkit.voxelize_pbr_view import _pbr_voxelize_view

    resolutions = [int(value) for value in resolutions_arg.split(",")]
    view_indices = parse_indices(view_indices_arg)
    dump_root = stage_root(work_root, "02", "dump")
    render_root = stage_root(work_root, "03", "render") / "renders_cond"
    result_records = []
    for record in records:
        result = dict(record)
        asset_id = record["asset_id"]
        dual = _dual_grid_mesh_view(
            None, asset_id, str(dump_root), str(render_root), str(output),
            resolutions, native_threads, view_indices,
        )
        pbr = _pbr_voxelize_view(
            None, asset_id, str(dump_root), str(render_root), str(output),
            resolutions, native_threads, view_indices,
        )
        errors = [value.get("error") for value in (dual, pbr) if value.get("error")]
        if errors:
            result.update(status="error", error="; ".join(errors))
        else:
            result.update(status="ok", voxel_root=str(output))
        result_records.append(result)
    atomic_write_jsonl(output / "manifest.jsonl", result_records)
    write_stage_info(
        output, stage="04_voxelize", total=len(records),
        batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank,
        config=str(config_path), succeeded=len(successful(result_records)), resolutions=resolutions_arg,
        view_indices=view_indices_arg,
    )
    print(output / "manifest.jsonl")
    return 0 if len(successful(result_records)) == len(result_records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
