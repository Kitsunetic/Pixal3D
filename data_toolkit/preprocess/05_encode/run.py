#!/usr/bin/env python3
"""05_encode: shape, SS, PBR latent을 한 CUDA 프로세스에서 순차 인코딩한다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
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
    require_single_visible_cuda_device,
    completed_batch_reason,
    read_jsonl,
    resolve_work_root,
    require_batch_ownership,
    stage_root,
    successful,
    write_stage_info,
)
from data_toolkit.preprocess._common.encode_partition import (
    exclude_configured_assets,
    expected_latent_paths,
    parse_view_indices,
    select_records_with_shape_1024_voxels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    add_rank_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resolutions", default=None)
    parser.add_argument("--ss-resolution", type=int, default=None)
    parser.add_argument("--view-indices", default=None)
    parser.add_argument("--loader-workers", type=int, default=None)
    parser.add_argument("--saver-workers", type=int, default=None)
    parser.add_argument("--micro-batch-sizes", default=None)
    parser.add_argument("--latent-dtype", choices=("float16", "float32"), default=None)
    parser.add_argument("--gpu-memory-target-percent", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--min-shape-1024-voxels", type=int, default=None)
    parser.add_argument("--max-shape-1024-voxels", type=int, default=None)
    parser.add_argument(
        "--partition-name",
        default=None,
        help="voxel-range asset partition의 이름; partition 실행에서 필수",
    )
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    batch_index = require_batch_ownership(arguments, config)
    encode = config_get(config, "stages", "encode")
    resolutions = arguments.resolutions or str(encode["resolutions"])
    ss_resolution = arguments.ss_resolution if arguments.ss_resolution is not None else int(encode["ss_resolution"])
    view_indices = arguments.view_indices or str(encode["view_indices"])
    loader_workers = arguments.loader_workers if arguments.loader_workers is not None else int(encode["loader_workers"])
    saver_workers = arguments.saver_workers if arguments.saver_workers is not None else int(encode["saver_workers"])
    micro_batch_sizes = arguments.micro_batch_sizes or str(encode["micro_batch_sizes"])
    latent_dtype = arguments.latent_dtype or str(encode["latent_dtype"])
    gpu_memory_target_percent = arguments.gpu_memory_target_percent if arguments.gpu_memory_target_percent is not None else float(encode["gpu_memory_target_percent"])
    timeout_seconds = arguments.timeout_seconds if arguments.timeout_seconds is not None else int(encode["timeout_seconds"])
    work_root = resolve_work_root(arguments, config)
    output = stage_root(work_root, "05", "encode")
    skip_reason = completed_batch_reason(config, arguments.source, arguments.shard, arguments.batch)
    if skip_reason is not None:
        atomic_write_jsonl(output / "manifest.jsonl", [])
        write_stage_info(
            output, stage="05_encode", config=str(config_path), total=0,
            batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank,
            skipped_completed=True, skip_reason=skip_reason,
            skipped_existing_prepared=skip_reason == "legacy_prepared",
        )
        print(output / "manifest.jsonl")
        return 0
    cuda_visible_devices = require_single_visible_cuda_device()
    manifest = arguments.manifest or stage_root(work_root, "04", "voxelize") / "manifest.jsonl"
    records = exclude_configured_assets(successful(read_jsonl(manifest)), config)
    partitioned = (
        arguments.min_shape_1024_voxels is not None
        or arguments.max_shape_1024_voxels is not None
    )
    if partitioned:
        if not arguments.partition_name:
            parser.error("voxel-range 실행에는 --partition-name이 필요합니다")
        if "/" in arguments.partition_name or arguments.partition_name in {".", ".."}:
            parser.error("--partition-name은 단일 경로 이름이어야 합니다")
        if 1024 not in {int(value) for value in resolutions.split(",")}:
            parser.error("voxel-range 실행에는 1024 shape resolution이 필요합니다")
    voxel = stage_root(work_root, "04", "voxelize")
    views = parse_view_indices(view_indices)
    max_new_voxels = encode.get("max_new_shape_1024_voxels")
    if max_new_voxels is not None and (type(max_new_voxels) is not int or max_new_voxels <= 0):
        parser.error("max_new_shape_1024_voxels는 양의 정수여야 합니다")
    selected = (
        select_records_with_shape_1024_voxels(
            records, voxel, view_indices=views,
            minimum=arguments.min_shape_1024_voxels,
            maximum=arguments.max_shape_1024_voxels,
        )
        if partitioned or max_new_voxels is not None
        else [(record, 0) for record in records]
    )
    resolutions_tuple = tuple(int(value) for value in resolutions.split(","))
    records_to_encode = []
    completed_oversized = []
    skipped_oversized = []
    for record, count in selected:
        if max_new_voxels is None or count <= max_new_voxels:
            records_to_encode.append(record)
        else:
            paths = expected_latent_paths(
                [record], output,
                resolutions=resolutions_tuple, ss_resolution=ss_resolution,
                view_indices=views,
            )
            if all(
                path.is_file()
                and path.with_name(f"{path.stem}_scale.json").is_file()
                for path in paths
            ):
                completed_oversized.append(record)
            else:
                skipped_oversized.append({
                    "asset_id": record["asset_id"], "status": "skipped_oversized",
                    "reason": "max_new_shape_1024_voxels",
                    "shape_1024_voxels": count,
                })
    if not selected:
        print("선택한 voxel-range에 04_voxelize 성공 asset이 없습니다")
        return 0
    partition_root = (
        output / "partitions" / arguments.partition_name
        if partitioned else output
    )
    instances = partition_root / "instances.txt"
    instances.parent.mkdir(parents=True, exist_ok=True)
    instances.write_text("\n".join(record["asset_id"] for record in records_to_encode) + "\n")
    root = Path(__file__).resolve().parents[3]
    command = [
        # Module execution preserves the repository root on sys.path.  Running
        # the file directly makes ``data_toolkit`` unavailable to the bundle.
        sys.executable, "-m", "data_toolkit.encode_latent_bundle",
        "--root", str(voxel), "--instances", str(instances),
        "--dual_grid_root", str(voxel), "--pbr_voxel_root", str(voxel),
        "--shape_latent_root", str(output / "shape"),
        "--pbr_latent_root", str(output / "pbr"),
        "--ss_latent_root", str(output / "ss"),
        "--resolutions", resolutions, "--ss_resolution", str(ss_resolution),
        "--view_indices", view_indices,
        "--loader_workers", str(loader_workers),
        "--saver_workers", str(saver_workers),
        "--latent_dtype", latent_dtype,
        "--micro_batch_sizes", micro_batch_sizes,
        "--gpu_memory_target_percent", str(gpu_memory_target_percent),
        "--timeout_seconds", str(timeout_seconds),
    ]
    environment = os.environ.copy()
    if records_to_encode:
        subprocess.run(command, cwd=root, env=environment, check=True)
    result_records = [
        dict(record, status="ok", latent_root=str(output))
        for record in (*records_to_encode, *completed_oversized)
    ]
    result_records.extend(skipped_oversized)
    atomic_write_jsonl(partition_root / "manifest.jsonl", result_records)
    write_stage_info(
        partition_root, stage="05_encode", config=str(config_path),
        cuda_visible_devices=cuda_visible_devices, total=len(records_to_encode) + len(completed_oversized),
        skipped_oversized=len(skipped_oversized),
        resolutions=resolutions, ss_resolution=ss_resolution,
        batch_index=batch_index, world_size=arguments.world_size,
        rank=arguments.rank, partition_name=arguments.partition_name,
        min_shape_1024_voxels=arguments.min_shape_1024_voxels,
        max_shape_1024_voxels=arguments.max_shape_1024_voxels,
    )
    print(partition_root / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
