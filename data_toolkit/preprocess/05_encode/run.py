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
            skipped_published_prepared_v2=skip_reason == "prepared_v2",
        )
        print(output / "manifest.jsonl")
        return 0
    cuda_visible_devices = require_single_visible_cuda_device()
    manifest = arguments.manifest or stage_root(work_root, "04", "voxelize") / "manifest.jsonl"
    records = successful(read_jsonl(manifest))
    if not records:
        parser.error("04_voxelize의 성공 asset이 없습니다")
    instances = output / "instances.txt"
    instances.parent.mkdir(parents=True, exist_ok=True)
    instances.write_text("\n".join(record["asset_id"] for record in records) + "\n")
    root = Path(__file__).resolve().parents[3]
    voxel = stage_root(work_root, "04", "voxelize")
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
    subprocess.run(command, cwd=root, env=environment, check=True)
    result_records = [dict(record, status="ok", latent_root=str(output)) for record in records]
    atomic_write_jsonl(output / "manifest.jsonl", result_records)
    write_stage_info(output, stage="05_encode", config=str(config_path), cuda_visible_devices=cuda_visible_devices, total=len(records), resolutions=resolutions, ss_resolution=ss_resolution, batch_index=batch_index, world_size=arguments.world_size, rank=arguments.rank)
    print(output / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
