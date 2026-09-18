#!/usr/bin/env python3
"""03_render: dump 산출물을 읽지 않고 원본 GLB만 embedded bpy로 렌더링한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_batch_location_arguments,
    add_config_argument,
    apply_batch_defaults,
    atomic_write_jsonl,
    config_get,
    load_config,
    materialize_glb,
    read_jsonl,
    resolve_work_root,
    require_single_visible_cuda_device,
    stage_root,
    successful,
    write_stage_info,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    resolution = arguments.resolution if arguments.resolution is not None else int(config_get(config, "stages", "render", "resolution"))
    seed = arguments.seed if arguments.seed is not None else config_get(config, "stages", "render", "seed")
    cuda_visible_devices = require_single_visible_cuda_device()

    from data_toolkit.blender_script import render_cond
    from data_toolkit.pipeline.camera import build_condition_views
    from data_toolkit.pipeline.config import RenderConfig

    work_root = resolve_work_root(arguments, config)
    scratch_root = Path(config_get(config, "paths", "scratch_root"))
    archive_binary = str(config_get(config, "raw", "archive_binary"))
    manifest = arguments.manifest or stage_root(work_root, "02", "dump") / "manifest.jsonl"
    output = stage_root(work_root, "03", "render")
    render_root = output / "renders_cond"
    camera_config = RenderConfig(
        num_views=8,
        resolution=resolution,
        fov_min_degrees=10.0,
        fov_max_degrees=70.0,
        camera_policy="pixal3d-mv-camera-v1",
        blender_version="4.5.1",
        cycles_device="OPTIX",
    )
    results = []
    for record in read_jsonl(manifest):
        result = dict(record)
        asset_id = record["asset_id"]
        final_dir = render_root / asset_id
        try:
            if record.get("status") != "ok":
                result.update(status="skipped", error="02_dump failed")
            elif (final_dir / "transforms.json").is_file():
                result.update(status="ok", render_dir=str(final_dir), skipped=True)
            else:
                temporary = render_root / f".{asset_id}.tmp"
                shutil.rmtree(temporary, ignore_errors=True)
                temporary.mkdir(parents=True, exist_ok=False)
                with materialize_glb(record, scratch_root, archive_binary) as object_path:
                    render_cond.main(
                        SimpleNamespace(
                        object=str(object_path),
                        cond_views=json.dumps(build_condition_views(asset_id, camera_config)),
                        cond_output_folder=str(temporary), cond_resolution=resolution,
                        boundary_fit_resolution=min(128, resolution),
                        boundary_fit_engine="CYCLES", boundary_fit_samples=1,
                        seed=seed, engine="CYCLES", cycles_device="OPTIX",
                        )
                    )
                if not (temporary / "transforms.json").is_file():
                    raise RuntimeError("renderer did not create transforms.json")
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, final_dir)
                result.update(status="ok", render_dir=str(final_dir), skipped=False)
        except Exception as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        results.append(result)
    atomic_write_jsonl(output / "manifest.jsonl", results)
    ok = successful(results)
    errors = [record for record in results if record.get("status") == "error"]
    write_stage_info(
        output, stage="03_render", config=str(config_path), cuda_visible_devices=cuda_visible_devices, total=len(results),
        succeeded=len(ok), skipped=len(results) - len(ok) - len(errors), errors=len(errors), views=8,
    )
    print(output / "manifest.jsonl")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
