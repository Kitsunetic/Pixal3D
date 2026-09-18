#!/usr/bin/env python3
"""02_dump: 한 번의 embedded bpy import로 mesh와 PBR dump를 함께 생성한다."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_toolkit.preprocess._common.runtime import (
    add_batch_location_arguments,
    add_config_argument,
    apply_batch_defaults,
    atomic_pickle_dump,
    atomic_write_jsonl,
    config_get,
    load_config,
    materialize_glb,
    read_jsonl,
    resolve_work_root,
    require_single_visible_cuda_device,
    stage_root,
    successful,
    write_legacy_metadata,
    write_stage_info,
)


def _gpu_extract_image(tex_node, channels):
    """Blender pixels를 가져온 뒤의 clamp/quantize만 CUDA에서 수행한다.

    ``foreach_get``는 bpy가 제공하는 CPU 경계이며, 결과 PNG schema는 legacy
    ``dump_pbr.extract_image``와 동일하다.
    """
    import numpy as np
    import torch
    from PIL import Image

    image = tex_node.image
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    data = torch.from_numpy(values).view(image.size[1], image.size[0], -1)
    data = data[..., channels]
    if data.dtype != torch.uint8:
        data = data.to("cuda:0", non_blocking=False).clamp_(0.0, 1.0).mul_(255).to(torch.uint8)
        data = data.cpu().numpy()
    else:
        data = data.numpy()
    if data.ndim == 2:
        pil_image = Image.fromarray(data, mode="L")
    elif data.shape[2] == 3:
        pil_image = Image.fromarray(data, mode="RGB")
    elif data.shape[2] == 4:
        pil_image = Image.fromarray(data, mode="RGBA")
    else:
        raise ValueError(f"Unsupported channel shape for image: {data.shape}")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return {
        "image": buffer.getvalue(),
        "interpolation": tex_node.interpolation,
        "extension": tex_node.extension,
    }


def _mesh_from_pbr(pbr: dict) -> dict:
    return {
        "objects": [
            {"vertices": item["vertices"], "faces": item["faces"]}
            for item in pbr["objects"]
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    add_batch_location_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=None)
    arguments = parser.parse_args()
    config, config_path = load_config(arguments.config)
    apply_batch_defaults(arguments, config)
    cuda_visible_devices = require_single_visible_cuda_device()

    # CUDA visibility must be fixed before torch is imported by the PBR path.
    import pickle
    from types import SimpleNamespace
    from data_toolkit.blender_script import dump_pbr

    work_root = resolve_work_root(arguments, config)
    scratch_root = Path(config_get(config, "paths", "scratch_root"))
    archive_binary = str(config_get(config, "raw", "archive_binary"))
    manifest = arguments.manifest or stage_root(work_root, "01", "manifest") / "manifest.jsonl"
    output = stage_root(work_root, "02", "dump")
    mesh_root = output / "mesh_dumps"
    pbr_root = output / "pbr_dumps"
    dump_pbr.extract_image = _gpu_extract_image

    results = []
    for record in read_jsonl(manifest):
        asset_id = record["asset_id"]
        mesh_path = mesh_root / f"{asset_id}.pickle"
        pbr_path = pbr_root / f"{asset_id}.pickle"
        result = dict(record)
        if record.get("status") != "ok":
            result["status"] = "skipped"
            results.append(result)
            continue
        temporary = pbr_path.with_name(f".{asset_id}.pbr.tmp")
        try:
            if mesh_path.is_file() and pbr_path.is_file():
                result.update(status="ok", mesh_path=str(mesh_path), pbr_path=str(pbr_path), skipped=True)
            else:
                temporary.parent.mkdir(parents=True, exist_ok=True)
                with materialize_glb(record, scratch_root, archive_binary) as object_path:
                    dump_pbr.main(SimpleNamespace(object=str(object_path), output_path=str(temporary)))
                with temporary.open("rb") as stream:
                    pbr = pickle.load(stream)
                atomic_pickle_dump(pbr_path, pbr)
                atomic_pickle_dump(mesh_path, _mesh_from_pbr(pbr))
                result.update(status="ok", mesh_path=str(mesh_path), pbr_path=str(pbr_path), skipped=False)
        except Exception as error:
            result.update(status="error", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}_error.txt").unlink(missing_ok=True)
        results.append(result)

    atomic_write_jsonl(output / "manifest.jsonl", results)
    ok = successful(results)
    errors = [record for record in results if record.get("status") == "error"]
    write_legacy_metadata(mesh_root / "metadata.csv", ok, "mesh_dumped")
    write_legacy_metadata(pbr_root / "metadata.csv", ok, "pbr_dumped")
    write_stage_info(
        output, stage="02_dump", config=str(config_path), cuda_visible_devices=cuda_visible_devices, total=len(results),
        succeeded=len(ok), skipped=len(results) - len(ok) - len(errors), errors=len(errors),
    )
    print(output / "manifest.jsonl")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
