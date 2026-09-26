import json
import os
import struct
from pathlib import Path
import subprocess
import sys

import yaml

from data_toolkit.preprocess._common.encode_partition import (
    exclude_configured_assets,
    expected_latent_paths,
    read_vxz_num_voxels,
    select_records_by_shape_1024_voxels,
)


def test_exclude_configured_assets_removes_only_named_successful_asset() -> None:
    excluded = "b4860fb771bfab25ef17224e38b729298a20adaa9207e573ced065fa50644129"
    retained = "a" * 64
    records = [{"asset_id": excluded, "status": "ok"}, {"asset_id": retained, "status": "ok"}]
    config = {"stages": {"encode": {"exclude_asset_ids": [excluded]}}}

    assert exclude_configured_assets(records, config) == [records[1]]
    assert records[0]["status"] == "ok"


def test_stage_05_does_not_encode_configured_excluded_asset(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    source = "ObjaverseXL_sketchfab"
    shard = "ObjaverseXL_sketchfab-00000"
    batch = "batch000"
    retained, excluded = "a" * 64, "b" * 64
    control = tmp_path / "control"
    batch_file = control / "shards" / source / shard / f"{batch}.txt"
    batch_file.parent.mkdir(parents=True)
    batch_file.write_text(f"{retained}\n{excluded}\n")
    work_base = tmp_path / "work"
    work = work_base / source / shard / batch
    voxel = work / "04_voxelize"
    voxel.mkdir(parents=True)
    (voxel / "manifest.jsonl").write_text(
        "".join(json.dumps({"asset_id": asset, "status": "ok"}) + "\n" for asset in (retained, excluded))
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": {"source": source},
        "batch": {"shard": shard, "name": batch},
        "paths": {"control_root": str(control), "work_root": str(work_base), "existing_prepared_root": str(tmp_path / "prepared")},
        "stages": {"encode": {
            "resolutions": "256,512,1024", "ss_resolution": 64, "view_indices": "0-1",
            "loader_workers": 1, "saver_workers": 1, "micro_batch_sizes": "64:1,256:1,512:1,1024:1",
            "latent_dtype": "float16", "gpu_memory_target_percent": 90.0,
            "timeout_seconds": 30, "exclude_asset_ids": [excluded],
        }},
    }))
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0",
        "PIXAL3D_LEAF_WORKER": str(repository / "tests/data_toolkit/fixtures/fake_leaf_worker.py"),
    }

    result = subprocess.run(
        [sys.executable, str(repository / "data_toolkit/preprocess/05_encode/run.py"),
         "--config", str(config), "--world-size", "1", "--rank", "0"],
        cwd=repository, env=environment, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (work / "05_encode/instances.txt").read_text().splitlines() == [retained]
    records = [json.loads(line) for line in (work / "05_encode/manifest.jsonl").read_text().splitlines()]
    assert [record["asset_id"] for record in records] == [retained]
    assert (work / "05_encode/shape/shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view" / retained / "view00.npz").is_file()
    assert not list((work / "05_encode").rglob(f"{excluded}/view00.npz"))


def test_stage_05_skips_only_unfinished_assets_above_voxel_limit(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    source, shard, batch = "ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000", "batch000"
    at_limit, unfinished, complete, missing_scale = (letter * 64 for letter in "abcd")
    control = tmp_path / "control"
    batch_file = control / "shards" / source / shard / f"{batch}.txt"
    batch_file.parent.mkdir(parents=True)
    batch_file.write_text("\n".join((at_limit, unfinished, complete, missing_scale)) + "\n")
    work = tmp_path / "work" / source / shard / batch
    voxel = work / "04_voxelize"
    voxel.mkdir(parents=True)
    (voxel / "manifest.jsonl").write_text("".join(
        json.dumps({"asset_id": asset, "status": "ok"}) + "\n"
        for asset in (at_limit, unfinished, complete, missing_scale)
    ))
    for asset, count in ((at_limit, 15_000_000), (unfinished, 15_000_001),
                         (complete, 20_000_000), (missing_scale, 20_000_001)):
        for view in (0, 1):
            _write_vxz_header(voxel / "dual_grid_view_1024" / asset / f"view{view:02d}.vxz", count)
    encode = work / "05_encode"
    for asset in (complete, missing_scale):
        for path in expected_latent_paths(
            [{"asset_id": asset}], encode,
            resolutions=(256, 512, 1024), ss_resolution=64, view_indices=(0, 1),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing")
            if asset == complete:
                path.with_name(f"{path.stem}_scale.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "dataset": {"source": source},
        "batch": {"shard": shard, "name": batch},
        "paths": {"control_root": str(control), "work_root": str(tmp_path / "work"),
                  "existing_prepared_root": str(tmp_path / "prepared")},
        "stages": {"encode": {
            "resolutions": "256,512,1024", "ss_resolution": 64, "view_indices": "0-1",
            "loader_workers": 1, "saver_workers": 1, "micro_batch_sizes": "64:1,256:1,512:1,1024:1",
            "latent_dtype": "float16", "gpu_memory_target_percent": 90.0,
            "timeout_seconds": 30, "max_new_shape_1024_voxels": 15_000_000,
        }},
    }))
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0",
        "PIXAL3D_LEAF_WORKER": str(repository / "tests/data_toolkit/fixtures/fake_leaf_worker.py"),
    }

    result = subprocess.run(
        [sys.executable, str(repository / "data_toolkit/preprocess/05_encode/run.py"),
         "--config", str(config), "--world-size", "1", "--rank", "0"],
        cwd=repository, env=environment, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (encode / "instances.txt").read_text().splitlines() == [at_limit]
    records = [json.loads(line) for line in (encode / "manifest.jsonl").read_text().splitlines()]
    assert [(record["asset_id"], record["status"]) for record in records] == [
        (at_limit, "ok"), (complete, "ok"), (unfinished, "skipped_oversized"),
        (missing_scale, "skipped_oversized"),
    ]
    assert [record["shape_1024_voxels"] for record in records[-2:]] == [
        15_000_001, 20_000_001,
    ]


def _write_vxz_header(path: Path, num_voxels: int) -> None:
    header = json.dumps({"num_voxel": num_voxels}).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"VXZ\x00" + struct.pack(">I", 8 + len(header)) + header)


def test_read_vxz_num_voxels_reads_header_without_payload(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "view00.vxz"
    _write_vxz_header(path, 1_234_567)
    path.write_bytes(path.read_bytes() + b"payload-must-not-be-parsed")

    # When
    value = read_vxz_num_voxels(path)

    # Then
    assert value == 1_234_567


def test_select_records_uses_largest_view_for_disjoint_voxel_ranges(tmp_path: Path) -> None:
    # Given
    voxel_root = tmp_path / "04_voxelize"
    low = "a" * 64
    high = "b" * 64
    _write_vxz_header(
        voxel_root / "dual_grid_view_1024" / low / "view00.vxz", 1_000
    )
    _write_vxz_header(
        voxel_root / "dual_grid_view_1024" / low / "view01.vxz", 1_500
    )
    _write_vxz_header(
        voxel_root / "dual_grid_view_1024" / high / "view00.vxz", 2_500
    )
    _write_vxz_header(
        voxel_root / "dual_grid_view_1024" / high / "view01.vxz", 2_000
    )
    records = [{"asset_id": low, "status": "ok"}, {"asset_id": high, "status": "ok"}]

    # When
    local = select_records_by_shape_1024_voxels(
        records, voxel_root, view_indices=(0, 1), maximum=1_500,
    )
    n17 = select_records_by_shape_1024_voxels(
        records, voxel_root, view_indices=(0, 1), minimum=1_501,
    )

    # Then
    assert [record["asset_id"] for record in local] == [low]
    assert [record["asset_id"] for record in n17] == [high]


def test_expected_latent_paths_cover_every_encoder_family_and_view(tmp_path: Path) -> None:
    # Given
    asset_id = "c" * 64

    # When
    paths = expected_latent_paths(
        [{"asset_id": asset_id, "status": "ok"}],
        tmp_path,
        resolutions=(256, 1024),
        ss_resolution=64,
        view_indices=(0, 1),
    )

    # Then
    assert len(paths) == 10
    assert (
        tmp_path
        / "shape"
        / "shape_latents"
        / "shape_enc_next_dc_f16c32_fp16_1024_view"
        / asset_id
        / "view01.npz"
    ) in paths
    assert (
        tmp_path
        / "pbr"
        / "pbr_latents"
        / "tex_enc_next_dc_f16c32_fp16_256_view_fix"
        / asset_id
        / "view00.npz"
    ) in paths
    assert (
        tmp_path
        / "ss"
        / "ss_latents"
        / "ss_enc_conv3d_16l8_fp16_64_view"
        / asset_id
        / "view01.npz"
    ) in paths
