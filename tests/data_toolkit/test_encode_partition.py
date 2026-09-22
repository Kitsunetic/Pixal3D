import json
import struct
from pathlib import Path

from data_toolkit.preprocess._common.encode_partition import (
    expected_latent_paths,
    read_vxz_num_voxels,
    select_records_by_shape_1024_voxels,
)


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
