"""End-to-end contracts for validating and merging finalized batch packs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
from PIL import Image
import yaml

from data_toolkit.pipeline.packing import PACK_FAMILIES, verify_pack
from data_toolkit.preprocess._common.finalize_packs import BatchIdentity, pack_relative
from data_toolkit.preprocess._common.runtime import legacy_prepared_batch_complete


SOURCE = "ObjaverseXL_sketchfab"
SHARD = "ObjaverseXL_sketchfab-00000"
BATCH = "batch000"
ASSET = "a" * 64
REPOSITORY = Path(__file__).resolve().parents[2]
PROGRAM = REPOSITORY / "data_toolkit/preprocess/06_finalize/run.py"
RANK_PROGRAM = REPOSITORY / "data_toolkit/preprocess/00_rank/run.py"


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    control = tmp_path / "control"
    batch_file = control / "shards" / SOURCE / SHARD / f"{BATCH}.txt"
    batch_file.parent.mkdir(parents=True)
    batch_file.write_text(f"{ASSET}\n")
    metadata = control / "metadata" / SOURCE / "metadata.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"sha256,aesthetic_score\n{ASSET},5.5\n")
    work = tmp_path / "work" / SOURCE / SHARD / BATCH
    voxel = work / "04_voxelize"
    voxel.mkdir(parents=True)
    (voxel / "manifest.jsonl").write_text(
        json.dumps({"asset_id": ASSET, "status": "ok"}) + "\n"
    )
    render = work / "03_render" / "renders_cond" / ASSET
    render.mkdir(parents=True)
    for view in range(8):
        Image.new("RGBA", (8, 8), (128, 64, 32, 255)).save(render / f"{view:03d}.png")
    (render / "transforms.json").write_text(json.dumps({
        "camera_angle_x": 0.7,
        "frames": [{"file_path": f"{view:03d}.png", "transform_matrix": [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1],
        ]} for view in range(8)],
    }))
    encode = work / "05_encode"
    for family, directory in (
        ("shape", "shape_enc_next_dc_f16c32_fp16_256_view"),
        ("shape", "shape_enc_next_dc_f16c32_fp16_512_view"),
        ("shape", "shape_enc_next_dc_f16c32_fp16_1024_view"),
        ("pbr", "tex_enc_next_dc_f16c32_fp16_256_view_fix"),
        ("pbr", "tex_enc_next_dc_f16c32_fp16_512_view_fix"),
        ("pbr", "tex_enc_next_dc_f16c32_fp16_1024_view_fix"),
        ("ss", "ss_enc_conv3d_16l8_fp16_64_view"),
    ):
        root = encode / family / f"{family}_latents" / directory / ASSET
        root.mkdir(parents=True)
        for view in (0, 1):
            if family == "ss":
                np.savez_compressed(root / f"view{view:02d}.npz", z=np.zeros((1, 16, 16, 16), dtype=np.float16))
            else:
                np.savez_compressed(
                    root / f"view{view:02d}.npz",
                    coords=np.array([[1, 2, 3]], dtype=np.uint8),
                    feats=np.ones((1, 32), dtype=np.float16),
                )
            (root / f"view{view:02d}_scale.json").write_text('{"total_scale": 1.0}')
    prepared = tmp_path / "prepared"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"source": SOURCE},
                "batch": {"shard": SHARD, "name": BATCH},
                "paths": {
                    "control_root": str(control),
                    "existing_prepared_root": str(prepared),
                    "work_root": str(tmp_path / "work"),
                    "prepared_root": str(prepared),
                    "local_temp_root": str(tmp_path / "local-temporary"),
                },
                "stages": {
                    "encode": {
                        "resolutions": "256,512,1024",
                        "ss_resolution": 64,
                        "view_indices": "0-1",
                    }
                },
            }
        )
    )
    return config, work, prepared


def _run(config: Path, *, publish: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PROGRAM),
        "--config",
        str(config),
        "--world-size",
        "1",
        "--rank",
        "0",
    ]
    if publish:
        command.append("--publish")
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )


def test_finalize_publishes_eight_verified_family_tars_when_batch_complete(
    tmp_path: Path,
) -> None:
    # Given
    config, work, prepared = _fixture(tmp_path)
    (tmp_path / "control/shards" / SOURCE / SHARD / f"{BATCH}.txt").write_text(
        f"{ASSET}\n{'b' * 64}\n"
    )
    existing = {
        family: {
            "pack": pack_relative(BatchIdentity(SOURCE, SHARD, "batch001"), family).as_posix(),
            "pack_sha256": "existing-pack",
            "manifest": pack_relative(BatchIdentity(SOURCE, SHARD, "batch001"), family).with_suffix(".tar.manifest.json").as_posix(),
            "manifest_sha256": "existing-manifest",
        }
        for family in PACK_FAMILIES
    }
    index_path = prepared / "index" / SOURCE / f"{SHARD}.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps({
        "gate": "production", "source": SOURCE, "shard_id": SHARD,
        "batches": {"batch001": existing},
    }))

    # When
    result = _run(config)

    # Then
    assert result.returncode == 0, result.stderr
    index = json.loads(index_path.read_text())
    assert index["batches"]["batch001"] == existing
    assert set(index["batches"][BATCH]) == set(PACK_FAMILIES)
    for family, entry in index["batches"][BATCH].items():
        archive = prepared / entry["pack"]
        manifest = prepared / entry["manifest"]
        verify_pack(archive, manifest)
        packed = json.loads(manifest.read_text())
        assert packed["family"] == family
        assert packed["asset_sha256s"] == [ASSET, "b" * 64]
        assert packed["included_asset_sha256s"] == [ASSET]
    with tarfile.open(prepared / index["batches"][BATCH]["common"]["pack"]) as archive:
        assert f"renders_cond/{ASSET}/000.png" in archive.getnames()
    with tarfile.open(prepared / index["batches"][BATCH]["shape-1024"]["pack"]) as archive:
        assert f"shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view/{ASSET}/view00_scale.json" in archive.getnames()
    assert legacy_prepared_batch_complete(
        yaml.safe_load(config.read_text()), SOURCE, SHARD, BATCH
    )
    report = json.loads((work / "06_finalize/report.json").read_text())
    assert report["published"] and report["loader_validated"]
    assert (tmp_path / "local-temporary").is_dir()
    assert not list((tmp_path / "local-temporary").rglob("*.tar"))


def test_finalize_does_not_publish_when_scale_is_missing(tmp_path: Path) -> None:
    # Given
    config, work, prepared = _fixture(tmp_path)
    missing = next((work / "05_encode" / "shape").rglob("view00_scale.json"))
    missing.unlink()

    # When
    result = _run(config)

    # Then
    assert result.returncode != 0
    assert not (prepared / "index" / SOURCE / f"{SHARD}.json").exists()
    assert not list(prepared.rglob("*.tar"))


def test_finalize_excludes_configured_asset_without_its_latents(tmp_path: Path) -> None:
    config, work, prepared = _fixture(tmp_path)
    excluded = "b" * 64
    manifest = work / "04_voxelize/manifest.jsonl"
    with manifest.open("a") as stream:
        stream.write(json.dumps({"asset_id": excluded, "status": "ok"}) + "\n")
    batch_file = tmp_path / "control/shards" / SOURCE / SHARD / f"{BATCH}.txt"
    batch_file.write_text(f"{ASSET}\n{excluded}\n")
    data = yaml.safe_load(config.read_text())
    data["stages"]["encode"]["exclude_asset_ids"] = [excluded]
    config.write_text(yaml.safe_dump(data))

    result = _run(config)

    assert result.returncode == 0, result.stderr
    report = json.loads((work / "06_finalize/report.json").read_text())
    assert report["assets"] == 1
    index = json.loads((prepared / "index" / SOURCE / f"{SHARD}.json").read_text())
    for entry in index["batches"][BATCH].values():
        packed = json.loads((prepared / entry["manifest"]).read_text())
        assert packed["asset_sha256s"] == [ASSET, excluded]
        assert packed["included_asset_sha256s"] == [ASSET]


def test_finalize_excludes_stage_05_oversized_asset(tmp_path: Path) -> None:
    config, work, prepared = _fixture(tmp_path)
    skipped = "b" * 64
    with (work / "04_voxelize/manifest.jsonl").open("a") as stream:
        stream.write(json.dumps({"asset_id": skipped, "status": "ok"}) + "\n")
    (tmp_path / "control/shards" / SOURCE / SHARD / f"{BATCH}.txt").write_text(
        f"{ASSET}\n{skipped}\n"
    )
    partition = work / "05_encode/partitions/n17-high"
    partition.mkdir(parents=True)
    (partition / "manifest.jsonl").write_text(json.dumps({
        "asset_id": skipped, "status": "skipped_oversized",
        "shape_1024_voxels": 15_000_001,
    }) + "\n")

    result = _run(config)

    assert result.returncode == 0, result.stderr
    report = json.loads((work / "06_finalize/report.json").read_text())
    assert report["assets"] == 1
    index = json.loads((prepared / "index" / SOURCE / f"{SHARD}.json").read_text())
    for entry in index["batches"][BATCH].values():
        packed = json.loads((prepared / entry["manifest"]).read_text())
        assert packed["asset_sha256s"] == [ASSET, skipped]
        assert packed["included_asset_sha256s"] == [ASSET]


def test_finalize_does_not_publish_when_training_loader_rejects_latent(tmp_path: Path) -> None:
    # Given
    config, work, prepared = _fixture(tmp_path)
    corrupted = next((work / "05_encode" / "shape").rglob("view00.npz"))
    corrupted.write_bytes(b"not an npz")

    # When
    result = _run(config)

    # Then
    assert result.returncode != 0
    assert not (prepared / "index" / SOURCE / f"{SHARD}.json").exists()
    assert not list(prepared.rglob("*.tar"))


def test_finalize_does_not_publish_when_second_training_view_is_corrupt(tmp_path: Path) -> None:
    # Given
    config, work, prepared = _fixture(tmp_path)
    image = work / "03_render" / "renders_cond" / ASSET / "001.png"
    image.write_bytes(b"not a png")

    # When
    result = _run(config)

    # Then
    assert result.returncode != 0
    assert not (prepared / "index" / SOURCE / f"{SHARD}.json").exists()
    assert not list(prepared.rglob("*.tar"))


def test_finalize_preserves_unindexed_existing_pack_on_collision(tmp_path: Path) -> None:
    # Given
    config, _, prepared = _fixture(tmp_path)
    existing = prepared / pack_relative(BatchIdentity(SOURCE, SHARD, BATCH), "common")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")

    # When
    result = _run(config)

    # Then
    assert result.returncode != 0
    assert existing.read_bytes() == b"existing"
    assert not (prepared / "index" / SOURCE / f"{SHARD}.json").exists()
    assert list(prepared.rglob("*.tar")) == [existing]


def test_rank_launcher_publishes_after_validation_only_marker(tmp_path: Path) -> None:
    # Given
    config, work, prepared = _fixture(tmp_path)
    assert _run(config, publish=False).returncode == 0
    assert not json.loads((work / "06_finalize/stage.json").read_text())["published"]

    # When
    result = subprocess.run(
        [
            sys.executable, str(RANK_PROGRAM), "--config", str(config),
            "--world-size", "1", "--rank", "0", "--stages", "06", "--publish",
        ],
        cwd=REPOSITORY, capture_output=True, text=True, check=False,
    )

    # Then
    assert result.returncode == 0, result.stderr
    assert (prepared / "index" / SOURCE / f"{SHARD}.json").is_file()
    assert json.loads((work / "06_finalize/report.json").read_text())["published"]
