import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from data_toolkit import (
    encode_latent_bundle,
    geometry_bundle,
    geometry_encode_bundle,
)
from data_toolkit.pipeline.instance_manifest import read_asset_ids
from data_toolkit.pipeline.commands import ShardContext, build_preprocessing_dag


def test_dag_bundles_geometry_and_encoder_lifecycles(config, tmp_path):
    # Given
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

    # When
    dag = build_preprocessing_dag(context, config)
    commands = {command.name: command for command in dag}

    # Then
    assert "prepare_bundle" in commands
    assert "geometry_encode_bundle" in commands
    assert "geometry_bundle" not in commands
    assert "encode_latent_bundle" not in commands
    assert not any(name.startswith("dual_grid_") for name in commands)
    assert not any(name.startswith("voxelize_pbr_") for name in commands)
    assert not any(name.startswith("encode_shape_") for name in commands)
    assert not any(name.startswith("encode_pbr_") for name in commands)
    assert not any(name.startswith("encode_ss_") for name in commands)
    assert commands["prepare_bundle"].gpu_ranks == 0
    assert commands["geometry_encode_bundle"].gpu_ranks == 0


def test_bundle_leaf_clis_expose_production_controls():
    # Given
    repository = Path(__file__).resolve().parents[2]
    expected = {
        "prepare_bundle.py": ("--render_workers", "--render_workers_per_gpu"),
        "geometry_bundle.py": ("--resolutions", "--native_threads"),
        "encode_latent_bundle.py": ("--resolutions", "--micro_batch_sizes"),
        "geometry_encode_bundle.py": ("--resolutions", "--micro_batch_sizes"),
    }

    # When / Then
    for script, flags in expected.items():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                f"data_toolkit.{script.removesuffix('.py')}",
                "--help",
            ],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert all(flag in result.stdout for flag in flags)


def test_geometry_bundle_skips_excluded_pbr_families(tmp_path, monkeypatch):
    asset = "a" * 64
    instances = tmp_path / "all.txt"
    instances.write_text(f"{asset}\n")
    family_paths = {}
    for family, contents in (("shape", f"{asset}\n"), ("PBR", "")):
        for resolution in (256, 512, 1024):
            path = tmp_path / f"{family}-{resolution}.txt"
            path.write_text(contents)
            family_paths[f"{family}-{resolution}"] = str(path)
    ss = tmp_path / "SS-64.txt"
    ss.write_text(f"{asset}\n")
    family_paths["SS-64"] = str(ss)
    family_manifest = tmp_path / "families.json"
    family_manifest.write_text(json.dumps(family_paths))
    launched = []
    monkeypatch.setattr(
        geometry_bundle,
        "run_bounded",
        lambda commands, max_processes: launched.extend(
            command for command, _environment in commands
        ),
    )

    assert geometry_bundle.main([
        "ABO", "--root", str(tmp_path), "--instances", str(instances),
        "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
        "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
        "--resolutions", "256,512,1024", "--view_indices", "0-1",
        "--ss_resolution", "64",
        "--max_workers", "11", "--native_threads", "1",
        "--family_instances_file", str(family_manifest),
    ]) == 0

    assert len(launched) == 2
    assert all("dual_grid_view.py" in command[1] for command in launched)
    assert all(command[command.index("--resolution") + 1] == "256,512,1024" for command in launched)
    assert [
        command[command.index("--max_workers") + 1]
        for command in launched
    ] == ["6", "5"]
    assert all(command[command.index("--native_threads") + 1] == "1" for command in launched)


def test_geometry_bundle_assigns_disjoint_affinity_lanes(tmp_path, monkeypatch):
    # Given
    assets = tuple(f"{index:064x}" for index in range(64))
    instances = tmp_path / "all.txt"
    instances.write_text("\n".join(assets) + "\n")
    launched = []
    monkeypatch.setattr(
        geometry_bundle,
        "run_bounded",
        lambda commands, max_processes: launched.extend(commands),
    )

    # When
    result = geometry_bundle.main([
        "ABO", "--root", str(tmp_path), "--instances", str(instances),
        "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
        "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
        "--resolutions", "256,512,1024", "--view_indices", "0-1",
        "--max_workers", "44", "--native_threads", "1",
    ])

    # Then
    assert result == 0
    assert [
        command[command.index("--max_workers") + 1]
        for command, _environment in launched
    ] == ["11", "11", "11", "11"]
    assert [
        environment["PIXAL3D_GEOMETRY_AFFINITY_OFFSET"]
        for _command, environment in launched
    ] == ["0", "11", "22", "33"]


def test_geometry_bundle_reuses_affinity_only_after_a_wave_finishes(
    tmp_path,
    monkeypatch,
):
    # Given
    asset = "a" * 64
    instances = tmp_path / "all.txt"
    instances.write_text(f"{asset}\n")
    waves = []
    monkeypatch.setattr(
        geometry_bundle,
        "run_bounded",
        lambda commands, max_processes: waves.append(tuple(commands)),
    )

    # When
    result = geometry_bundle.main([
        "ABO", "--root", str(tmp_path), "--instances", str(instances),
        "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
        "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
        "--resolutions", "256", "--view_indices", "0-5",
        "--max_workers", "4", "--native_threads", "4",
    ])

    # Then
    assert result == 0
    assert [len(wave) for wave in waves] == [4, 4, 4]
    assert all(
        [
            environment["PIXAL3D_GEOMETRY_AFFINITY_OFFSET"]
            for _command, environment in wave
        ]
        == ["0", "1", "2", "3"]
        for wave in waves
    )


def test_encoder_bundle_skips_excluded_pbr_families(tmp_path, monkeypatch):
    asset = "a" * 64
    instances = tmp_path / "all.txt"
    shape = tmp_path / "shape.txt"
    pbr = tmp_path / "pbr.txt"
    instances.write_text(f"{asset}\n")
    shape.write_text(f"{asset}\n")
    pbr.write_text("")
    family_paths = {
        **{f"shape-{resolution}": str(shape) for resolution in (256, 512, 1024)},
        **{f"PBR-{resolution}": str(pbr) for resolution in (256, 512, 1024)},
        "SS-64": str(shape),
    }
    family_manifest = tmp_path / "families.json"
    family_manifest.write_text(json.dumps(family_paths))
    monkeypatch.setenv("PIXAL3D_LEAF_WORKER", "fake.py")
    launched = []
    monkeypatch.setattr(
        encode_latent_bundle,
        "_run_fake_leaf",
        lambda script, arguments, override: launched.append(script.name),
    )

    assert encode_latent_bundle.main([
        "--root", str(tmp_path), "--instances", str(instances),
        "--dual_grid_root", str(tmp_path), "--pbr_voxel_root", str(tmp_path),
        "--shape_latent_root", str(tmp_path), "--pbr_latent_root", str(tmp_path),
        "--ss_latent_root", str(tmp_path), "--resolutions", "256,512,1024",
        "--ss_resolution", "64", "--loader_workers", "1", "--saver_workers", "1",
        "--latent_dtype", "float16", "--micro_batch_sizes", "64:1,256:1,512:1,1024:1",
        "--gpu_memory_target_percent", "80",
        "--family_instances_file", str(family_manifest),
    ]) == 0

    assert launched == [
        "encode_shape_latent_view.py",
        "encode_shape_latent_view.py",
        "encode_shape_latent_view.py",
        "encode_ss_latent_view.py",
    ]


def test_fused_encoder_preserves_legacy_sibling_imports(tmp_path):
    helper = tmp_path / "legacy_bundle_helper.py"
    worker = tmp_path / "worker.py"
    output = tmp_path / "result.txt"
    helper.write_text("VALUE = 'loaded'\n")
    worker.write_text(
        "import pathlib, sys\n"
        "from legacy_bundle_helper import VALUE\n"
        "pathlib.Path(sys.argv[1]).write_text(VALUE)\n"
    )

    encode_latent_bundle._run_leaf(worker, (str(output),))

    assert output.read_text() == "loaded"


def test_family_manifest_fails_closed_on_missing_key(tmp_path, monkeypatch):
    asset = "a" * 64
    instances = tmp_path / "all.txt"
    instances.write_text(f"{asset}\n")
    family_manifest = tmp_path / "families.json"
    family_manifest.write_text(json.dumps({"shape-256": str(instances)}))

    with pytest.raises(ValueError, match="unexpected family keys"):
        geometry_bundle.main([
            "ABO", "--root", str(tmp_path), "--instances", str(instances),
            "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
            "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
            "--resolutions", "256", "--max_workers", "1",
            "--ss_resolution", "64",
            "--native_threads", "1",
            "--family_instances_file", str(family_manifest),
        ])


def test_instance_reader_rejects_symlink_and_unsorted_ids(tmp_path):
    first = "a" * 64
    second = "b" * 64
    target = tmp_path / "target.txt"
    target.write_text(f"{second}\n{first}\n")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(OSError):
        read_asset_ids(link)
    with pytest.raises(ValueError, match="sorted and unique"):
        read_asset_ids(target)


def test_encoder_rank_count_rejects_gpu_allowlist_mismatch(tmp_path, monkeypatch):
    asset = "a" * 64
    instances = tmp_path / "instances.txt"
    instances.write_text(f"{asset}\n")
    launched = []

    class Process:
        args = ("worker",)

        def __init__(self, command, env=None):
            self.command = command
            self.env = env or os.environ.copy()
            launched.append(self)

        def poll(self):
            return 0

    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")
    monkeypatch.setattr(geometry_encode_bundle.subprocess, "Popen", Process)
    monkeypatch.setattr(geometry_encode_bundle, "wait_all", lambda _processes: None)

    with pytest.raises(ValueError, match="exactly 7 GPU indices"):
        geometry_encode_bundle.main([
            "ABO", "--root", str(tmp_path), "--instances", str(instances),
            "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
            "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
            "--shape_latent_root", str(tmp_path), "--pbr_latent_root", str(tmp_path),
            "--ss_latent_root", str(tmp_path), "--resolutions", "256,512,1024",
            "--view_indices", "0-1", "--latent_dtype", "float16",
            "--micro_batch_sizes", "64:1,256:1,512:1,1024:1",
            "--ss_resolution", "64", "--max_workers", "1", "--native_threads", "1",
            "--loader_workers", "1", "--saver_workers", "1", "--encoder_ranks", "7",
            "--gpu_count", "7", "--timeout_seconds", "1",
            "--gpu_memory_target_percent", "80",
        ])

    assert launched == []


def test_empty_bundle_still_validates_family_manifest(tmp_path):
    instances = tmp_path / "instances.txt"
    instances.write_text("")
    family_manifest = tmp_path / "families.json"
    family_manifest.write_text(json.dumps({"shape-256": str(instances)}))

    with pytest.raises(ValueError, match="unexpected family keys"):
        geometry_encode_bundle.main([
            "ABO", "--root", str(tmp_path), "--instances", str(instances),
            "--family_instances_file", str(family_manifest),
            "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
            "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
            "--shape_latent_root", str(tmp_path),
            "--pbr_latent_root", str(tmp_path), "--ss_latent_root", str(tmp_path),
            "--resolutions", "256", "--view_indices", "0-1",
            "--latent_dtype", "float16", "--micro_batch_sizes", "64:1,256:1",
            "--ss_resolution", "64", "--max_workers", "1",
            "--native_threads", "1", "--loader_workers", "1",
            "--saver_workers", "1", "--encoder_ranks", "1", "--gpu_count", "1",
            "--timeout_seconds", "1", "--gpu_memory_target_percent", "80",
        ])
