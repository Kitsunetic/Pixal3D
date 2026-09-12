import json
import os

from data_toolkit import geometry_encode_bundle


def _write_assets(path, count):
    path.write_text("".join(f"{index:064x}\n" for index in range(count)))


def test_encoder_wait_covers_all_geometry_worker_waves(tmp_path, monkeypatch):
    # Given: 64 assets split across four geometry jobs and 44 workers.
    instances = tmp_path / "instances.txt"
    _write_assets(instances, 64)
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
    monkeypatch.setattr(
        geometry_encode_bundle,
        "wait_all",
        lambda _processes: None,
    )

    # When: the pipelined geometry/encoder bundle is launched.
    result = geometry_encode_bundle.main([
        "ABO", "--root", str(tmp_path), "--instances", str(instances),
        "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
        "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
        "--shape_latent_root", str(tmp_path),
        "--pbr_latent_root", str(tmp_path),
        "--ss_latent_root", str(tmp_path),
        "--resolutions", "256,512,1024", "--view_indices", "0-1",
        "--latent_dtype", "float16",
        "--micro_batch_sizes", "64:1,256:1,512:1,1024:1",
        "--ss_resolution", "64", "--max_workers", "44",
        "--native_threads", "1", "--loader_workers", "1",
        "--saver_workers", "1", "--encoder_ranks", "1",
        "--gpu_count", "1", "--timeout_seconds", "1",
        "--gpu_memory_target_percent", "80",
    ])

    # Then: six producer waves, not one leaf timeout, bound encoder waits.
    encoder = next(
        process
        for process in launched
        if "data_toolkit.encode_latent_bundle" in process.command
    )
    timeout_index = encoder.command.index("--timeout_seconds") + 1
    assert result == 0
    assert encoder.command[timeout_index] == "6"
    assert encoder.env["PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS"] == "6"


def test_encoder_wait_uses_family_manifest_worker_waves(tmp_path, monkeypatch):
    # Given: the production topology of 12 jobs sharing 44 workers.
    instances = tmp_path / "instances.txt"
    _write_assets(instances, 63)
    family_paths = {}
    for family, count in (
        ("shape-256", 63), ("shape-512", 63), ("shape-1024", 63),
        ("PBR-256", 55), ("PBR-512", 55), ("PBR-1024", 55),
        ("SS-64", 63),
    ):
        path = tmp_path / f"{family}.txt"
        _write_assets(path, count)
        family_paths[family] = str(path)
    manifest = tmp_path / "families.json"
    manifest.write_text(json.dumps(family_paths))
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
    monkeypatch.setattr(
        geometry_encode_bundle,
        "wait_all",
        lambda _processes: None,
    )

    # When: the production-shaped geometry/encoder bundle is launched.
    geometry_encode_bundle.main([
        "ObjaverseXL", "--root", str(tmp_path),
        "--instances", str(instances),
        "--family_instances_file", str(manifest),
        "--mesh_dump_root", str(tmp_path), "--pbr_dump_root", str(tmp_path),
        "--transform_root", str(tmp_path), "--voxel_root", str(tmp_path),
        "--shape_latent_root", str(tmp_path),
        "--pbr_latent_root", str(tmp_path), "--ss_latent_root", str(tmp_path),
        "--resolutions", "256,512,1024", "--view_indices", "0-1",
        "--latent_dtype", "float32",
        "--micro_batch_sizes", "64:16,256:16,512:8,1024:4",
        "--ss_resolution", "64", "--max_workers", "44",
        "--native_threads", "1", "--loader_workers", "2",
        "--saver_workers", "1", "--encoder_ranks", "1",
        "--gpu_count", "1", "--timeout_seconds", "900",
        "--gpu_memory_target_percent", "80",
    ])

    # Then: 63 assets at three workers per job require 21 producer waves.
    encoder = next(
        process
        for process in launched
        if "data_toolkit.encode_latent_bundle" in process.command
    )
    timeout_index = encoder.command.index("--timeout_seconds") + 1
    assert encoder.command[timeout_index] == "18900"
    assert encoder.env["PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS"] == "18900"
