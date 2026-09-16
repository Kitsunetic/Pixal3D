from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import pytest

from data_toolkit.geometry_encode_bundle import (
    _gpu_indices as encode_gpu_indices,
)
from data_toolkit.prepare_bundle import (
    _gpu_indices as render_gpu_indices,
    _worker_budget,
)
from data_toolkit.pipeline.commands import (
    CommandSpec,
    ShardContext,
    WorkerProfile,
    build_preprocessing_dag,
    choose_worker_profile,
    expand_ranked,
    select_render_workers,
)
from data_toolkit.pipeline.parallelism import geometry_profile
from data_toolkit.pipeline.subprocess_control import wait_all


CPU_ENV = (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)
RENDER_ENV = (
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
)
ENCODE_ENV = CPU_ENV + (
    ("ATTN_BACKEND", "sdpa"),
    ("SPARSE_ATTN_BACKEND", "sdpa"),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
)


class _FakeEncoderProcess:
    def __init__(self, status):
        self.status = status
        self.args = ("encoder",)
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.status

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return -15 if self.terminated else self.status


def test_encoder_bundle_terminates_geometry_and_siblings_after_rank_failure():
    geometry = _FakeEncoderProcess(None)
    hanging = _FakeEncoderProcess(None)
    failed = _FakeEncoderProcess(7)

    with pytest.raises(subprocess.CalledProcessError, match="exit status 7"):
        wait_all(
            (
                geometry,
                hanging,
                failed,
            )
        )

    assert geometry.terminated is True
    assert geometry.waited is True
    assert hanging.terminated is True
    assert hanging.waited is True


def _by_name(dag, name):
    return next(command for command in dag if command.name == name)


def _python_command(script, *args):
    if script in {
        "prepare_bundle.py",
        "geometry_bundle.py",
        "encode_latent_bundle.py",
        "geometry_encode_bundle.py",
    }:
        return (
            sys.executable,
            "-m",
            f"data_toolkit.{script.removesuffix('.py')}",
            *args,
        )
    return (sys.executable, f"data_toolkit/{script}", *args)


def test_worker_profile_ramps_after_stable_telemetry(config):
    first = WorkerProfile(32, 8, 4, 7, 7, 2)
    stable = [
        {"cpu_percent": 60, "io_wait_percent": 2, "available_ram_gib": 200, "reasons": []}
    ] * 3
    second = choose_worker_profile(stable, config, first)
    assert second.dump_workers == 36
    assert (second.voxel_workers, second.voxel_threads_per_worker) == (10, 4)


def test_initial_worker_profile_owns_full_geometry_lane(config):
    selected = choose_worker_profile((), config)

    assert selected.dump_workers == 44
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (44, 1)
    assert selected.render_workers_per_gpu == 3


def test_worker_profile_steps_down_on_pressure(config):
    previous = WorkerProfile(40, 10, 4, 7, 7)
    pressure = [{"cpu_percent": 70, "io_wait_percent": 12, "available_ram_gib": 100, "reasons": ["I/O wait"]}]
    selected = choose_worker_profile(pressure, config, previous)
    assert selected.dump_workers == 36
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (8, 4)


def test_geometry_profile_steps_up_only_after_three_stable_boundaries(config):
    previous = WorkerProfile(32, 8, 4, 7, 7, 2)
    stable = {
        "cpu_percent": 60,
        "io_wait_percent": 2,
        "available_ram_gib": 200,
        "reasons": [],
    }

    assert choose_worker_profile([stable] * 2, config, previous) == previous
    selected = choose_worker_profile([stable] * 3, config, previous)
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (10, 4)


def test_worker_profile_applies_render_step_at_next_dag_boundary(config):
    previous = choose_worker_profile((), config)
    stable_gpu = [
        {
            "cpu_percent": 60,
            "io_wait_percent": 2,
            "available_ram_gib": 200,
            "reasons": [],
            "gpu_metrics": [
                {
                    "memory_used_mib": 60_000,
                    "memory_total_mib": 97_887,
                    "temperature_celsius": 70,
                }
            ],
        }
    ] * 3

    selected = choose_worker_profile(stable_gpu, config, previous)

    assert previous.render_workers_per_gpu == 3
    assert selected.render_workers_per_gpu == 4
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (44, 1)


def test_full_geometry_profile_steps_down_to_existing_safe_profile_on_pressure(config):
    previous = choose_worker_profile((), config)
    pressure = [{
        "cpu_percent": 91,
        "io_wait_percent": 0,
        "available_ram_gib": 200,
        "reasons": [],
    }]

    selected = choose_worker_profile(pressure, config, previous)

    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (11, 4)


def test_full_geometry_profile_ignores_soft_cpu_target(config):
    # Given
    previous = choose_worker_profile((), config)
    soft_cpu_load = [{
        "cpu_percent": 80.4,
        "io_wait_percent": 0,
        "available_ram_gib": 200,
        "reasons": [],
    }]

    # When
    selected = choose_worker_profile(soft_cpu_load, config, previous)

    # Then
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (44, 1)


def test_reduced_geometry_profile_recovers_full_lane_when_stable(config):
    # Given
    previous = WorkerProfile(40, 11, 4, 7, 7, 2)
    stable = [{
        "cpu_percent": 60,
        "io_wait_percent": 2,
        "available_ram_gib": 200,
        "reasons": [],
    }] * 3

    # When
    selected = choose_worker_profile(stable, config, previous)

    # Then
    assert (selected.voxel_workers, selected.voxel_threads_per_worker) == (44, 1)


def test_dag_accepts_worker_profile(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    profile = WorkerProfile(44, 11, 4, 7, 7)
    dag = build_preprocessing_dag(context, config, profile)
    assert str(profile.dump_workers) in _by_name(dag, "prepare_bundle").argv
    assert str(profile.voxel_workers) in _by_name(
        dag, "geometry_encode_bundle"
    ).argv


def test_dag_has_exact_order_for_all_configured_resolutions(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

    dag = build_preprocessing_dag(context, config)

    expected_names = ["download", "stage_raw", "prepare_bundle"]
    expected_names.append("geometry_encode_bundle")
    expected_names.extend(
        f"cleanup_voxels_{resolution}"
        for resolution in config.targets.resolutions
    )
    expected_names.extend(
        [
            "validate_outputs",
            "build_packs",
            "archive_raw",
            "cleanup_local",
        ]
    )
    assert [command.name for command in dag] == expected_names


def test_commands_have_exact_parser_compatible_argv(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    base = (
        "ABO",
        "--root",
        str(context.metadata_root),
        "--instances",
        str(context.instances),
    )

    assert _by_name(dag, "download").argv == _python_command(
        "download.py",
        *base,
        "--download_root",
        str(context.source_root),
        "--record_root",
        str(context.metadata_root / "download_records"),
        "--max_workers",
        "8",
        "--record_prefix",
        "ABO-00000_batch000_",
    )
    blender = str(
        config.paths.local_root
        / "tools"
        / f"blender-{config.render.blender_version}-linux-x64"
        / "blender"
    )
    assert _by_name(dag, "prepare_bundle").argv == _python_command(
        "prepare_bundle.py",
        *base,
        "--download_root",
        str(context.download_root),
        "--work_root",
        str(context.work_root),
        "--output_root",
        str(context.output_root),
        "--num_cond_views",
        str(config.render.num_views),
        "--cond_resolution",
        str(config.render.resolution),
        "--boundary_fit_resolution",
        "128",
        "--boundary_fit_engine",
        "BLENDER_EEVEE_NEXT",
        "--boundary_fit_samples",
        "1",
        "--renderer_mode",
        "external",
        "--native_worker_max_assets",
        "8",
        "--blender_path",
        blender,
        "--cycles_device",
        config.render.cycles_device,
        "--dump_workers",
        str(config.workers.dump_workers),
        "--render_workers",
        str(config.workers.render_workers),
        "--render_workers_per_gpu",
        str(config.parallelism.render_workers_per_gpu_steps[1]),
        "--gpu_count",
        str(config.parallelism.gpu_count),
    )

    geometry = geometry_profile(config.parallelism)
    resolutions = ",".join(str(value) for value in config.targets.resolutions)
    ss_resolution = config.targets.ss_resolution
    micro_batches = ",".join(
        f"{value}:{config.parallelism.micro_batch(value)}"
        for value in (*config.targets.resolutions, ss_resolution)
    )
    assert _by_name(dag, "geometry_encode_bundle").argv == _python_command(
        "geometry_encode_bundle.py",
        *base,
        "--mesh_dump_root",
        str(context.work_root),
        "--pbr_dump_root",
        str(context.work_root),
        "--transform_root",
        str(context.output_root / "renders_cond"),
        "--voxel_root",
        str(context.work_root),
        "--shape_latent_root",
        str(context.output_root),
        "--pbr_latent_root",
        str(context.output_root),
        "--ss_latent_root",
        str(context.output_root),
        "--resolutions",
        resolutions,
        "--ss_resolution",
        str(ss_resolution),
        "--view_indices",
        "0-1",
        "--max_workers",
        str(geometry.processes),
        "--native_threads",
        str(geometry.native_threads),
        "--loader_workers",
        str(config.workers.encoder_loader_threads),
        "--saver_workers",
        str(config.workers.encoder_saver_threads),
        "--latent_dtype",
        config.targets.latent_dtype,
        "--micro_batch_sizes",
        micro_batches,
        "--gpu_memory_target_percent",
        str(config.parallelism.gpu_memory_target_percent),
        "--encoder_ranks",
        str(config.workers.encoder_ranks),
        "--gpu_count",
        str(config.parallelism.gpu_count),
        "--timeout_seconds",
        "900",
    )


def test_internal_command_names_and_arguments_are_exact(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)

    assert _by_name(dag, "stage_raw").argv == ("internal:stage_raw",)
    for resolution in config.targets.resolutions:
        assert _by_name(dag, f"cleanup_voxels_{resolution}").argv == (
            "internal:cleanup_voxels",
            str(resolution),
        )
    assert _by_name(dag, "validate_outputs").argv == (
        "internal:validate_outputs",
    )
    assert _by_name(dag, "build_packs").argv == ("internal:build_packs",)
    assert _by_name(dag, "archive_raw").argv == ("internal:archive_raw",)
    assert _by_name(dag, "cleanup_local").argv == ("internal:cleanup_local",)


def test_config_context_namespaces_shared_metadata_records(config):
    context = ShardContext.from_config(
        config,
        "ObjaverseXL_sketchfab",
        "ObjaverseXL_sketchfab-00012",
        "batch003",
    )

    assert context.record_prefix == "ObjaverseXL_sketchfab-00012_batch003_"


@pytest.mark.parametrize(
    ("source", "canonical_source"),
    [
        ("ObjaverseXL_sketchfab", "sketchfab"),
        ("ObjaverseXL_github", "github"),
    ],
)
def test_objaversexl_source_mapping(source, canonical_source, config, tmp_path):
    context = ShardContext.for_test(tmp_path, source, f"{source}-00000")
    dag = build_preprocessing_dag(context, config)

    dataset_commands = [
        command
        for command in dag
        if command.name
        in {
            "download",
            "prepare_bundle",
            "geometry_encode_bundle",
        }
    ]
    assert dataset_commands
    for command in dataset_commands:
        dataset_index = command.argv.index("ObjaverseXL")
        assert command.argv[dataset_index : dataset_index + 3] == (
            "ObjaverseXL",
            "--source",
            canonical_source,
        )


def test_render_and_cpu_stages_apply_thread_and_worker_caps(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    render = _by_name(dag, "prepare_bundle")

    assert render.env == RENDER_ENV
    assert render.gpu_ranks == 0
    assert render.workers_per_gpu == 1
    assert render.argv[render.argv.index("--render_workers_per_gpu") + 1] == "3"
    assert render.argv[render.argv.index("--num_cond_views") + 1] == "8"
    assert render.argv[render.argv.index("--cond_resolution") + 1] == "512"
    assert (
        render.argv[render.argv.index("--boundary_fit_resolution") + 1]
        == "128"
    )
    assert (
        render.argv[render.argv.index("--boundary_fit_engine") + 1]
        == "BLENDER_EEVEE_NEXT"
    )
    assert render.argv[render.argv.index("--boundary_fit_samples") + 1] == "1"
    assert render.argv[render.argv.index("--cycles_device") + 1] == "OPTIX"
    assert render.argv[render.argv.index("--blender_path") + 1].endswith(
        "blender-4.5.1-linux-x64/blender"
    )
    assert "--camera_policy" not in render.argv
    assert "--fov_min_degrees" not in render.argv
    assert "--fov_max_degrees" not in render.argv

    external = [
        command
        for command in dag
        if command.argv[0] == sys.executable
    ]
    assert _by_name(dag, "download").env == CPU_ENV
    assert _by_name(dag, "geometry_encode_bundle").env == ENCODE_ENV
    assert (
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    ) in _by_name(dag, "geometry_encode_bundle").env
    assert all(command.gpu_ranks == 0 for command in external)
    assert all(
        command.env == () and command.gpu_ranks == 0
        for command in dag
        if command.argv[0].startswith("internal:")
    )


def test_all_voxel_and_encoder_commands_use_anchor_views(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    dag = build_preprocessing_dag(context, config)
    view_commands = [
        command
        for command in dag
        if command.name == "geometry_encode_bundle"
    ]

    assert view_commands
    assert all(
        command.argv[command.argv.index("--view_indices") + 1] == "0-1"
        for command in view_commands
    )


def test_native_renderer_requires_single_gpu_execution_config(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_RENDERER_MODE", "native")
    context = ShardContext.for_test(
        tmp_path, "ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"
    )

    with pytest.raises(ValueError, match="exactly one visible GPU"):
        build_preprocessing_dag(context, config)


def test_native_renderer_uses_one_long_lived_worker(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_RENDERER_MODE", "native")
    monkeypatch.setenv("PIXAL3D_NATIVE_WORKER_MAX_ASSETS", "8")
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")
    context = ShardContext.for_test(
        tmp_path, "ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"
    )

    dag = build_preprocessing_dag(context, config)
    command = _by_name(dag, "prepare_bundle")
    geometry = _by_name(dag, "geometry_encode_bundle")

    assert command.argv[command.argv.index("--renderer_mode") + 1] == "native"
    assert command.argv[
        command.argv.index("--native_worker_max_assets") + 1
    ] == "8"
    assert command.argv[
        command.argv.index("--render_workers_per_gpu") + 1
    ] == "1"
    assert command.argv[command.argv.index("--gpu_count") + 1] == "1"
    assert geometry.argv[geometry.argv.index("--gpu_count") + 1] == "1"
    assert geometry.argv[geometry.argv.index("--encoder_ranks") + 1] == "1"


def test_native_renderer_setting_falls_back_for_other_dataset_adapters(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_RENDERER_MODE", "native")
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

    command = _by_name(build_preprocessing_dag(context, config), "prepare_bundle")

    assert command.argv[command.argv.index("--renderer_mode") + 1] == "external"


def test_runtime_gpu_allowlist_cannot_exceed_configured_gpu_count(
    config, tmp_path, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0,1,2,3,4,5,6,7")
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")

    with pytest.raises(ValueError, match="configured GPU count"):
        build_preprocessing_dag(context, config)


def test_geometry_encoder_command_uses_runtime_loader_override_without_changing_config_identity(
    config, tmp_path, monkeypatch
):
    # Given
    config = replace(
        config,
        workers=replace(config.workers, cpu_threads=7),
    )
    config_hash = config.config_hash()
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    monkeypatch.setenv("PIXAL3D_ENCODER_LOADER_WORKERS", "4")

    # When
    dag = build_preprocessing_dag(context, config)

    # Then
    command = _by_name(dag, "geometry_encode_bundle")
    assert command.argv[command.argv.index("--loader_workers") + 1] == "4"
    assert config.config_hash() == config_hash


@pytest.mark.parametrize("value", ("", "0", "-1", "+4", "4.0", " 4", "4 "))
def test_geometry_encoder_command_rejects_invalid_runtime_loader_override(
    value, config, tmp_path, monkeypatch
):
    # Given
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    monkeypatch.setenv("PIXAL3D_ENCODER_LOADER_WORKERS", value)

    # When / Then
    with pytest.raises(
        ValueError,
        match="PIXAL3D_ENCODER_LOADER_WORKERS must be a positive integer",
    ):
        build_preprocessing_dag(context, config)


def test_geometry_encoder_command_rejects_runtime_loader_override_above_effective_cpu_budget(
    config, tmp_path, monkeypatch
):
    # Given
    config = replace(
        config,
        workers=replace(config.workers, cpu_threads=7),
    )
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    monkeypatch.setenv("PIXAL3D_ENCODER_LOADER_WORKERS", "8")

    # When / Then
    with pytest.raises(
        ValueError,
        match="PIXAL3D_ENCODER_LOADER_WORKERS must not exceed worker CPU threads",
    ):
        build_preprocessing_dag(context, config)


def test_render_workers_map_round_robin_to_seven_gpus(config, tmp_path):
    assert render_gpu_indices(config.parallelism.gpu_count) == tuple(range(7))
    assert encode_gpu_indices(config.parallelism.gpu_count) == tuple(range(7))


def test_ranked_workers_honor_explicit_gpu_allowlist(config, tmp_path, monkeypatch):
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0,1,2,3,4,5,6")
    expected = tuple(range(7))
    assert render_gpu_indices(config.parallelism.gpu_count) == expected
    assert encode_gpu_indices(config.parallelism.gpu_count) == expected


@pytest.mark.parametrize("value", ("0", "0,1,2,3,4,5,6,7"))
def test_rank_expansion_rejects_gpu_allowlist_cardinality_mismatch(
    value, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", value)

    with pytest.raises(ValueError, match="exactly 7 GPU indices"):
        expand_ranked(CommandSpec("ranked", ("worker",), gpu_ranks=7))


@pytest.mark.parametrize("indices", (render_gpu_indices, encode_gpu_indices))
def test_bundle_gpu_indices_reject_allowlist_cardinality_mismatch(
    indices, monkeypatch
):
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")

    with pytest.raises(ValueError, match="exactly 2 GPU indices"):
        indices(2)


@pytest.mark.parametrize("indices", (render_gpu_indices, encode_gpu_indices))
def test_bundle_gpu_indices_reject_nonpositive_count(indices, monkeypatch):
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0,1")
    with pytest.raises(ValueError, match="GPU count must be positive"):
        indices(-1)


def test_prepare_bundle_splits_one_cpu_budget_across_concurrent_work():
    assert _worker_budget(44, 21) == (21, 11, 23)
    assert _worker_budget(32, 2) == (2, 15, 30)


def test_unranked_command_expands_once_without_mutation():
    command = CommandSpec("cpu", ("python", "script.py"), CPU_ENV)

    assert expand_ranked(command) == ((command.argv, command.env),)
    assert command == CommandSpec("cpu", ("python", "script.py"), CPU_ENV)


@pytest.mark.parametrize(
    "command",
    [
        CommandSpec("negative", ("worker",), gpu_ranks=-1),
        CommandSpec(
            "zero-workers", ("worker",), gpu_ranks=7, workers_per_gpu=0
        ),
        CommandSpec("too-many", ("worker",), gpu_ranks=7, workers_per_gpu=5),
        CommandSpec("cpu-multiplier", ("worker",), workers_per_gpu=2),
    ],
)
def test_rank_expansion_rejects_invalid_worker_counts(command):
    with pytest.raises(ValueError):
        expand_ranked(command)


def test_render_worker_selector_steps_only_at_profile_boundaries():
    steps = (2, 3, 4)

    assert select_render_workers(
        current=2,
        peak_percent=65.0,
        temperature_celsius=70.0,
        failed=False,
        steps=steps,
    ) == 3
    assert select_render_workers(
        current=3,
        peak_percent=81.0,
        temperature_celsius=70.0,
        failed=False,
        steps=steps,
    ) == 2
    assert select_render_workers(
        current=4,
        peak_percent=75.0,
        temperature_celsius=80.0,
        failed=False,
        steps=steps,
    ) == 3
    assert select_render_workers(
        current=3,
        peak_percent=75.0,
        temperature_celsius=70.0,
        failed=True,
        steps=steps,
    ) == 2


def test_context_from_config_builds_paths_under_trusted_roots(config):
    context = ShardContext.from_config(
        config, "ObjaverseXL_github", "ObjaverseXL_github-00000", "batch000"
    )
    local = (
        config.paths.local_root
        / "preprocess"
        / "active"
        / "ObjaverseXL_github-00000"
        / "batch000"
    )

    assert context == ShardContext(
        source="ObjaverseXL_github",
        shard_id="ObjaverseXL_github-00000",
        instances=config.paths.data2_root
        / "control/shards/ObjaverseXL_github/ObjaverseXL_github-00000/batch000.txt",
        metadata_root=config.paths.data2_root
        / "control/metadata/ObjaverseXL_github",
        source_root=config.paths.data2_root / "raw/ObjaverseXL_github",
        download_root=local / "source",
        work_root=local / "work",
        output_root=local / "output",
        batch_id="batch000",
        record_prefix="ObjaverseXL_github-00000_batch000_",
    )
    assert context.instances.is_relative_to(config.paths.data2_root / "control")
    assert context.metadata_root.is_relative_to(config.paths.data2_root / "control")
    assert context.source_root.is_relative_to(config.paths.data2_root / "raw")
    assert context.download_root.is_relative_to(config.paths.local_root)
    assert context.work_root.is_relative_to(config.paths.local_root)
    assert context.output_root.is_relative_to(config.paths.local_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", ""),
        ("source", "../ABO"),
        ("source", "/absolute"),
        ("source", "ABO\\escape"),
        ("shard_id", "."),
        ("shard_id", ".."),
        ("shard_id", "ABO/00000"),
        ("batch_id", "../batch000"),
        ("batch_id", "/tmp/batch000"),
        ("batch_id", "batch000\\escape"),
    ],
)
def test_context_from_config_rejects_unsafe_identifiers(config, field, value):
    values = {
        "source": "ABO",
        "shard_id": "ABO-00000",
        "batch_id": "batch000",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ShardContext.from_config(config, **values)


def test_context_and_dag_are_frozen_and_deterministic(config, tmp_path):
    context = ShardContext.for_test(tmp_path, "ABO", "ABO-00000")
    first = build_preprocessing_dag(context, config)
    second = build_preprocessing_dag(context, config)

    assert isinstance(first, tuple)
    assert first == second
    assert hash(context)
    assert hash(first)
    with pytest.raises(FrozenInstanceError):
        context.source = "HSSD"
    with pytest.raises(FrozenInstanceError):
        first[0].name = "changed"
