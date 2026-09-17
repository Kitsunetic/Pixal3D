from dataclasses import replace

from data_toolkit.pipeline.commands import ShardContext, build_preprocessing_dag
from data_toolkit.pipeline.runtime_profile import parallel_pipeline_runtime


def test_gpu5_profile_uses_two_renderers_and_sixteen_loaders(
    config, tmp_path, monkeypatch
):
    # Given: the single-GPU production CPU budget and runtime-only overrides.
    config = replace(
        config,
        workers=replace(config.workers, cpu_threads=16),
    )
    config_hash = config.config_hash()
    monkeypatch.setenv("PIXAL3D_RENDERER_MODE", "native")
    monkeypatch.setenv("PIXAL3D_NATIVE_RENDER_WORKERS", "2")
    monkeypatch.setenv("PIXAL3D_ENCODER_LOADER_WORKERS", "16")
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")
    context = ShardContext.for_test(
        tmp_path, "ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"
    )

    # When: the production DAG is generated.
    dag = build_preprocessing_dag(context, config)

    # Then: renderer and loader concurrency change without changing identity.
    render = next(command for command in dag if command.name == "prepare_bundle")
    geometry = next(
        command for command in dag if command.name == "geometry_encode_bundle"
    )
    assert render.argv[render.argv.index("--render_workers") + 1] == "2"
    assert render.argv[render.argv.index("--render_workers_per_gpu") + 1] == "2"
    assert geometry.argv[geometry.argv.index("--loader_workers") + 1] == "16"
    assert config.config_hash() == config_hash


def test_overlap32_profile_splits_prepare_bundle_without_changing_identity(
    config, tmp_path, monkeypatch
):
    config_hash = config.config_hash()
    monkeypatch.setenv("PIXAL3D_PIPELINE_PROFILE", "overlap32")
    monkeypatch.setenv("PIXAL3D_GPU_INDICES", "0")
    context = ShardContext.for_test(
        tmp_path, "ObjaverseXL_sketchfab", "ObjaverseXL_sketchfab-00000"
    )

    commands = {
        command.name: command
        for command in build_preprocessing_dag(context, config)
    }

    assert commands["prepare_bundle"].argv[-2:] == ("--phase", "dump")
    assert commands["render_bundle"].argv[-2:] == ("--phase", "render")
    dump_argv = commands["prepare_bundle"].argv
    assert dump_argv[dump_argv.index("--dump_workers") + 1] == "18"
    encode_argv = commands["geometry_encode_bundle"].argv
    assert encode_argv[encode_argv.index("--max_workers") + 1] == "18"
    assert config.config_hash() == config_hash


def test_overlap32_profile_keeps_fixed_worker_counts_without_cpu_admission(
    monkeypatch,
):
    monkeypatch.setenv("PIXAL3D_PIPELINE_PROFILE", "overlap32")

    runtime = parallel_pipeline_runtime(
        cpu_limit=1,
        configured_chunk_assets=64,
    )

    assert runtime.chunk_assets == 32
    assert runtime.prepare_cpu_cores == 18
    assert runtime.render_cpu_cores == 8
    assert runtime.encode_cpu_cores == 18
    assert runtime.geometry_workers == 18
    assert runtime.prepare_cpu_cores + runtime.encode_cpu_cores == 36
    assert 2 * runtime.prepare_cpu_cores == 36
    assert runtime.split_prepare_bundle is True
