from dataclasses import replace

from data_toolkit.pipeline.commands import ShardContext, build_preprocessing_dag


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
