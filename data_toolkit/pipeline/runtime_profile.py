"""Runtime-only scheduler profiles that do not change dataset identity."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class ParallelPipelineRuntime:
    chunk_assets: int
    prepare_cpu_cores: int
    render_cpu_cores: int
    encode_cpu_cores: int
    geometry_workers: int
    split_prepare_bundle: bool


def parallel_pipeline_runtime(
    *,
    cpu_limit: int,
    configured_chunk_assets: int,
) -> ParallelPipelineRuntime:
    """Resolve an opt-in execution profile without touching the config hash."""
    name = os.environ.get("PIXAL3D_PIPELINE_PROFILE", "default")
    if name == "default":
        return ParallelPipelineRuntime(
            chunk_assets=configured_chunk_assets,
            prepare_cpu_cores=cpu_limit,
            render_cpu_cores=cpu_limit,
            encode_cpu_cores=cpu_limit,
            geometry_workers=cpu_limit,
            split_prepare_bundle=False,
        )
    if name != "overlap32":
        raise ValueError(
            "PIXAL3D_PIPELINE_PROFILE must be default or overlap32"
        )
    return ParallelPipelineRuntime(
        chunk_assets=32,
        prepare_cpu_cores=4,
        render_cpu_cores=8,
        encode_cpu_cores=18,
        geometry_workers=18,
        split_prepare_bundle=True,
    )
