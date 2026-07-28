import json

import pytest

from data_toolkit.pipeline.parallelism_policy import (
    ParallelismPolicyError,
    ParallelismRuntimePolicy,
    read_parallelism_runtime_policy,
)


def test_missing_parallelism_policy_uses_canonical_values(tmp_path):
    assert read_parallelism_runtime_policy(
        tmp_path / "parallelism_policy.json",
        canonical_max_chunks_in_flight=3,
        canonical_render_workers_per_gpu_steps=(2, 3, 4),
        canonical_dump_workers_max=44,
    ) == ParallelismRuntimePolicy(3, (2, 3, 4), 44)


def test_parallelism_policy_reads_node_local_overrides(tmp_path):
    path = tmp_path / "parallelism_policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_chunks_in_flight": 1,
                "render_workers_per_gpu_steps": [1, 2, 3],
                "dump_workers_max": 16,
            }
        )
    )

    assert read_parallelism_runtime_policy(
        path,
        canonical_max_chunks_in_flight=3,
        canonical_render_workers_per_gpu_steps=(2, 3, 4),
        canonical_dump_workers_max=44,
    ) == ParallelismRuntimePolicy(1, (1, 2, 3), 16)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "max_chunks_in_flight": 0,
         "render_workers_per_gpu_steps": [1, 2, 3],
         "dump_workers_max": 16},
        {"schema_version": 1, "max_chunks_in_flight": 1,
         "render_workers_per_gpu_steps": [2, 1, 3],
         "dump_workers_max": 16},
        {"schema_version": 1, "max_chunks_in_flight": 1,
         "render_workers_per_gpu_steps": [1, 2, 8],
         "dump_workers_max": 16},
        {"schema_version": 1, "max_chunks_in_flight": 1,
         "render_workers_per_gpu_steps": [1, 2],
         "dump_workers_max": 0},
        {"schema_version": 1, "max_chunks_in_flight": 1,
         "render_workers_per_gpu_steps": [1, 2],
         "dump_workers_max": 45},
    ],
)
def test_invalid_parallelism_policy_fails_closed(tmp_path, payload):
    path = tmp_path / "parallelism_policy.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ParallelismPolicyError):
        read_parallelism_runtime_policy(
            path,
            canonical_max_chunks_in_flight=3,
            canonical_render_workers_per_gpu_steps=(2, 3, 4),
            canonical_dump_workers_max=44,
        )
