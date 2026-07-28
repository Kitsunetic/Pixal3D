from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .orchestrator import _read_regular_bytes_nofollow


PARALLELISM_POLICY_SCHEMA_VERSION = 1


class ParallelismPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParallelismRuntimePolicy:
    max_chunks_in_flight: int
    render_workers_per_gpu_steps: tuple[int, ...]
    dump_workers_max: int


def _validate(
    policy: ParallelismRuntimePolicy,
    *,
    canonical_max_chunks_in_flight: int,
    canonical_render_workers_per_gpu_steps: tuple[int, ...],
    canonical_dump_workers_max: int,
) -> None:
    chunks = policy.max_chunks_in_flight
    steps = policy.render_workers_per_gpu_steps
    dump_workers_max = policy.dump_workers_max
    if (
        type(chunks) is not int
        or not 0 < chunks <= canonical_max_chunks_in_flight
        or not steps
        or any(type(value) is not int or value <= 0 for value in steps)
        or tuple(sorted(steps)) != steps
        or len(set(steps)) != len(steps)
        or max(steps) > max(canonical_render_workers_per_gpu_steps)
        or type(dump_workers_max) is not int
        or not 0 < dump_workers_max <= canonical_dump_workers_max
    ):
        raise ParallelismPolicyError(
            "node parallelism policy may only reduce canonical positive "
            "chunk, render-worker, and dump-worker limits"
        )


def read_parallelism_runtime_policy(
    path: Path,
    *,
    canonical_max_chunks_in_flight: int,
    canonical_render_workers_per_gpu_steps: tuple[int, ...],
    canonical_dump_workers_max: int,
) -> ParallelismRuntimePolicy:
    fallback = ParallelismRuntimePolicy(
        canonical_max_chunks_in_flight,
        canonical_render_workers_per_gpu_steps,
        canonical_dump_workers_max,
    )
    payload = _read_regular_bytes_nofollow(Path(path), missing_ok=True)
    if payload is None:
        _validate(
            fallback,
            canonical_max_chunks_in_flight=canonical_max_chunks_in_flight,
            canonical_render_workers_per_gpu_steps=(
                canonical_render_workers_per_gpu_steps
            ),
            canonical_dump_workers_max=canonical_dump_workers_max,
        )
        return fallback
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "max_chunks_in_flight",
            "render_workers_per_gpu_steps",
            "dump_workers_max",
        }:
            raise ValueError("unexpected fields")
        if value["schema_version"] != PARALLELISM_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        raw_steps = value["render_workers_per_gpu_steps"]
        if not isinstance(raw_steps, list):
            raise ValueError("render worker steps must be a list")
        policy = ParallelismRuntimePolicy(
            value["max_chunks_in_flight"],
            tuple(raw_steps),
            value["dump_workers_max"],
        )
        _validate(
            policy,
            canonical_max_chunks_in_flight=canonical_max_chunks_in_flight,
            canonical_render_workers_per_gpu_steps=(
                canonical_render_workers_per_gpu_steps
            ),
            canonical_dump_workers_max=canonical_dump_workers_max,
        )
        return policy
    except (
        ParallelismPolicyError,
        TypeError,
        ValueError,
        UnicodeError,
    ) as error:
        raise ParallelismPolicyError(
            f"invalid node parallelism policy: {error}"
        ) from error
