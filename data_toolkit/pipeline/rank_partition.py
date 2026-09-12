"""Deterministic rank partitions that balance ordered producer streams."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override


@dataclass(frozen=True, slots=True)
class RankPartitionError(ValueError):
    field: str
    value: int

    @override
    def __str__(self) -> str:
        return f"invalid rank partition {self.field}: {self.value}"


def interleaved_rank_indices(
    item_count: int,
    rank: int,
    world_size: int,
) -> range:
    if type(item_count) is not int or item_count < 0:
        raise RankPartitionError("item_count", item_count)
    if type(world_size) is not int or world_size <= 0:
        raise RankPartitionError("world_size", world_size)
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise RankPartitionError("rank", rank)
    return range(rank, item_count, world_size)
