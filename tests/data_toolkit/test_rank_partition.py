from importlib import import_module

import pytest


def test_interleaved_rank_indices_balance_producer_order():
    # Given
    partition = import_module(
        "data_toolkit.pipeline.rank_partition"
    ).interleaved_rank_indices

    # When
    ranks = tuple(tuple(partition(10, rank, 4)) for rank in range(4))

    # Then
    assert ranks == ((0, 4, 8), (1, 5, 9), (2, 6), (3, 7))


def test_interleaved_rank_indices_cover_each_task_once():
    # Given
    partition = import_module(
        "data_toolkit.pipeline.rank_partition"
    ).interleaved_rank_indices

    # When
    assigned = [
        index
        for rank in range(4)
        for index in partition(17, rank, 4)
    ]

    # Then
    assert sorted(assigned) == list(range(17))


@pytest.mark.parametrize(
    ("item_count", "rank", "world_size"),
    ((-1, 0, 1), (1, -1, 1), (1, 0, 0), (1, 1, 1)),
)
def test_interleaved_rank_indices_reject_invalid_partition(
    item_count,
    rank,
    world_size,
):
    # Given
    module = import_module("data_toolkit.pipeline.rank_partition")

    # When / Then
    with pytest.raises(module.RankPartitionError):
        module.interleaved_rank_indices(item_count, rank, world_size)
