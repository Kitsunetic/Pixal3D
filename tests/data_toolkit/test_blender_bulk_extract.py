from __future__ import annotations

import numpy as np
import pytest

from data_toolkit.blender_script.bulk_extract import (
    foreach_get_array,
    material_ids_for_polygons,
    triangle_loop_values,
)


class FakeForeachGetCollection:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.requests: list[str] = []

    def foreach_get(self, property_name: str, destination: np.ndarray) -> None:
        self.requests.append(property_name)
        np.copyto(destination, self.values[property_name].reshape(-1))


def test_foreach_get_array_returns_preallocated_bulk_values() -> None:
    # Given
    collection = FakeForeachGetCollection(
        {"co": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)}
    )

    # When
    values = foreach_get_array(
        collection, property_name="co", item_count=2, item_width=3, dtype=np.float32
    )

    # Then
    assert collection.requests == ["co"]
    np.testing.assert_array_equal(
        values, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    )
    assert values.dtype == np.float32


def test_triangle_loop_values_gathers_each_triangulated_polygon() -> None:
    # Given
    loop_values = np.array([10, 11, 12, 99, 20, 21, 22], dtype=np.int32)
    loop_starts = np.array([0, 4], dtype=np.int32)
    loop_totals = np.array([3, 3], dtype=np.int32)

    # When
    triangles = triangle_loop_values(loop_values, loop_starts, loop_totals)

    # Then
    np.testing.assert_array_equal(
        triangles, np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    )


def test_triangle_loop_values_rejects_non_triangular_polygons() -> None:
    # Given
    loop_values = np.array([10, 11, 12, 13], dtype=np.int32)
    loop_starts = np.array([0], dtype=np.int32)
    loop_totals = np.array([4], dtype=np.int32)

    # When / Then
    with pytest.raises(ValueError):
        triangle_loop_values(loop_values, loop_starts, loop_totals)


def test_material_ids_for_polygons_preserves_missing_slots_and_global_ids() -> None:
    # Given
    polygon_material_indices = np.array([1, 0, 1], dtype=np.int32)
    slot_material_ids = np.array([17, -1], dtype=np.int32)

    # When
    material_ids = material_ids_for_polygons(
        polygon_material_indices, slot_material_ids
    )

    # Then
    np.testing.assert_array_equal(material_ids, np.array([-1, 17, -1], dtype=np.int32))


def test_material_ids_for_polygons_marks_polygons_missing_when_object_has_no_slots() -> None:
    # Given
    polygon_material_indices = np.array([0, 1], dtype=np.int32)
    slot_material_ids = np.empty(0, dtype=np.int32)

    # When
    material_ids = material_ids_for_polygons(
        polygon_material_indices, slot_material_ids
    )

    # Then
    np.testing.assert_array_equal(material_ids, np.array([-1, -1], dtype=np.int32))
