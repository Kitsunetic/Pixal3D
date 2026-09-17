"""NumPy helpers for copying Blender RNA collections without Python item loops."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ForeachGetCollection(Protocol):
    """The Blender RNA collection surface required for bulk extraction."""

    def foreach_get(self, property_name: str, destination: np.ndarray) -> None:
        """Copy a property from every collection item into a flat destination."""


def foreach_get_array(
    collection: ForeachGetCollection,
    *,
    property_name: str,
    item_count: int,
    item_width: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Return a preallocated NumPy array filled through RNA ``foreach_get``."""
    shape = (item_count,) if item_width == 1 else (item_count, item_width)
    values = np.empty(shape, dtype=dtype)
    collection.foreach_get(property_name, values.reshape(-1))
    return values


def triangle_loop_values(
    loop_values: np.ndarray,
    polygon_loop_starts: np.ndarray,
    polygon_loop_totals: np.ndarray,
) -> np.ndarray:
    """Gather loop values into ``(face_count, 3, ...)`` triangulated face arrays."""
    if not np.all(polygon_loop_totals == 3):
        raise ValueError("triangulated mesh polygons must have exactly three loops")
    offsets = polygon_loop_starts[:, np.newaxis] + np.arange(3, dtype=np.intp)
    return loop_values[offsets]


def material_ids_for_polygons(
    polygon_material_indices: np.ndarray, slot_material_ids: np.ndarray
) -> np.ndarray:
    """Map polygon material slots to global material IDs, or ``-1`` without slots."""
    if slot_material_ids.size == 0:
        return np.full(polygon_material_indices.shape, -1, dtype=np.int32)
    return slot_material_ids[polygon_material_indices]
