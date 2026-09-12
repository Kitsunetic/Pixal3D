"""Pure control-flow helpers shared by Blender boundary fitting and tests."""

from __future__ import annotations


def target_requires_legacy_replay(
    fit_exhausted: bool,
    touches_boundary: bool = False,
    too_far: bool = False,
) -> bool:
    """Return whether target-resolution fitting must replay the legacy loop."""
    return fit_exhausted or touches_boundary or too_far


def legacy_replay_state(
    original_radius: float,
    max_retry: int,
) -> tuple[float, int, int]:
    """Reset radius and retry accounting to the full legacy retry budget."""
    return original_radius, 0, max_retry
