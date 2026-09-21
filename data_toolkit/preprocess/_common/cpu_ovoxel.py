"""Load only the CPU voxel modules needed by preprocessing.

``o_voxel`` eagerly imports its GLB post-processing surface, which imports
``flex_gemm`` and initializes CUDA even though voxelization only needs
``convert`` and ``io``.  That prevents a CPU-only 04 worker from running on a
node without an available GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types


def import_cpu_ovoxel():
    """Return an ``o_voxel`` package exposing only convert and io.

    Constructing the package module directly avoids executing the upstream
    eager ``o_voxel.__init__`` while preserving normal relative imports in its
    CPU conversion and IO submodules.
    """
    existing = sys.modules.get("o_voxel")
    if existing is not None:
        return existing

    spec = importlib.util.find_spec("o_voxel")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("o_voxel is required for 04_voxelize")

    package = types.ModuleType("o_voxel")
    package.__file__ = spec.origin
    package.__package__ = "o_voxel"
    package.__path__ = list(spec.submodule_search_locations)
    package.__spec__ = spec
    sys.modules["o_voxel"] = package
    try:
        package.convert = importlib.import_module("o_voxel.convert")
        package.io = importlib.import_module("o_voxel.io")
    except BaseException:
        sys.modules.pop("o_voxel", None)
        raise
    return package
