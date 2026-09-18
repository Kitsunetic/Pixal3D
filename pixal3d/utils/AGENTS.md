# RUNTIME UTILITIES

## OVERVIEW

Shared distributed, data-ordering, rendering, mesh, loss, precision, and visualization helpers used across the model stack.

## WHERE TO LOOK

- `data_utils.py`: resumable/balanced samplers, recursive device transfer, and ordering-sensitive data helpers.
- `dist_utils.py`, `elastic_utils.py`: process groups, rank coordination, and elastic memory behavior.
- `render_utils.py`, `mesh_utils.py`: renderer construction, camera conversion, frame/video output, and PLY IO.
- `general_utils.py`, `random_utils.py`: nested mappings, image helpers, deterministic sampling, and low-level shared behavior.
- `loss_utils.py`, `grad_clip_utils.py`: training losses and adaptive gradient clipping.

## CONVENTIONS

- Utility behavior is high-centrality despite small files; preserve device, dtype, rank, and determinism semantics.
- `ResumableSampler` state is part of checkpoint/resume compatibility, not an incidental iterator detail.
- Rendering helpers bridge representations and external renderers; keep camera conventions aligned with pipeline and dataset code.
- Import optional visualization/rendering dependencies lazily where existing modules do so.

## ANTI-PATTERNS

- Do not change sampler ordering, rank partitioning, or checkpoint serialization without updating resume behavior and tests.
- Do not introduce implicit CUDA moves into general helpers.
- Do not alter camera transforms or mesh attribute meanings as a local rendering cleanup.
