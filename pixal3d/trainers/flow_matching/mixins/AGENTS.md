# FLOW-MATCHING CONDITIONING MIXINS

## OVERVIEW

Composable conditioning layers for classifier-free guidance, text, image, and projected DINOv3 features, including multi-view projection features.

## WHERE TO LOOK

- `image_conditioned_proj.py`: `ProjGrid`, DINOv3 projection extraction, single/multi-view feature conditioning, and projection buffers.
- `image_conditioned.py`, `text_conditioned.py`, `classifier_free_guidance.py`: orthogonal conditioning behaviors.
- `../flow_matching.py` and `../sparse_flow_matching.py`: base objective, sparse handling, and trainer composition.
- Related tests/configs: `configs/gen/` and `tests/data_toolkit/` where conditioning is exercised indirectly through preprocessing/integration paths.

## CONVENTIONS

- Method-resolution order is intentional: mixins provide narrow conditioning hooks while base trainers own optimization and checkpointing.
- Projection grids and feature tensors are shape/device/dtype-sensitive; preserve geometry transforms and multi-view fusion semantics.
- `ProjGrid` contains buffers only; it is not a trainable model entry.

## ANTI-PATTERNS

- Do not register `ProjGrid` in `self.models`.
- Do not pass a transformed camera matrix into a projection path that already front-view transforms latent geometry.
- Do not add extra kwargs to `SparseStructureFlowModel.forward()` as a shortcut for conditioning.
