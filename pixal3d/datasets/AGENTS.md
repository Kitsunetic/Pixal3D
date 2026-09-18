# MODEL DATASETS

## OVERVIEW

Training-time dataset and conditioning definitions for sparse structure, structured latent, voxel/PBR, and multi-view samples.

## WHERE TO LOOK

- `components.py`: shared dataset bases plus image/view/multi-image conditioning mixins.
- `sparse_structure_latent.py`, `structured_latent*.py`: latent and image-conditioned sample contracts.
- `sparse_voxel_pbr.py`, `flexi_dual_grid.py`: voxel/PBR and FlexiDualGrid data paths.
- `__init__.py`: lazy dataset discovery used by `train.py` JSON names.

## CONVENTIONS

- Dataset class names are selected dynamically from training JSON; preserve public export names and constructor compatibility.
- Sparse coordinates, batch indices, view ordering, camera transforms, latent dtype, and resolution are model/pipeline contracts.
- Image-conditioned and view-conditioned mixins compose behavior; keep conditioning assembly separate from base sample loading.
- Multi-view data carries calibrated transforms and camera metadata; preserve front-view conventions expected by projection conditioning.

## ANTI-PATTERNS

- Do not reorder views or sparse coordinates without checking projection fusion, sampler ordering, and decoder expectations.
- Do not pass already-transformed camera matrices through a second geometry transform.
- Do not silently change latent dtype/resolution or constructor fields used by serialized configs.
