# PIXAL3D CORE PACKAGE

## OVERVIEW

PyTorch model/runtime package for the image-to-3D latent cascade: image conditions feed sparse structure, shape SLat, and texture/PBR stages before mesh/voxel rendering.

## STRUCTURE

```text
datasets/ -> models/ -> pipelines/ -> representations/renderers
                  \-> trainers/ -> modules/{sparse,attention,transformer}
utils/ provides distributed, data, rendering, loss, and sampling helpers.
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Single/multi-view cascade | `pipelines/pixal3d_image_to_3d.py`, `pixal3d_mv_image_to_3d.py` |
| Flow/decoder models | `models/`, `models/sc_vaes/` |
| Sparse and attention primitives | `modules/sparse/`, `modules/attention/`, `modules/transformer/` |
| Training lifecycle | `trainers/`, especially `trainers/basic.py` |
| Latent/mesh/voxel data contracts | `representations/`, `datasets/` |
| Rendering and GLB-facing output | `renderers/`, `utils/render_utils.py` |
| Dynamic public exports | package `__init__.py` files |

## CONVENTIONS

- Public model, dataset, and pipeline names are dynamically/lazily resolved through package exports. Register new public symbols in the relevant `__init__.py` mapping.
- `SparseTensor` preserves coordinate-aligned features through the cascade. Keep sparse coordinate bounds, ordering, and feature shapes compatible with decoders and renderers.
- `train.py` selects dataset, model, and trainer classes from JSON config names; configs in `configs/gen/` are part of the runtime contract.
- Shared utilities are high-centrality: `utils/data_utils.py` affects sampler/resume ordering; `dist_utils.py` affects process-group setup; `render_utils.py` imports renderer/representation layers.
- The model cascade is structure -> shape -> texture/PBR -> `MeshWithVoxel` -> renderer. Multi-view inference overrides conditioning and reuses the single-view cascade.

## ANTI-PATTERNS

- Do not pass `transform_matrix` to `ProjGrid`; latent geometry is already front-view transformed.
- Do not pass extra kwargs to `SparseStructureFlowModel.forward()`.
- Do not register buffer-only `ProjGrid` as a trainable model component.
- Preserve the rembg/MoGe VRAM load order in inference and the fixed `pbr_attr_layout` channel meanings.
