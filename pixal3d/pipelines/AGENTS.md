# IMAGE-TO-3D PIPELINES

## OVERVIEW

Inference orchestration for single-view, posed multi-view, sampler, and image-background-removal paths.

## STRUCTURE

```text
base.py -> pixal3d_image_to_3d.py -> pixal3d_mv_image_to_3d.py
                         \-> samplers/ -> flow matching and guidance
                         \-> rembg/    -> optional image matting
```

## WHERE TO LOOK

- `pixal3d_image_to_3d.py`: four-stage structure/shape/texture/PBR cascade and latent decoding.
- `pixal3d_mv_image_to_3d.py`: posed-view validation, projection-condition fusion, and single-view reuse.
- `samplers/`: flow Euler and classifier-free/guidance-interval behavior.
- `base.py`: model loading, device movement, and pretrained pipeline config handling.
- Root `inference.py`, `inference_mv.py`, and `app.py`: public runtime entrypoints.

## CONVENTIONS

- Pipeline package exports are lazy; preserve the `__init__.py` mapping and avoid eager heavyweight imports.
- Multi-view inputs already carry calibrated transforms. The multi-view path does not run MoGe camera estimation.
- Stage model names, latent keys, projection grid resolution, and `pbr_attr_layout` are cross-stage contracts.
- Low-VRAM execution explicitly moves models between CPU and GPU; retain the documented load/offload order.

## ANTI-PATTERNS

- Do not pass `transform_matrix` into `ProjGrid` when the latent geometry has already been front-view transformed.
- Do not silently change sampler step semantics, guidance intervals, or latent keys used by saved pipeline configs.
- Do not load MoGe while rembg still occupies VRAM.
