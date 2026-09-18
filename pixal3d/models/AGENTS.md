# MODEL IMPLEMENTATIONS

## OVERVIEW

Flow-matching and sparse VAE model components used by the image-to-3D cascade and training entrypoint.

## WHERE TO LOOK

- `sparse_structure_flow.py`: structure-stage flow model and timestep embedding.
- `structured_latent_flow.py`: SLat flow models, including elastic variants.
- `sparse_structure_vae.py`, `sc_vaes/`: sparse structure, FlexiDualGrid, and sparse U-Net VAE encoders/decoders.
- `__init__.py`: lazy name-to-module export map and `from_pretrained()` loader.
- `train.py`, `configs/gen/`: dynamic model selection and serialized constructor arguments.

## CONVENTIONS

- Public classes are resolved lazily; update the export mapping when adding a model.
- Model forward signatures are runtime contracts with pipelines, trainers, and checkpoint configs. Preserve argument names, conditioning shapes, and sparse coordinate alignment.
- Elastic models are opt-in variants; keep memory/controller behavior separate from the base model path.
- Optional sparse/attention dependencies must remain lazy so importing the package does not require every backend.

## ANTI-PATTERNS

- Do not add convenience kwargs to `SparseStructureFlowModel.forward()` unless every caller and checkpoint path is updated together.
- Do not change latent channel layouts or spatial resolutions without updating the matching configs, VAE, pipeline, and renderer contracts.
- Do not make model import eagerly initialize CUDA or optional native backends.
