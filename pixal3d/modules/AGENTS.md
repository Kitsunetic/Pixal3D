# MODEL MODULES

## OVERVIEW

Dense and sparse attention, transformer, normalization, spatial, feature-extraction, and backend-dispatch primitives.

## STRUCTURE

```text
modules/{attention,transformer,spatial,norm.py,utils.py}
modules/sparse/ -> sparse/AGENTS.md for coordinate-aware implementations
```

## WHERE TO LOOK

- `attention/`: dense attention, rotary embeddings, projection attention, and backend selection.
- `transformer/`: dense transformer blocks and modulation.
- `sparse/`: sparse tensor, convolution, attention, spatial, and transformer implementations; see its child guide.
- `image_feature_extractor.py`: DINO feature extraction used by projection conditioning.
- `config.py` files: environment-controlled backend/debug switches.

## CONVENTIONS

- Backend vocabulary is intentionally fixed. Attention and sparse convolution implementations keep optional native dependencies behind dispatch boundaries.
- Dense and sparse blocks share conceptual APIs but are not interchangeable: preserve tensor layout, coordinate alignment, dtype, and device semantics.
- Package exports are lazy in sparse and related packages; avoid eager imports of every backend.

## ANTI-PATTERNS

- Do not make FlashAttention, xFormers, spconv, torchsparse, or FlexGEMM mandatory merely by importing a module.
- Do not silently change backend defaults or accept undocumented environment values.
- Do not reorder/deduplicate sparse coordinates or alter feature alignment as an optimization shortcut.
