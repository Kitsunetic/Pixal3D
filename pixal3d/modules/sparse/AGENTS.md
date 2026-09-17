# SPARSE MODULES

## OVERVIEW

Coordinate-aware sparse tensor primitives and backend-specific attention, convolution, spatial, and transformer implementations.

## WHERE TO LOOK

- `basic.py`: `SparseTensor`, coordinate/feature layout, indexing, and sparse transforms.
- `attention/`: full/windowed/projected attention and rotary embeddings.
- `conv/`: common convolution API plus `flex_gemm`, `spconv`, and `torchsparse` backends.
- `spatial/`: interpolation, subdivision, and space/channel conversion.
- `transformer/`: sparse blocks and modulation layers.

## CONVENTIONS

- `__init__.py` uses a deliberate lazy export map (`__attributes`, `__submodules`, `__getattr__`). Add public symbols there rather than importing every backend eagerly.
- Backend implementations are not interchangeable: preserve their configuration and optional-dependency boundaries.
- Coordinate alignment, batch indices, feature shape, device, and dtype are part of every sparse tensor contract; test transformations at their boundary.

## ANTI-PATTERNS

- Do not silently reorder or deduplicate coordinates when a downstream decoder expects stable alignment.
- Do not make an optional sparse backend a package-import requirement.
- Be cautious with existing broad exception handlers in `basic.py`; preserve failure visibility when changing them.
