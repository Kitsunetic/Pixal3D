# DATASET ADAPTERS

## OVERVIEW

Source-specific adapters for ABO, HSSD, ObjaverseXL, 3D-FUTURE, TexVerse, and Toys4K.

## WHERE TO LOOK

- `__init__.py` and `pipeline/dataset_adapter.py` define adapter discovery and integration.
- Each source module owns its download/archive/metadata rules; preserve the external source spelling in filenames and identifiers.
- `tests/data_toolkit/test_dataset_adapters.py` and `test_dataset_package.py` cover archive safety, path traversal, symlinks, digests, and package loading.

## CONVENTIONS

- Asset identity, deduplication, and source ordering are part of production compatibility, not incidental metadata.
- 3D-FUTURE uses the `image.jpg` SHA-256 as canonical identity. Toys4K is evaluation-only.
- HSSD downloads are checksum-verified and bounded by the documented transfer policy.
- Package-relative imports may fall back to top-level imports for standalone script execution; keep both paths working when editing shared helpers.

## ANTI-PATTERNS

- Do not substitute an OBJ hash for the canonical 3D-FUTURE image hash.
- Do not disable checksum validation, accept unsupported material graphs, or classify provider/infrastructure failures as asset quarantine.
