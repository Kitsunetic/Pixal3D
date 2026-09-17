# DATA TOOLKIT

## OVERVIEW

Dataset download, rendering, latent encoding, packing, and the production multi-view preprocessing system.

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Production command boundary | `pipeline/cli.py` |
| Pipeline services and orchestration | `pipeline/` |
| Source-specific adapters | `datasets/` |
| Blender-side scripts | `blender_script/` |
| Legacy leaf encoders/renderers | Direct files in this directory |
| Operational contract | `README.md`, `configs/multiview_preprocess.yaml` |

## CONVENTIONS

- Production work uses `python -m data_toolkit.pipeline.cli`; direct top-level scripts are development/legacy paths.
- The YAML config is a closed, typed contract. Its normalized contents are hashed; config hashes travel through queues, checkpoints, reports, evidence, and packs.
- Dataset source names and external identities are preserved. Toys4K is evaluation-only; 3D-FUTURE identity is the `image.jpg` SHA-256, not the OBJ hash.
- Several scripts support both package-relative and top-level fallback imports, so invocation directory and `PYTHONPATH` matter.
- Shared helpers in `utils.py` overlap with `pixal3d/utils`; do not merge or replace one side without checking both import contexts and consumers.

## ANTI-PATTERNS

- Do not collapse `/root/data2/pixal3d`, `/root/data3/pixal3d`, and `/root/node17/data/pixal3d` into one storage role.
- Do not publish unvalidated outputs or delete raw/prepared/control trees broadly. Cleanup follows validation, eight-pack publication, raw archive verification, and reference checks.
- Do not disable checksum verification or silently change source scope, camera policy, dtype, encoder checkpoint, retention, or producing commit.
