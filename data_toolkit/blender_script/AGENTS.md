# BLENDER RUNTIME SCRIPTS

## OVERVIEW

Scripts executed inside Blender for mesh import, camera/lighting setup, rendering, and archive/install support.

## WHERE TO LOOK

- `render_cond.py` owns scene setup, normalization, camera, lighting, and render output.
- `dump_mesh.py` and `dump_pbr.py` are Blender-side extraction boundaries.
- Host orchestration and validation live in `data_toolkit/pipeline/`; communicate through validated artifacts rather than importing Blender-only behavior.

## CONVENTIONS

- The supported Blender version is 4.5.1. Blender 4.x OBJ import uses `bpy.ops.wm.obj_import`.
- Output paths, frame names, image formats, camera matrices, and alpha behavior are validated by pipeline validators and contract tests.
- Keep Blender dependencies isolated from host-side Python imports; installation/archive helpers are separate from pipeline scheduling.

## ANTI-PATTERNS

- Do not restore the pre-Blender-4 OBJ importer.
- Do not publish partial render directories or bypass host-side validation and atomic publication.
