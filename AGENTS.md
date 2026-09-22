# PROJECT KNOWLEDGE BASE

**Generated:** 2026-09-18T00:00:00Z
**Commit:** 9d99a62
**Branch:** master

## OVERVIEW

Pixal3D is a Python/PyTorch system for image-to-3D generation, distributed training, and multi-view data preprocessing. The repository has two major runtime surfaces: the model stack under `pixal3d/` and the production preprocessing system under `data_toolkit/pipeline/`.

## STRUCTURE

```text
.
├── inference.py, inference_mv.py  single- and multi-view GLB inference
├── app.py                         Gradio web demo
├── train.py                       distributed training launcher
├── pixal3d/                       models, datasets, pipelines, renderers, trainers
│   └── modules/                   dense and sparse tensor/attention primitives
├── data_toolkit/                  dataset preparation and production pipeline
├── configs/gen/                   stage-specific JSON model configs
├── tests/data_toolkit/            pipeline contract and recovery tests
├── docs/                          runbooks, plans, specs, benchmark records
├── docker/, ops/, scripts/        deployment and node operations
└── tools/                         GPU presence utility
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Single-view inference | `inference.py` | MoGe camera estimation plus four-stage pipeline |
| Multi-view inference | `inference_mv.py` | Uses the multi-view condition builder |
| Web demo/API | `app.py`, `index.html` | Gradio server and GLB export |
| Training | `train.py`, `configs/gen/` | Dataset/model/trainer classes are selected dynamically from JSON |
| Production preprocessing | `data_toolkit/pipeline/cli.py` | Run as `python -m data_toolkit.pipeline.cli` |
| Pipeline contracts | `tests/data_toolkit/` | Config, queue, scheduler, worker, validation, packing, and recovery tests |
| Model internals | `pixal3d/models/`, `pixal3d/modules/` | Lazy exports; sparse and dense backends have optional dependencies |
| Rendering/output | `pixal3d/representations/`, `pixal3d/renderers/` | Mesh/voxel/PBR conversion and GLB-facing rendering |
| Operator workflow | `data_toolkit/README.md`, `docs/ops/` | Fixed paths, gates, hardware, and cleanup rules |

## CODE MAP

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `Pixal3DImageTo3DPipeline` | class | `pixal3d/pipelines/pixal3d_image_to_3d.py` | inference/app | Structure → shape → texture/PBR cascade |
| `Pixal3DMVImageTo3DPipeline` | class | `pixal3d/pipelines/pixal3d_mv_image_to_3d.py` | inference_mv | Posed-view condition fusion over single-view cascade |
| `main` | function | `data_toolkit/pipeline/cli.py` | CLI boundary | Preprocessing command dispatch |
| `PipelineConfig.config_hash` | method | `data_toolkit/pipeline/config.py` | queue/report/pack contracts | Configuration identity |
| `ParallelChunkScheduler` | class | `data_toolkit/pipeline/scheduler.py` | orchestrator/tests | Frozen staged execution |
| `SparseTensor` | class | `pixal3d/modules/sparse/basic.py` | models/pipelines/renderers | Coordinate-aligned latent features |
| `BasicTrainer` | class | `pixal3d/trainers/basic.py` | `train.py` and trainer variants | Distributed lifecycle/checkpoints |
| `FlowEulerSampler` | class | `pixal3d/pipelines/samplers/flow_euler.py` | inference pipeline | Flow-matching inference sampler |
| `ProjGrid` | class | `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` | datasets/trainers/pipelines | Buffer-only image projection geometry |

Model-facing LSP tools were unavailable during generation. The code map uses ast-grep outlines, AST-shaped Python searches, and import tracing. Package `__init__.py` files in `pixal3d/models`, `pixal3d/datasets`, `pixal3d/pipelines`, and `pixal3d/renderers` use lazy/dynamic exports, so static imports undercount consumers. Direct core tests are concentrated in sparse preprocessing; the broader model stack has limited direct test coverage.

## CONVENTIONS

- There is no project formatter, linter, type checker, package manifest, Makefile, or CI workflow. Direct script/module invocation is the normal interface.
- For lab-server project work, use `ssh nXjh` for Git, code inspection, tests, and preprocessing. Reserve `ssh nX` and `docker exec` for host/container lifecycle, bind-mount, or host-hardware inspection; do not use them as the normal project-command path.
- Treat `HOST_SERVER` inside a container as an identity verification value only. It must not override the explicitly selected `nXjh` SSH alias or be used to reroute a project command to another server.
- Runtime controls use uppercase environment variables; preprocessing overrides generally use the `PIXAL3D_` prefix. Attention accepts a fixed `ATTN_BACKEND` vocabulary; `sdpa` is the fallback when FlashAttention is unavailable.
- Model configs encode stage, modality, size, resolution, dtype, projection, and fine-tuning state in filenames.
- Dataset adapter filenames intentionally preserve source names (`3D-FUTURE.py`, `ObjaverseXL.py`, etc.) rather than Python identifier conventions.
- Outputs are written to temporary sibling paths, validated, then atomically published where the pipeline owns the artifact.
- Training is launched with `torch.multiprocessing.spawn`, not `torchrun`; dataset/model/trainer classes are selected from JSON names.
- Inference scripts configure CUDA/OpenEXR/attention environment at import time; accepted attention backends are a fixed vocabulary with `sdpa` as the safe fallback.
- Preprocessing runtime profiles may change scheduling/resource allocation without changing dataset identity; config hashes exclude those runtime-only tuning controls.

## ANTI-PATTERNS (THIS PROJECT)

- Do not mix the three preprocessing filesystem roles: `/root/data2/pixal3d` (control/prepared), `/root/data3/pixal3d` (verified raw), and `/root/node17/data/pixal3d` (scratch/training).
- Do not run production preprocessing through legacy leaf scripts; use the pipeline CLI and the documented gate order.
- Do not run `queue --action init` against the shared production queue.
- Do not overwrite frozen historical shard state with current-commit metadata, disable checksum validation, or treat infrastructure failures as asset quarantine.
- Do not overlap GPU-heavy preprocessing and fine-tuning without preserving the resource-admission policy.
- Preserve camera/latent contracts: do not pass `transform_matrix` to `ProjGrid`, pass no extra kwargs to `SparseStructureFlowModel.forward()`, and load MoGe only after rembg is offloaded.
- Do not make optional attention/sparse backends package-import requirements or silently change backend selection.

## COMMANDS

```bash
pytest -m "not integration and not gpu"
pytest tests/data_toolkit
pytest tests/data_toolkit/test_cli.py tests/data_toolkit/test_orchestrator.py
python -m compileall -q data_toolkit
python -m data_toolkit.pipeline.cli --help
python inference.py --help
python inference_mv.py --help
python train.py --help
```

Integration tests may need external binaries/models/data; GPU tests require CUDA. Production commands should run from the repository root in the `pixal3d` environment after access and hardware preflight. There is no package manifest, Makefile, formatter, type checker, or CI workflow; requirements files and direct script/module invocation are authoritative.

## NOTES

Keep generated/runtime trees out of structural guidance: `.omo/`, `.worktrees/`, `tmp/`, `temp/`, `autoresearch-results/`, caches, `__pycache__/`, and `docs/benchmarks/evidence/`. Treat `data_toolkit/README.md` and `docs/ops/` as operational authorities for production sequencing and cleanup.
