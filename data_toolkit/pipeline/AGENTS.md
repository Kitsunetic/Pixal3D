# PRODUCTION PREPROCESSING PIPELINE

## OVERVIEW

This package is the stateful production system: CLI dispatch, frozen staged execution, worker leases, resource policy, validation, packing, evidence, and reporting.

## STRUCTURE

```text
cli.py -> runtime.py -> orchestrator.py -> scheduler.py -> commands.py
                                      -> workers/queue
                                      -> validation.py -> packing/evidence/reporting
```

## WHERE TO LOOK

| Concern | Files |
|---------|-------|
| CLI and exit codes | `cli.py` |
| Service composition/read-only vs mutating paths | `runtime.py` |
| Shard lifecycle, checkpointing, recovery | `orchestrator.py`, `scheduler.py` |
| Command DAG and worker profiles | `commands.py`, `full_run.py`, `runtime_profile.py` |
| Queue/worker lifecycle | `work_queue.py`, `worker_registry.py`, `worker_runtime.py`, `production_worker.py`, `worker_supervisor.py` |
| Resource admission | `resources.py`, `parallelism.py`, `parallelism_policy.py`, `gpu_policy.py` |
| Artifact identity and acceptance | `config.py`, `registry.py`, `instance_manifest.py`, `validation.py`, `output_compatibility.py` |
| Publication and audit surfaces | `packing.py`, `evidence.py`, `reporting.py`, `operator_handoff.py` |
| Contract tests | `tests/data_toolkit/test_cli.py`, `test_orchestrator.py`, `test_scheduler.py`, `test_work_queue.py`, `test_pipeline_integration.py` |

## CONVENTIONS

- The stage graph is ordered `prepare -> render -> encode -> finalize`. Asset membership is frozen before work starts; restart validates completed artifacts before advancing.
- `PipelineConfig.config_hash()` is compatibility identity. Queue manifests, leases, checkpoints, runtime reports, evidence, packs, and gate reports must reject mismatches.
- `WorkUnit`/`WorkLease` ownership is explicit: claim, heartbeat, complete, release, handoff, and retry semantics are covered by tests.
- Validators must run before packing or evidence publication. Packs are checksummed and published atomically.
- `plan`, `resume`, and `audit` are read-only with respect to freezing; only `run` may freeze missing batches. Exit codes encode operator/provider, resource, and data-quality stops.

## ANTI-PATTERNS

- Do not change worker counts, stage overlap, retry policy, or resource thresholds without preserving telemetry, admission, checkpoint, and recovery behavior.
- Do not infer resume truth from stale metadata; use validated outputs/checkpoints and the exact producing commit.
- Do not quarantine infrastructure/provider failures as asset failures.
- Never run `queue --action init` against the shared queue.
- Do not use broad recursive deletion against production `raw`, `prepared`, `archive`, `control`, or `preprocess/active` roots.
