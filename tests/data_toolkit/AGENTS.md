# DATA TOOLKIT TESTS

## OVERVIEW

Pytest contract suite for the preprocessing pipeline, emphasizing fail-closed artifacts, recovery, resource policy, and worker semantics.

## WHERE TO LOOK

| Behavior | Tests |
|----------|-------|
| CLI/gates/exit codes | `test_cli.py` |
| Orchestration/recovery | `test_orchestrator.py`, `test_pipeline_integration.py` |
| Stage DAG/scheduling | `test_commands.py`, `test_scheduler.py`, `test_fast_command_graph.py` |
| Queue/worker lifecycle | `test_work_queue.py`, `test_production_worker.py`, `test_worker_registry.py`, `test_worker_runtime.py`, `test_worker_supervisor.py` |
| Artifact safety/publication | `test_validation.py`, `test_packing.py`, `test_atomic_io.py`, `test_leaf_worker_contracts.py` |
| Evidence/reporting/resources | `test_evidence.py`, `test_reporting.py`, `test_resources.py` |
| Shared fixtures | `conftest.py`, `fixtures/fake_leaf_worker.py` |

## CONVENTIONS

- Run with `pytest`; discovery is rooted at `tests` by `pytest.ini`. Use `-m integration` and `-m gpu` explicitly for external/GPU cases.
- Prefer `tmp_path`, synthetic metadata, monkeypatching, fake subprocesses, and parametrized edge cases. Shared fixtures are `config`, `tmp_config`, and `synthetic_config`.
- Test names describe behavior (`rejects`, `preserves`, `is_read_only`, `uses`) and mirror production module names.
- Update these tests with changes to config hashes, frozen membership, leases, retries, validation, atomic publication, or report schemas.

## ANTI-PATTERNS

- Do not turn an integration/GPU test into a default local test by removing its marker or CUDA guard.
- Do not assert only stale metadata when testing resume; validate the artifact/checkpoint truth used by the pipeline.
- Do not weaken tests for path traversal, symlinks, checksums, atomic writes, resource stops, or quarantine classification; these are production safety contracts.
