# HSSD Throughput Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Node16's counterproductive 72-Blender overlap with a checkpoint-preserving, empirically selected node-local concurrency policy that maximizes completed HSSD assets per hour.

**Architecture:** Extend the existing token-fenced production queue with an explicit operator handoff that archives ownership without consuming a failure attempt. Use the existing node-local parallelism policy to reduce chunk and Blender concurrency without changing the canonical pipeline identity, then compare complete 64-asset chunk timings and retain the faster safe policy.

**Tech Stack:** Python 3.11, pytest, filesystem-backed token-fenced queue, tmux worker supervisors, Blender 4.5.1, CUDA 12.8, PyTorch 2.8+, JSON node-local runtime policies.

## Global Constraints

- Preserve downloaded raw assets, completed render directories, command checkpoints, and published outputs.
- Never allow two nodes to own the same work unit.
- Operator tuning must not consume a queue failure attempt.
- Keep GPU memory target at 80% and hard ceiling at 100%.
- Node16 GPU pool is 0-5 with CPU limit 40; Node17 GPU pool is 0-7 with CPU limit 44.
- Keep source priority `ABO -> 3D-FUTURE -> HSSD -> ObjaverseXL_sketchfab`.
- Terminal assets/hour, not process count or instantaneous GPU utilization, decides the winning policy.
- No OOM, stale lease, terminal queue failure, persistent swap growth, or failed output audit is acceptable.

---

### Task 1: Token-fenced operator handoff

**Files:**
- Modify: `data_toolkit/pipeline/work_queue.py`
- Test: `tests/data_toolkit/test_work_queue.py`

**Interfaces:**
- Consumes: `WorkLease`, `ProductionWorkQueue.assert_owned()`, and the existing lease/history directory layout.
- Produces:
  `ProductionWorkQueue.owned_lease(unit_id: str, *, node_id: str, token: str) -> WorkLease`
  and
  `ProductionWorkQueue.handoff(lease: WorkLease, *, reason: str, now: datetime) -> None`.
- Invariant: the next claim receives the same attempt number as the handed-off lease, while released/failed leases still increment normally.

- [ ] **Step 1: Write the failing queue test**

```python
def test_operator_handoff_preserves_attempt_and_token_fences(tmp_path):
    queue = initialized_queue(tmp_path)
    lease = queue.claim("node16", now=NOW, token="held-token")
    assert lease is not None

    queue.handoff(
        lease,
        reason="operator concurrency reload",
        now=NOW + timedelta(seconds=1),
    )

    replacement = queue.claim(
        "node16",
        now=NOW + timedelta(seconds=2),
        token="replacement-token",
    )
    assert replacement is not None
    assert replacement.unit == lease.unit
    assert replacement.attempt == lease.attempt
    assert (
        tmp_path
        / "history"
        / f"{lease.unit.unit_id}.{lease.token}.handoff"
        / "handoff.json"
    ).is_file()
    with pytest.raises(LeaseLostError):
        queue.handoff(
            lease,
            reason="stale operator",
            now=NOW + timedelta(seconds=3),
        )
```

- [ ] **Step 2: Verify the test fails for the missing API**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_work_queue.py::test_operator_handoff_preserves_attempt_and_token_fences
```

Expected: FAIL because `ProductionWorkQueue` has no `handoff` method.

- [ ] **Step 3: Implement the minimal handoff**

Add a token-fenced method that validates the reason and aware timestamp,
writes `handoff.json` beside `owner.json`, rechecks ownership, and atomically
renames the lease directory to
`history/<unit>.<token>.handoff`. Update `_next_attempt()` so a valid
`.handoff` history directory contributes `attempt - 1`, while `.released`,
`.stale`, and failure histories retain their current attempt accounting.

```python
def owned_lease(self, unit_id, *, node_id, token):
    unit = next(
        (candidate for candidate in self.units()
         if candidate.unit_id == unit_id),
        None,
    )
    if unit is None:
        raise ValueError(f"unknown production work unit: {unit_id}")
    lease = self._read_lease(unit)
    if lease is None or lease.node_id != node_id or lease.token != token:
        raise LeaseLostError(f"work lease was lost: {unit_id}")
    return lease

def handoff(self, lease, *, reason, now):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("handoff reason must be non-empty")
    _aware(now)
    self.assert_owned(lease)
    _write_json(
        self._lease_dir(lease.unit) / "handoff.json",
        {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "unit_id": lease.unit.unit_id,
            "node_id": lease.node_id,
            "token": lease.token,
            "attempt": lease.attempt,
            "reason": reason,
            "handed_off_at": _timestamp(now),
        },
    )
    self.assert_owned(lease)
    self._lease_dir(lease.unit).rename(
        self.history_root
        / f"{lease.unit.unit_id}.{lease.token}.handoff"
    )
```

- [ ] **Step 4: Verify queue behavior**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_work_queue.py
```

Expected: all queue tests pass.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/work_queue.py \
  tests/data_toolkit/test_work_queue.py
git commit -m "feat: add checkpoint-preserving queue handoff"
```

### Task 2: Operator handoff CLI

**Files:**
- Modify: `data_toolkit/pipeline/cli.py`
- Test: `tests/data_toolkit/test_cli.py`

**Interfaces:**
- Consumes: `ProductionWorkQueue.handoff()` and a currently held lease.
- Produces: queue action
  `handoff --unit-id ID --node-id NODE --lease-token TOKEN --reason TEXT`.
- Output: a fresh queue snapshot after the atomic handoff.

- [ ] **Step 1: Write the failing CLI test**

Create a queue with one active lease, invoke:

```python
result = main([
    "queue", "--config", str(tmp_config),
    "--action", "handoff",
    "--unit-id", lease.unit.unit_id,
    "--node-id", lease.node_id,
    "--lease-token", lease.token,
    "--reason", "operator concurrency reload",
])
assert result == 0
assert queue.status(now=NOW)["pending"] == 1
replacement = queue.claim("node16", now=NOW + timedelta(seconds=1))
assert replacement.attempt == lease.attempt
```

Also assert that a wrong token raises `LeaseLostError` and leaves the original
lease owned.

- [ ] **Step 2: Verify the CLI test fails**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_cli.py -k 'queue and handoff'
```

Expected: FAIL because `handoff` is not an accepted queue action.

- [ ] **Step 3: Implement the CLI action**

Add the four required handoff arguments to the queue parser. Resolve the unit
through `queue.owned_lease()`, call `queue.handoff()`, and print
`queue.snapshot()`. Do not construct preprocessing services for this
metadata-only action.

```python
if args.action == "handoff":
    lease = queue.owned_lease(
        args.unit_id,
        node_id=args.node_id,
        token=args.lease_token,
    )
    queue.handoff(
        lease,
        reason=args.reason,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(queue.snapshot(now=datetime.now(timezone.utc))))
    return SUCCESS
```

- [ ] **Step 4: Verify CLI and queue regressions**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_work_queue.py \
  tests/data_toolkit/test_production_worker.py \
  tests/data_toolkit/test_worker_supervisor.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add data_toolkit/pipeline/cli.py \
  tests/data_toolkit/test_cli.py
git commit -m "feat: expose safe production lease handoff"
```

### Task 3: Deploy node-local concurrency policies

**Files:**
- Runtime create on Node16:
  `/home/youngwoo/data/pixal3d/control/runtime/parallelism_policy.json`
- Runtime modify on Node17:
  `/root/node17/data/pixal3d/control/runtime/parallelism_policy.json`
- Reuse tests: `tests/data_toolkit/test_parallelism_policy.py`,
  `tests/data_toolkit/test_worker_runtime.py`

**Interfaces:**
- Consumes: `read_parallelism_runtime_policy()` and registered execution roots.
- Produces:
  - Node16 policy `{max_chunks_in_flight: 2, render_workers_per_gpu_steps: [1, 2]}`
  - Node17 policy `{max_chunks_in_flight: 1, render_workers_per_gpu_steps: [1, 2]}`

- [ ] **Step 1: Re-run policy validation tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_parallelism_policy.py \
  tests/data_toolkit/test_worker_runtime.py
```

Expected: all tests pass.

- [ ] **Step 2: Drain Node16 and capture the live lease identity**

Use the worker registry to set Node16 to `draining`. Read and retain the exact
`unit_id`, `node_id`, token, attempt, and heartbeat from its `owner.json`.
Verify the heartbeat is fresh and the unit is HSSD batch003.

- [ ] **Step 3: Stop Node16 leaf groups and worker**

Send `SIGUSR2` to each direct command-supervisor child so each Blender/render
process group is killed. Wait until no process command contains the held
batch003 path, then terminate the Node16 worker. Keep its outer supervisor
alive and its registration drained.

- [ ] **Step 4: Perform the token-fenced handoff**

Invoke the new CLI action with the captured unit, node, and token. Verify:

- the lease directory is absent;
- the `.handoff` history directory exists;
- failed and stale counts remain zero; and
- the unit is pending with its original attempt preserved.

- [ ] **Step 5: Atomically install both policy files**

Write each policy through a temporary regular file in the same runtime
directory, set mode `0644`, and rename it over
`parallelism_policy.json`. Read each policy through
`read_parallelism_runtime_policy()` using that node's canonical bounds.

- [ ] **Step 6: Reactivate Node16**

Set Node16 active and verify its supervisor starts a new worker, the worker
reclaims HSSD batch003 with the preserved attempt, and no batch004 lease is
created by Node16 first.

### Task 4: Measure and select the faster Node16 policy

**Files:**
- Runtime evidence:
  `/root/data2/pixal3d/control/logs/hssd-throughput-tuning-20260728.jsonl`
- Runtime Node16 policy:
  `/home/youngwoo/data/pixal3d/control/runtime/parallelism_policy.json`

**Interfaces:**
- Consumes: chunk checkpoint `resource_peaks`, render completion directories,
  queue snapshot, resource telemetry, and GPU metrics.
- Produces: selected Node16 `max_chunks_in_flight` value and an evidence record
  containing terminal assets/hour and safety observations.

- [ ] **Step 1: Capture the two-chunk baseline**

At 60-second intervals record Node16 render completion counts, active chunk
IDs, Blender executable count, load average, CPU user/system/iowait,
available RAM, swap activity, per-GPU utilization/VRAM/temperature, and queue
counts. Continue until the first post-reload 64-asset chunk completes and its
output audit passes.

- [ ] **Step 2: Compute valid throughput**

Use checkpoint stage timestamps and terminal quality outcomes:

```text
terminal_assets_per_hour =
    completed_or_quarantined_assets / elapsed_wall_clock_hours
```

Reject the sample if it contains an OOM, stale lease, terminal queue failure,
persistent swap growth, resource hard stop, or failed output audit.

- [ ] **Step 3: Compare against one-chunk execution**

Drain Node16 only at a completed chunk boundary. Change only
`max_chunks_in_flight` from 2 to 1, preserve `[1, 2]` render steps, reload the
worker through the same token-fenced handoff, and measure the next completed
64-asset chunk with the same evidence fields.

- [ ] **Step 4: Select and persist the winner**

Retain the policy with the higher valid terminal assets/hour. If rates differ
by less than 5%, retain two chunks only when its peak load, RAM, swap, and
failure observations are no worse; otherwise choose one chunk.

- [ ] **Step 5: Verify production continuity**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m data_toolkit.pipeline.cli queue \
  --config data_toolkit/configs/multiview_preprocess.yaml --action status
```

Expected:

- both registered nodes are healthy and active after reload;
- HSSD is the highest-priority unfinished source;
- failed and stale counts are zero;
- HSSD completed count increased; and
- active nodes own distinct work units.

### Task 5: Final regression and operational report

**Files:**
- Verify only; no additional production files.

**Interfaces:**
- Consumes: Tasks 1-4 implementation and evidence.
- Produces: a concise report of the winning policy, measured assets/hour,
  revised HSSD ETA, and any remaining operational risk.

- [ ] **Step 1: Run focused regression tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m pytest -q \
  tests/data_toolkit/test_cli.py \
  tests/data_toolkit/test_work_queue.py \
  tests/data_toolkit/test_production_worker.py \
  tests/data_toolkit/test_worker_supervisor.py \
  tests/data_toolkit/test_parallelism_policy.py \
  tests/data_toolkit/test_worker_runtime.py \
  tests/data_toolkit/test_scheduler.py \
  tests/data_toolkit/test_commands.py
```

Expected: all tests pass with no failure.

- [ ] **Step 2: Verify live state**

Re-read both node registrations, queue snapshot, active process trees, latest
chunk checkpoints, and policy files. Confirm the observed state matches the
winning policy and there are no orphan Blender processes from the discarded
configuration.

- [ ] **Step 3: Report the result**

Report the before/after Blender concurrency, load, assets/hour, queue counts,
selected policy, and HSSD completion ETA. Clearly distinguish measured values
from projected values.
