# HSSD throughput tuning design

## Goal

Maximize completed HSSD assets per hour across Node16 and Node17 without
changing the canonical Pixal3D preprocessing format, invalidating completed
artifacts, consuming queue attempts for operator tuning, or risking OOM and
host instability.

## Observed bottleneck

Node16 currently admits three chunks concurrently and allows the render
profile to rise to four Blender workers per GPU. With six GPUs, this produces
up to 72 simultaneous Blender subprocesses across three chunks. The measured
load average exceeded 230 while GPU utilization remained between 0% and 6%.
The bottleneck is therefore host-side Blender/process contention rather than
insufficient GPU parallelism.

Node17's node-local one-chunk policy provides the working comparison. Its
completed 64-asset HSSD chunk took about 45 minutes, with rendering accounting
for about 24 minutes, while active GPUs reached materially higher utilization.

## Runtime policy

Use the existing node-local
`<local_root>/control/runtime/parallelism_policy.json` mechanism. It preserves
the canonical config hash and only reduces admitted parallelism.

- Node16 starts with two chunks in flight and render-worker steps `(1, 2)`.
  This permits 12 initial and at most 24 simultaneous Blender subprocesses.
- Node17 keeps one chunk in flight and uses render-worker steps `(1, 2)`.
  Its eight registered GPUs and encoder ranks remain available.
- GPU selection remains dynamic within each registered node pool. Existing
  target-80%, hard-100% VRAM policy remains unchanged.
- CPU, RAM, swap, storage, temperature, and queue lease safety checks remain
  authoritative.

The tuning does not introduce a new Pixal3D model or preprocessing operation.
It only changes how many existing chunk and Blender commands run concurrently.

## Safe transition

Workers enter `draining` before reload. Active preprocessing is stopped only
through a token-fenced, checkpoint-preserving operator handoff:

1. Stop new queue claims.
2. Stop the active worker and its supervised leaf process group.
3. Preserve downloaded raw assets, completed render directories, command
   checkpoints, and published outputs.
4. Return the active unit to pending without treating the operator handoff as
   a data or command failure.
5. Install and validate the node-local policy.
6. Reactivate the node and resume the same unit from its held checkpoints.

No batch directory, raw archive, completed render, or publication is deleted.
An ownership-token mismatch or a failure to persist the handoff leaves the
node drained and requires inspection rather than allowing a second owner.

## Empirical selection

The first completed 64-asset HSSD chunk after reload is the measurement unit.
For Node16, compare the two-chunk policy with a one-chunk policy using
checkpoint-bound stage timings and completed assets per wall-clock hour.
Change one variable at a time and retain the policy with the higher terminal
assets/hour rate.

The comparison is valid only when:

- there is no OOM or command failure;
- queue failures and stale leases remain zero;
- available RAM stays above the configured soft floor;
- swap-in/swap-out does not grow persistently;
- GPU and CPU temperatures stay below their configured limits; and
- the completed output audit passes.

GPU utilization is diagnostic evidence, not the optimization target.
Terminal assets/hour is the deciding metric.

## Error handling

- A malformed node policy fails closed before a worker claims a unit.
- A resource guard stop preserves checkpoints and leaves the unit retryable.
- Reproducible asset-quality failures continue through the existing
  quarantine/family-exclusion path.
- Infrastructure failures do not quarantine assets.
- Node16 and Node17 can be drained, removed, reactivated, or re-registered
  independently without invalidating the other node's lease.

## Verification

Automated tests cover policy bounds, canonical identity preservation,
checkpoint-preserving operator handoff, token fencing, and non-consumption of
failure attempts during a controlled reload.

Operational verification records, before and after the change:

- active Blender subprocesses by node and chunk;
- HSSD render completions over a fixed interval;
- per-stage elapsed seconds and terminal assets/hour;
- GPU utilization and VRAM use;
- load average, CPU system time, available RAM, and swap activity; and
- queue completed/running/pending/failed/stale counts.

The production run continues with the faster validated policy and retains the
source priority `ABO -> 3D-FUTURE -> HSSD -> ObjaverseXL_sketchfab`.
