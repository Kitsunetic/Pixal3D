# Node11 GPU Presence Watchdog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep a minimal CUDA context inside the existing `youngwoo_pixal3d` container so Grafana can identify the container even while the active preprocessing stage is CPU-bound.

**Architecture:** Add a standalone Python presence utility that allocates a bounded tensor on each visible GPU and sleeps with signal handling. Start it as a separate process inside the already-running Node11 container, and install a host-side watchdog that restarts only this presence process after container or process restarts. The preprocessing supervisor/worker and shared queue are not modified.

**Tech Stack:** Python 3.11, PyTorch 2.8/cu128 already installed in `youngwoo_pixal3d`, Docker CLI, systemd user service on Node11.

## Global Constraints

- Do not recreate or restart `youngwoo_pixal3d` while a production batch is active.
- Use all eight Node11 GPUs with a 16 MiB payload per GPU.
- Keep the presence process independent of the preprocessing supervisor/worker.
- Do not modify the shared queue, worker registry, or GPU admission policy.
- Stop cleanly on SIGTERM/SIGINT and release all CUDA tensors.

---

### Task 1: Add the bounded GPU presence utility

**Files:**
- Create: `tools/gpu_presence.py`
- Create: `tests/test_gpu_presence.py`

**Interfaces:**
- CLI: `python tools/gpu_presence.py --payload-mib 16 --interval-seconds 30`
- The utility discovers visible CUDA devices, allocates one float32 tensor per device sized to the requested payload, prints a JSON status line, and remains alive until SIGTERM/SIGINT.

- [ ] **Step 1: Write the failing unit tests**

Test tensor sizing and device selection without requiring a CUDA host by factoring pure helpers:

```python
from tools.gpu_presence import payload_elements, parse_device_ids

def test_payload_elements_uses_float32_bytes():
    assert payload_elements(16) == 16 * 1024 * 1024 // 4

def test_parse_device_ids_rejects_duplicate_or_negative_ids():
    assert parse_device_ids("0,2,7") == [0, 2, 7]
    for value in ("0,0", "-1", ""):
        try:
            parse_device_ids(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest -q tests/test_gpu_presence.py`
Expected: FAIL because `tools/gpu_presence.py` does not exist.

- [ ] **Step 3: Implement the minimal utility**

Implement `payload_elements`, `parse_device_ids`, signal handlers, CUDA availability checks, per-device allocation, periodic JSON heartbeat, and cleanup. Refuse a payload below 1 MiB or above 128 MiB; default to 16 MiB. If one device cannot allocate, release all allocations and exit non-zero without touching other processes.

- [ ] **Step 4: Run the focused test**

Run: `pytest -q tests/test_gpu_presence.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gpu_presence.py tests/test_gpu_presence.py
git commit -m "ops: add bounded gpu presence utility"
```

### Task 2: Add the Node11 watchdog and operator documentation

**Files:**
- Create: `scripts/node11_gpu_presence_watchdog.sh`
- Create: `ops/systemd/pixal3d-node11-gpu-presence.service`
- Modify: `docs/ops/node11-worker.md`

**Interfaces:**
- Watchdog command: `scripts/node11_gpu_presence_watchdog.sh`
- Environment: `PIXAL3D_PRESENCE_CONTAINER=youngwoo_pixal3d`, `PIXAL3D_PRESENCE_GPUS=0,1,2,3,4,5,6,7`, `PIXAL3D_PRESENCE_PAYLOAD_MIB=16`.

- [ ] **Step 1: Add watchdog logic**

The script must check that the named container is running, find the presence PID with `pgrep -f 'tools/gpu_presence.py'` inside the container, launch it with `docker exec -d` if absent, and sleep 30 seconds between checks. It must never stop or signal the preprocessing supervisor/worker.

- [ ] **Step 2: Add the user systemd unit and document installation/removal**

Add a user-scoped systemd unit with `Restart=always`; do not reference the system-scoped `docker.service` from the user manager. Document the exact Node11 commands to install the unit, verify the process and `nvidia-smi`, and remove the watchdog. Include the expected container name and the fact that CUDA context overhead is in addition to the 16 MiB payload.

- [ ] **Step 3: Shell-check the watchdog**

Run: `bash -n scripts/node11_gpu_presence_watchdog.sh`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/node11_gpu_presence_watchdog.sh docs/ops/node11-worker.md
git commit -m "ops: keep node11 gpu container visible"
```

### Task 3: Deploy without interrupting preprocessing and verify Grafana-visible GPU presence

**Files:**
- No repository files.

**Interfaces:**
- Remote target: `youngwoo@n11.unist.info:55555`
- Container: `youngwoo_pixal3d`

- [ ] **Step 1: Copy the utility and watchdog into the mounted checkout**

The container already mounts the repository, so use the worktree files directly; do not recreate the container.

- [ ] **Step 2: Start the watchdog as a detached Node11 user service**

Install a service that runs the watchdog with `Restart=always`, then start it. Do not restart Docker or the preprocessing worker.

- [ ] **Step 3: Verify presence process and GPU footprint**

Check:

```bash
docker exec youngwoo_pixal3d pgrep -af tools/gpu_presence.py
docker exec youngwoo_pixal3d nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pgrep -af 'data_toolkit.pipeline.cli (supervisor|worker)'
```

Expected: one presence process, small nonzero memory on GPUs 0–7, and the existing supervisor/worker PIDs unchanged.

- [ ] **Step 4: Test watchdog recovery without touching the worker**

Terminate only the presence PID, wait one watchdog interval, and verify a new presence PID appears while the supervisor/worker PIDs and current batch lease remain unchanged.

- [ ] **Step 5: Record the result**

Report the observed container name, per-GPU memory, presence PID, and unchanged preprocessing worker PIDs.
