# Output Compatibility Attestation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Safely adopt the 1,489 usable ABO assets produced by the two reviewed historical tool commits while leaving every pack, raw archive, manifest, checksum, and provenance field unchanged.

**Architecture:** Add a small fail-closed compatibility-attestation reader that validates one immutable control-plane JSON file and answers whether a manifest producer commit may be treated as output-compatible. Wire that predicate into the existing pack and raw-archive verifiers only for production; all existing identity, membership, hash, quality, path, and schema checks remain unchanged, and smoke/pilot retain exact current-commit matching. Publish the reviewed artifact, reconcile the six legacy batches, deploy the same code and artifact to both nodes, and resume the shared queue.

**Tech Stack:** Python 3.11, dataclasses/JSON/SHA-256, pytest, existing Pixal3D pipeline and queue control code, Docker/SSH deployment on node16.

---

### Task 1: Implement strict compatibility-artifact validation

**Files:**
- Create: `data_toolkit/pipeline/output_compatibility.py`
- Create: `tests/data_toolkit/test_output_compatibility.py`

**Step 1: Write the failing tests**

Cover:

- missing evidence returns no approved historical commits;
- a valid, exact artifact approves both reviewed commits and the exact review
  baseline as their trust anchor;
- unknown keys, duplicate commits, malformed/full-length commit hashes, an incorrect config hash, an incorrect pipeline version, a different review baseline, a changed-path digest mismatch, a non-empty output-affecting path list, a naive timestamp, and a symlinked artifact fail closed;
- changed-path evidence is sorted and its digest is recomputed from newline-delimited paths;
- the artifact reader never writes or modifies the evidence file.

Use temporary `PipelineConfig` roots and capture the artifact bytes before and after reading.

**Step 2: Run the tests to verify they fail**

Run:

```bash
conda run -n pixal3d python -m pytest tests/data_toolkit/test_output_compatibility.py -q
```

Expected: FAIL because `data_toolkit.pipeline.output_compatibility` does not exist.

**Step 3: Write the minimal implementation**

Implement:

- immutable schema constants for `pixal3d-mv-v2` and review baseline `1d36d55d7a15e32d85ce892a7a2b91f9809d149a`;
- strict exact-key validation with two exact reviewed historical commits;
- no-follow, regular-file-only JSON reading rooted at `config.paths.data2_root / "control"`;
- timezone-aware RFC 3339 timestamp validation;
- sorted changed-path validation and SHA-256 recomputation;
- exact `config.config_hash()` and pipeline-version binding;
- explicit rejection of any output-affecting changed path;
- a result object exposing the approved commits without mutating the artifact.

Missing evidence returns an empty approval set. Present-but-invalid evidence raises a dedicated validation exception so callers can report a precise fail-closed reason.

**Step 4: Run the tests to verify they pass**

Run:

```bash
conda run -n pixal3d python -m pytest tests/data_toolkit/test_output_compatibility.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add data_toolkit/pipeline/output_compatibility.py tests/data_toolkit/test_output_compatibility.py
git commit -m "feat: validate output compatibility attestations"
```

### Task 2: Apply compatibility only to production verification

**Files:**
- Modify: `data_toolkit/pipeline/orchestrator.py`
- Modify: `tests/data_toolkit/test_orchestrator.py`

**Step 1: Write the failing verifier tests**

Build ordinary valid pack and raw-archive fixtures, then rewrite only their original publication `tool_commit` before checksums are finalized or republish them with a historical commit. Prove:

- production pack verification accepts an exact approved historical commit;
- production raw-archive verification accepts the same commit;
- the same historical commit is rejected for smoke and pilot;
- a non-approved commit is rejected;
- a missing or malformed artifact rejects the historical commit;
- all other pack/archive validation failures still reject an approved historical commit;
- manifest bytes remain unchanged across verification.

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
conda run -n pixal3d python -m pytest tests/data_toolkit/test_orchestrator.py -q -k 'compatib or tool_commit'
```

Expected: new compatibility tests FAIL because verification still requires exact current-commit identity.

**Step 3: Write the minimal integration**

Add one `PipelineServices` helper:

```python
def _tool_commit_is_admissible(self, context, producer_commit):
    if producer_commit == self._resolved_tool_commit():
        return True
    if context.gate != "production":
        return False
    return producer_commit in load_output_compatibility(
        self.config
    ).compatible_tool_commits
```

Convert artifact validation failures into the existing `ValidationError`. Replace only the two exact `tool_commit` comparisons in `_verify_published_batch` and `_verify_raw_archive`; do not weaken any other predicate.

**Step 4: Run the focused tests**

Run:

```bash
conda run -n pixal3d python -m pytest tests/data_toolkit/test_output_compatibility.py tests/data_toolkit/test_orchestrator.py -q -k 'compatib or tool_commit'
```

Expected: PASS.

**Step 5: Run the full test suite**

Run:

```bash
conda run -n pixal3d python -m pytest -q
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add data_toolkit/pipeline/orchestrator.py tests/data_toolkit/test_orchestrator.py
git commit -m "feat: admit attested production outputs"
```

### Task 3: Generate and validate the production attestation

**Files:**
- Create operational artifact: `/root/data2/pixal3d/control/output_compatibility.json`

**Step 1: Recompute review evidence**

For each historical commit, compute the sorted changed paths against baseline:

```bash
git diff --name-only 0f4b290f3d23419f9389e669734cb7b8a50ec817 1d36d55d7a15e32d85ce892a7a2b91f9809d149a | LC_ALL=C sort -u
git diff --name-only fc26830338348e17929d7a734a4015ed4dca7bdd 1d36d55d7a15e32d85ce892a7a2b91f9809d149a | LC_ALL=C sort -u
```

Review the complete lists and confirm no changes to adapters, command construction, rendering, geometry/voxel/latent production, packing, family membership, or output formats.

**Step 2: Write the artifact atomically**

Use an adjacent temporary regular file, `fsync`, and `os.replace`. Include:

- exact schema/artifact type;
- current config hash;
- `pixal3d-mv-v2`;
- timezone-aware approval time;
- exact review baseline;
- both reviewed commits;
- complete sorted changed-path lists and their newline-delimited SHA-256 values;
- empty `output_affecting_changed_paths`.

Mode must be non-executable and the target must not be a symlink.

**Step 3: Validate with production code**

Run a short `pixal3d` environment check that loads the artifact and prints the two exact approved commits. Record the artifact SHA-256 and preserve its bytes through later reconciliation.

### Task 4: Reconcile legacy batches without modifying publications

**Files:**
- Read/write queue state under: `/root/data2/pixal3d/control/runtime/work_queue/`
- Read-only publications under: `/root/data2/pixal3d/prepared/` and `/root/data3/pixal3d/archive/raw/`

**Step 1: Capture pre-reconciliation evidence**

Record SHA-256 values for each legacy batch002 and batch004–batch008 pack manifest and raw-archive manifest, plus the compatibility artifact.

**Step 2: Run reconciliation**

With both nodes drained, invoke the existing queue reconcile action under the new runtime commit. It must run full pack/archive verification and adopt only valid existing publications.

**Step 3: Verify adoption and immutability**

Confirm:

- batch002 and batch004–batch008 are completed;
- batch009 and batch010 remain completed;
- missing/unpublished batches remain pending/failed according to queue policy;
- all pre-recorded publication manifest SHA-256 values are unchanged;
- the compatibility artifact SHA-256 is unchanged;
- official completed usable ABO assets total 1,948, comprising 1,489 compatible legacy plus 459 current-commit assets.

If any non-commit validation fails, leave that batch unadopted and report the exact predicate; do not edit or regenerate its publication.

### Task 5: Deploy, reactivate both nodes, and verify live progress

**Files:**
- Deploy the committed runtime to node17 worktree and node16 container repo.
- Share `/root/data2/pixal3d/control/output_compatibility.json` through the mounted data root.

**Step 1: Deploy exact committed code**

Copy only committed source files needed by the runtime to `/home/youngwoo/Pixal3D` inside node16's `youngwoo_diyscene` container. Confirm source SHA-256 equality between node16 and node17. Use the exact new Git commit as `PIXAL3D_TOOL_COMMIT` on both nodes.

**Step 2: Reactivate worker registries**

Set:

- node16 GPUs `0,1,2,3`;
- node17 GPUs `1,2,3,4,5,6`;
- source priority `ABO`, `3D-FUTURE`, `HSSD`, `ObjaverseXL_sketchfab`;
- target GPU utilization 80% with 100% hard VRAM ceiling;
- the existing work-conserving shared queue policy.

Start or restart both supervisors only after deployment and reconciliation.

**Step 3: Verify distinct live claims**

Poll registry, queue, process, and checkpoint evidence until:

- node16 and node17 are healthy/active;
- each node claims a distinct pending production unit, or a node is legitimately waiting on the same source's finite tail;
- no stale lease or duplicate claim exists;
- GPU/CPU stage processes appear when their claimed batch reaches those stages;
- queue counts advance without altering the eight adopted publications.

**Step 4: Report current production state**

Report exact completed/pending/running/failed counts, official completed usable assets by source, active batch IDs and stages per node, and the next expected source transition.
