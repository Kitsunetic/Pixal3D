# Output Compatibility Attestation Design

## Goal

Adopt the 1,489 usable ABO assets produced by tool commits
`0f4b290f3d23419f9389e669734cb7b8a50ec817` and
`fc26830338348e17929d7a734a4015ed4dca7bdd` without changing their packs,
manifests, member
checksums, or original `tool_commit` provenance. Continue to require strict
current-commit identity for every other historical commit.

This is a production-control compatibility mechanism. It does not change
Pixal3D rendering, geometry processing, voxel conversion, latent encoding,
family membership, or packing behavior.

## Evidence and trust boundary

The operator publishes one immutable JSON artifact at:

`control/output_compatibility.json`

The artifact is bound to:

- schema and artifact type;
- the active preprocessing configuration hash;
- pipeline version `pixal3d-mv-v2`;
- the exact approved historical commit IDs;
- the reviewed comparison baseline commit;
- the SHA-256 of the sorted changed-path evidence;
- an explicit empty list of output-affecting changed paths;
- a timezone-aware approval timestamp.

The approved historical commits are:

- `0f4b290f3d23419f9389e669734cb7b8a50ec817`, which introduced worker
  supervision;
- `fc26830338348e17929d7a734a4015ed4dca7bdd`, which applied shared GPU
  policy.

The review baseline is `1d36d55d7a15e32d85ce892a7a2b91f9809d149a`.
Repository comparison confirmed that the intervening changes affect
supervision, work-queue priority, GPU admission/policy, freeze-lock handling,
gate lifetime, telemetry serialization, and recovery controls. They do not
change adapters, Blender rendering, dual-grid/voxel generation, latent
encoding, family eligibility, or tar member construction.

The runtime reads the artifact with the existing no-follow regular-file safety
helpers. Unknown fields, duplicate commits, malformed hashes, a mismatched
configuration, a mismatched pipeline version, non-empty output-affecting paths,
or an unsafe file make the attestation invalid.

## Admission behavior

Pack and raw-archive verification retain all existing checks:

- pack and member SHA-256 verification;
- canonical source, shard, batch, gate, and family identity;
- exact asset membership and quarantine counts;
- active configuration hash;
- canonical index paths and manifest hashes;
- quality-ledger and raw-metadata agreement.

Only the `tool_commit` predicate changes:

1. The current runtime commit is always accepted, as before.
2. For the production gate only, an original manifest commit is accepted when
   it is listed in the valid compatibility attestation.
3. Smoke and pilot remain strict and never use compatibility evidence.
4. Missing or invalid compatibility evidence does not broaden admission.

The original manifests are never edited. The compatibility artifact supplies
the separate operator decision explaining why an older producer commit is
output-equivalent.

## Queue recovery

After both drained legacy validation attempts release their leases:

1. publish and validate the compatibility attestation;
2. run targeted reconciliation for production batches with existing packs;
3. adopt batch002 and batch004 through batch008 after complete pack/archive
   verification;
4. preserve batch009 and batch010 as current-commit completions;
5. keep batches with no valid publication pending or terminal according to
   their existing queue record;
6. reactivate node16 and node17.

The expected adopted usable ABO count is 1,948 assets:

- legacy-compatible: 1,489;
- current-commit: 459.

## Failure handling

- If an approved manifest fails any non-commit validation, it is not adopted.
- If the attestation is missing or invalid, historical commits fail closed.
- Queue remediation history remains under
  `control/runtime/work_queue/remediated/`.
- No existing pack or raw archive is deleted or overwritten.
- A future output-producing code change requires a new explicit compatibility
  review; this attestation does not automatically approve new commits.

## Verification

Automated tests must prove:

- strict rejection without an attestation;
- acceptance of an exact approved commit for production packs and raw
  archives;
- strict rejection for smoke and pilot;
- rejection of malformed, mismatched, or broadened attestations;
- preservation of original manifest bytes during reconciliation;
- queue adoption only after all pack and archive checks pass.

Before deployment, run the complete test suite. After deployment, compare
checksums on node16 and node17, reconcile the legacy batches, verify the
official asset count, and confirm both workers claim distinct pending units.
