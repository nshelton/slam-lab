# Solver and workbench backend additions

Status: in progress. This list turns the workbench feedback into ordered backend tasks.
The current file-based pipeline remains usable; preserve SQLite/NPZ/JSON and optional
Rerun export. The proposed contract is specified in
[WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md#planned-workbench-contract--not-yet-implemented).

Suggested slice 1 is implemented for the single supported solver workflow. Matching is
the self-contained root artifact, geometry and reconstruction bind to exact parent IDs,
and every stage validates payload hashes. New solves also publish stable mapping arrays,
atomic structured job status, `solve-status`, and the manifested `ArtifactReader`.
There is no compatibility adapter for unmanifested artifacts.

## P0 — reliable selection and solve monitoring

### 1. Introduce artifact identity and parent manifests

- [x] Reserve opaque artifact IDs independently of paths and configuration hashes.
- [x] Add schema/version, type, direct parent ID, producer job ID, payload inventory
  and hashes. Matching is the root and embeds the selected observation namespace.
- [x] Require and validate schema/capabilities; reject unmanifested inputs.
- [x] Require geometry/solve inputs to identify a fixed parent snapshot. Preserve
  pair-coverage metadata for intentionally partial inputs.

Acceptance: moving an artifact preserves identity; mismatched parents are rejected;
two runs with identical settings are not conflated merely because their settings match.

Likely touchpoints: `config.py`, `cache.py`, `correspondence.py`, `verification.py`,
`solver.py`, and a shared artifact/manifest module.

### 2. Preserve observation -> track -> point lineage

- [x] Carry `(matching_artifact_id, matching_row, feature_id)` through every stage.
- [x] Export `frame_rows`, `source_track_ids` and stable `point_ids` in reconstruction
  NPZ; record source matching and geometry artifact IDs in metadata.
- [x] Preserve mappings through seed retries, triangulation, BA, point filtering and
  array compaction. Do not reuse point IDs within a job.
- [x] Keep `observations[:, 1]` as an array-row index; expose the stable ID separately.
- [x] Treat source tracks as geometry-scoped IDs, not appearance-track IDs. Allow
  future one-track-to-many-point splitting without breaking selection semantics.

Acceptance: select any final point, recover its verified source track, visit every
retained observation and recover the exact source frame/feature/pixel. Repeat with
frame subsampling, rejected observations, removed points and reordered arrays.

Likely touchpoints: `ReconstructionProblem`, `GraphMapper.track_points`,
`Mapper.finalize()`/`final_point_ids`, and `save_reconstruction()`.

### 3. Add structured job status and solve progress

- [x] Add a minimal job directory with atomically replaced `status.json`; do not wait
  for snapshots or a full workspace manager to support solve monitoring.
- [x] Separate common lifecycle `state` from stage-specific `phase`; add `job_id`,
  `revision`, `updated_at`, `last_error`, intended artifact ID and worker liveness.
- [x] Emit registered/total frames, current matching row, current points and genuine
  BA iteration progress when available (otherwise null).
- [x] Expose `solve-status` and a reader `status()`.
- [x] Report startup, initialization failures, cancellation, publication and exit.
  A canceled solve is not paused/resumable just because a preview exists.

Acceptance: a GUI follows a complete solve without parsing stdout; observed JSON is
always complete, revisions increase, no false complete state follows failure, and a
worker crash is reconciled by the supervisor. Callback/evaluation counts are not
misreported as optimizer iterations.

Likely touchpoints: `cli.py`, `GraphMapper.run()`, `initialize_graph()`,
`extend_tracks()`, `bundle.py`, `solve_geometry()`, and a shared job-status writer.

## P1 — live visualization and robust publication

### 4. Separate job workspaces from immutable published artifacts

- [ ] Add `jobs/<job-id>/{job.json,status.json,stdout.log,stderr.log,draft/}` and
  `artifacts/<artifact-id>/`; retain an explicit compatibility path for existing CLI runs.
- [ ] Keep active matching/verification databases in drafts. Freeze published inputs;
  extending a completed run creates a new artifact instead of rebuilding it in place.
- [ ] Make matching while verifying explicit: freeze the consumed pair set as an
  immutable partial parent, rather than pointing at a mutable live database.
- [x] Store full effective argv as an array, cwd, resolved interpreter/options,
  allowlisted runtime/environment provenance, input/output IDs and locations,
  timestamps, termination result and retry/resume lineage.

Acceptance: rerun-with-modifications creates a new attributable job; paths containing
spaces remain exact argv entries; no credentials are persisted; an older reconstruction's
parent remains readable and unchanged while newer matching/geometry proceeds.

### 5. Publish optional live solve snapshots

- [ ] Capture coherent solver state at safe boundaries with configurable cadence.
- [ ] Write immutable NPZ snapshots with poses, points, colors, observations, K,
  mappings, stable point IDs, source track IDs and valid/unknown residual flags.
- [ ] Publish snapshot first, then its reference in status; include snapshot revision,
  capture time and whether a seed/solution is provisional.
- [ ] Keep snapshot row indices consistent when sampling points; distinguish previews
  from computation checkpoints and expose absence of resume support.

Acceptance: the GUI can load every referenced snapshot during solving; selection
survives BA and compaction; removed points disappear without their IDs being reused;
preview writing can be disabled without changing the numerical result.

### 6. Enforce publication and crash-recovery invariants

- [x] Share atomic JSON-writing helpers; order payload -> manifest -> job completion.
- [x] Freeze SQLite with a supported backup/checkpoint strategy, independent of live
  WAL/SHM files. Validate readable payload inventory before publication.
- [ ] Publish the complete artifact directory atomically without overwriting another
  artifact, and document/implement the intended durability guarantees.
- [ ] Recover a crash between artifact publication and status completion by validating
  the artifact and reconciling status, not misclassifying a successful publication.
- [ ] Keep failures/cancellation in jobs; move optional Rerun exports outside sealed
  artifacts so export failures cannot change core-computation success.

Acceptance: injected failures at each publication boundary expose either a valid
complete artifact or a partial job, never a manifest referencing unfinished files.

## P1/P2 — stable reader facade and migration

### 7. Add supported, paginated reader methods

- [x] Provide `manifest()`, `list_frames()`, `frame()`, `list_pairs()`, `pair()`,
  `status()`, `load_snapshot()` and `load_reconstruction()` with type capabilities.
- [ ] Implement opaque live pair cursors based on append order/revision, not pair
  sorting; test late out-of-order commits and empty-page polling.
- [ ] Advance draft `data_revision` with committed batches and distinguish it from
  JSON status revisions so readers can detect data changes between progress updates.
- [x] Validate schema versions, ancestry, canonical pair/pose direction and cursor scope.
- [x] Add round-trip examples for point selection and job watching to the interface
  document once these APIs actually ship; do not mark proposals as available early.

Acceptance: a GUI reads manifested artifacts through one facade, discovers every
committed pair exactly once, and detects unsupported features.

Start the facade alongside P0 for `status()` and mappings. Broader paging/migration
work can ship alongside the first workbench prototype.

## Suggested implementation slices

1. ~~Artifact envelope + final observation mappings + minimal atomic solve status.~~
2. Job layout/provenance + frozen publication + reader facade.
3. Optional live snapshots + complete pagination/recovery coverage.

Document all additions as planned until each slice is implemented and tested. This
list is integration work, not a change to the SfM algorithm. Global solver adapters,
periodic/local BA, robust track splitting, calibration and dynamic-scene handling
remain separate numerical-solver work in [SOLVER_DESIGN.md](SOLVER_DESIGN.md).
