# SLAM workbench integration

The processing API is Python functions plus persistent SQLite/NPZ/JSON artifacts.
Rerun is a presentation adapter, not the processing engine. There is no HTTP/RPC
service or live event bus. A GUI can supervise CLI subprocesses and read artifacts,
or wrap the Python API in its own worker process. Avoid running compute in the UI
thread. The sections through "Current progress, cancellation and concurrency"
describe the implemented solver. The final contract section marks implemented identity,
lineage and status fields separately from the remaining job/snapshot/reader work.
The ordered backend task list is [SOLVER_TODO.md](SOLVER_TODO.md).
The verified result inventory is in [SESSION_HANDOFF.md](SESSION_HANDOFF.md), with
inspection and experiment commands in [WORKFLOW.md](WORKFLOW.md).

Keep SQLite/NPZ/JSON as the interchange formats. The backend should not emit Three.js
objects or frontend-specific scene files. The workbench supervises CLI subprocesses,
uses reader APIs, and treats persisted state as authoritative.

## Pipeline and commands

```text
video -> process -> feature cache
feature cache -> track-online -> causal descriptor landmarks
feature cache -> match-all -> raw pair matches + appearance-only tracks
raw pair matches -> verify-matches -> F/E/H + verified tracks
verified tracks -> solve -> camera poses + reconstructed points
any artifact -> GUI or optional Rerun viewer/export
```

Run commands from the project directory; use absolute paths in a GUI supervisor.
The CPU environment is `.venv`; CUDA LightGlue uses `.venv-cuda`.

```bash
.venv/bin/slam-lab list
.venv/bin/slam-lab process /path/to/video.mp4 --max-frames 300 --stride 6
.venv-cuda/bin/slam-lab track-online /path/to/feature-cache \
  --output /path/to/online-tracks --device cuda
.venv/bin/slam-lab track-status /path/to/online-tracks
.venv-cuda/bin/slam-lab match-all /path/to/feature-cache \
  --output /path/to/matches --matcher lightglue --device cuda --workers 1 --no-rerun
.venv/bin/slam-lab match-status /path/to/matches
.venv/bin/slam-lab verify-matches /path/to/matches \
  --output /path/to/geometry --fx 886.8 --fy 886.8 --quiet
.venv/bin/slam-lab geometry-status /path/to/geometry
.venv/bin/slam-lab solve /path/to/geometry --output /path/to/solution --no-rerun
```

`verify-matches --frame-step 10 --max-frames 30` selects every tenth matching row,
up to 30 images. It checks every available pair among those images. Omitting these
options selects all images. `--fx/--fy/--cx/--cy` use cached preview pixels. Defaults
are a shared 60-degree horizontal FOV, square pixels, centered principal point,
and zero distortion. These are assumptions, not an estimated calibration.

Feature extraction, matching and verification can resume. Repeat the same command
and output path. Changed geometry settings require a new output. `solve` publishes
a new output directory and refuses to overwrite an existing solution. `--no-rerun`
is supported by `match-all` and `solve`. Viewer
commands are separate: `view`, `view-matches`, `view-geometry`, `view-reconstruction`.

The only supported reconstruction path is **verify-matches -> solve**.

`track-online` is independent of the reconstruction path. It consumes the same
frame-level SuperPoint cache in order, compares each descriptor only with eligible
landmark means from earlier frames, enforces one observation per landmark per
frame, and commits each completed frame to `tracks.sqlite3`. The workbench can read
the database while it is running through SQLite WAL snapshots. Completed outputs
also contain `summary.json` and an immutable `manifest.json`. Online track artifacts
do not copy preview JPEGs; the source feature cache remains their authoritative
image store and is identified in metadata and lineage.

## Python entry points

```python
from pathlib import Path
from slam_lab.reader import ArtifactReader

matches = ArtifactReader(Path("recordings/osaka-allpairs-lightglue"))
frame = matches.frame(100)
pair = matches.pair(100, 101)

geometry = ArtifactReader(Path("recordings/osaka-cosine-geometry-step10"))
diagnostics = geometry.pair(100, 110)

solution = ArtifactReader(Path("recordings/osaka-lightglue-solver-v5"))
arrays = solution.load_reconstruction()
```

Core functions do not spawn a viewer. `match_all(..., export=False)` is the default
for Python callers. `verify_matches` returns a summary dict; `solve_geometry` returns
the output `Path`. Failures raise exceptions. The CLI returns nonzero on failure;
stdout is not uniformly a JSON protocol. Only `match-status` and `geometry-status`
promise a single JSON object. Capture stdout/stderr as job logs for other commands.

`ArtifactReader` is the only supported read interface. Compute functions remain available
for workers and tests. `ReconstructionProblem` carries a fixed camera, descriptor-free image observations,
observation-to-track IDs, track lengths, ranked seed candidates and provenance.
The initial `GraphMapper` backend uses all available 2D–3D tracks for registration;
it does not invoke SuperPoint, LightGlue, or nearest-neighbor matching. Other solver
backends can consume the same observations and tracks. Full F/E/H view-graph records
remain available through the geometry artifact reader.

## IDs and coordinate conventions

Do not conflate these three IDs:

* **Matching row**: zero-based row in `matches.sqlite3.frames`, used in pair keys.
* **Original frame index**: source video frame index, potentially separated by stride.
* **Solution row**: compact row in the geometry selection/reconstruction arrays.

`geometry/tracks.npz.frame_rows[solution_row]` maps to matching row. Geometry SQLite
retains the original matching row IDs even after subsampling. Pair APIs and
`view-geometry --pair` use these matching rows. `GeometryStore.pair(a,b)` requires
canonical `a < b`; `MatchStore.pair` accepts either order and swaps indices.

A feature ID is its index in that frame's cached keypoint array. Pixel coordinates
refer to the cached image, with x right and y down. Timestamps are integer nanoseconds.
Track IDs are local to one artifact, and may change when rebuilding tracks with new
matches. Use the artifact identity together with the track ID, never track ID alone.

Saved solution poses are **T_world_camera**: camera-to-world 4x4 transforms with
camera axes right, down, forward. Matrix translation is the camera center in world
coordinates. Internal geometric poses and `pose_second_from_first` map the other
direction: `X_second = R @ X_first + t`. Two-view t has unit length and unknown scale.
The final reconstruction is also arbitrary-scale; seed baseline = 1, not meters.

## Current match artifacts

`matches.sqlite3` is authoritative. Prefer the reader class over decoding blobs.
The table descriptions below explain today's storage; raw SQL remains internal.

* `metadata(key TEXT PRIMARY KEY, value TEXT)` stores JSON values: identity, active
  matcher, phase, frame count, source/cache paths, timing and final summary.
* `frames(row_id, frame_index, timestamp_ns, width, height, keypoints,
  keypoint_count)` contains raw little-endian float32 Nx2 coordinates. Images are
  addressed in the source video by timestamp.
* `pairs(first, second, count, matches, elapsed_ms, runtime_id)` stores one upper
  triangle entry per pair. `matches` is zlib-compressed structured data with fields
  `first:int32`, `second:int32`, `score:float32`, `distance:float32` (little endian).
* `match_runtimes(runtime_id, provenance)` identifies CPU/GPU/PyTorch provenance per
  pair.

`tracks.npz` here contains **appearance-only** tracks. It is not solver input.
`track_ids[offsets[i]:offsets[i+1]]` maps keypoints in matching row i to track IDs.
Additional arrays: `track_lengths`, `pair_match_counts`, `frame_indices`, `timestamps_ns`.
`manifest.json` identifies this root artifact and hashes the database, tracks and summary.

## Current geometry artifacts

`geometry.sqlite3` contains its own selected keypoints and references the source video
through its parent artifacts. Source matches remain unchanged.

* `metadata` holds JSON: `identity` (source identity, selected rows, camera, config,
  estimator/OpenCV version), `phase`, `tracks_identity`, `source_match_run`, summary.
* `frames` has the same layout as matching, with matching row IDs preserved.
* `pairs(first, second, source_hash, arrays, summary)` stores a hash of the exact raw
  match blob, an NPZ blob, and a JSON diagnostic summary. Reopening a completed pair
  does not recompute RANSAC. Source mutations are detected on verification resume.

`GeometryStore.pair(a,b)` returns the decoded arrays and summary:

| Array | Shape / meaning |
| --- | --- |
| `indices`, `scores` | Original matches Nx2 and appearance scores N |
| `F`, `E`, `H` | 3x3 matrices; NaNs when estimation failed |
| `F_inliers`, `E_inliers`, `H_inliers` | Raw estimator masks N, aligned to `indices` |
| `F_error_px`, `E_error_px` | Square-root Sampson residuals in cached pixels |
| `H_error_px` | Forward homography transfer error in pixels |
| `verified` | F AND E inliers, cleared when support is below minimum |
| `cheirality` | recoverPose output mask, separate from raw E RANSAC mask |
| `triangulated` | Positive-depth, reprojection, parallax and F checks |
| `parallax_deg` | Triangulation angles N; interpret with the validity masks |
| `pose_second_from_first` | Relative world-to-camera convention described above |

Summaries contain support counts, coverage, parallax, homography fraction,
`seed_eligible`, `seed_score`, rejection `reasons`, and optional `estimator_errors`.
Homography-dominated/low-parallax pairs can retain useful track edges while being
ineligible as seeds. An inlier is not proof of a static scene or correct calibration.

Geometry `tracks.npz` arrays: `frame_rows`, `offsets`, `track_ids`, `track_lengths`,
scalar `geometry_identity`. Unlike appearance tracks, these use only verified edges.
Association orders edges by geometric residual and enforces one feature per image.
Raw pair records preserve the edges; individual union-conflict decisions are not yet
exported (only counts). Tracks are candidates subject to multiview rejection.
`manifest.json` binds this artifact to the exact matching parent and hashes its database,
tracks and summary. Verification rejects missing, modified or mismatched parents.

## Current reconstruction artifacts

`reconstruction.npz` uses `allow_pickle=False` and contains:

| Array | Shape / meaning |
| --- | --- |
| `points`, `colors` | P x 3 world coordinates, P x 3 uint8 RGB |
| `T_world_camera` | F x 4 x 4 camera-to-world poses; unregistered rows are NaN |
| `registered` | F boolean mask; never interpolate missing poses implicitly |
| `frame_indices`, `timestamps_ns` | F original frame IDs and nanosecond timestamps |
| `K` | Shared fixed 3x3 pinhole intrinsic matrix |
| `observations` | O x 3 integers: solution row, point array row, frame-local feature ID |
| `observation_xy` | O x 2 observed cached-image coordinates |
| `reprojection_errors` | O residual lengths in pixels, after final filtering |
| `track_lengths` | P surviving observation counts |
| `frame_rows` | F original matching rows; present in new graph-solver artifacts |
| `source_track_ids` | P verified geometry-track IDs; present in new graph-solver artifacts |
| `point_ids` | P job-scoped stable point IDs; present in new graph-solver artifacts |

`poses.json` holds per-frame status, pose or null, support and PnP counts.
`metadata.json` records input paths/identity, camera assumptions, solver settings,
seed/third-view decisions, BA report, counts and limitations. `points.ply` is a
convenient point-cloud export. `ransac.npz`/`ransac.json` hold raw PnP/seed decisions
separately from final retained observations. Reconstruction artifacts also export
`frame_rows`, `source_track_ids`, and stable `point_ids`.
New outputs include `manifest.json` with their opaque reconstruction ID, producer job,
declared capabilities, and SHA-256/size inventory for all core payloads.

The first solver is a prototype graph-based incremental backend with final joint
BA, not global SfM. It reconstructs one seed-connected component, tries up to 50
ranked seeds, and requires third-view registration when more than two views exist.
It has no track splitting or periodic/local BA yet. It leaves other images unposed.

## Current progress, cancellation and concurrency

SQLite writers use WAL; readers should open read-only and keep transactions short.
`MatchStore` and `GeometryStore` default to read-only. Do not let the GUI update live
databases. One writer per output is enforced with an OS file lock. Launch new jobs
as supervised subprocesses, rather than importing them into the web server thread.

`match-status`: `running`, `phase`, completed/total pairs, measured rate/ETA and
matcher/runtime provenance. `geometry-status`: `running`, `phase`, completed pairs,
`available_pairs` at snapshot start, `possible_pairs` in the selection, source
completion flag, last error and final summary. Check `running` as well as phase;
a killed process can leave its last phase unchanged. Rate/ETA can also be stale.

Verification consumes only a completed, manifested matching artifact. Completed geometry
is immutable: repeating the same command validates and returns it, while changed inputs,
settings, payloads or parents require a new output directory.

Send SIGINT for graceful cancellation of matching/verification. Committed pairs
remain resumable. `solve` now writes atomic structured progress under
`jobs/<job-id>/status.json`; read it with `solve-status <output-or-job-directory>` or
`ArtifactReader.status()`. It still has no checkpoint/resume support. It publishes
the final artifact directory atomically after success; an interrupted solve must restart.

`.rrd` files are optional exported snapshots, not live state or a data interchange
requirement. Rerun calls live in `viewer.py`, `match_view.py`, `geometry_view.py`,
the viewing half of `reconstruction_io.py`, and `ransac_view.py` logging helpers.

## Planned workbench contract — not yet implemented

Stable mappings, mandatory matching/geometry/reconstruction manifests, immutable
cross-stage ancestry, structured solve status and the minimal reconstruction reader are
implemented. Solve provenance and immutable published-artifact pagination are also
implemented. Live snapshots, live draft pagination and crash reconciliation remain planned.

### Stable observation and point mappings

The canonical observation key within a matching lineage will be:

```text
(matching_artifact_id, matching_row, feature_id)
```

Geometry subsampling must preserve the matching row and feature ID. A verified
track is identified by `(geometry_artifact_id, track_id)`; appearance-track IDs
from matching are a separate namespace. New geometry artifacts may regroup or
renumber tracks without changing the observations they reference.

Add these fields to reconstruction artifacts and live snapshots:

| Field | Contract |
| --- | --- |
| `matching_artifact_id` in metadata | Namespace of all source observations |
| `source_geometry_artifact_id` in metadata | Namespace of `source_track_ids`; equals the reconstruction's direct parent |
| `frame_rows[F]` in NPZ | Solution row -> original matching row, including unregistered images |
| `source_track_ids[P]` in NPZ | Point array row -> verified input track ID, retained through filtering and BA |
| `point_ids[P]` in NPZ | Stable point identifiers across snapshots and the final result from one solve job |

Keep the current `observations[O,3]` layout: `(solution_row, point_array_row,
feature_id)`. Its second column remains an array row, not the new stable point ID.
For an observation `r`, resolve selection with:

```python
observation_key = (
    matching_artifact_id,
    frame_rows[observations[r, 0]],
    observations[r, 2],
)
stable_point_id = point_ids[observations[r, 1]]
source_track_id = source_track_ids[observations[r, 1]]
```

Point IDs are unique and never reused within a job, including after a rejected seed
attempt or point removal. Array rows may compact; the IDs must not. Snapshots identify
their job, and the final manifest records `producer_job_id`, so the GUI can follow
these IDs through publication. They are not correspondence IDs across separate solves.

The current backend creates at most one point per input track. Future track splitting
may produce multiple points with the same source track ID, so consumers must not
assume the reverse mapping is one-to-one. If a future backend merges source tracks,
it must add an explicit many-to-many lineage table rather than invent a single ID.
Imported points without lineage must advertise missing mappings, not guess them.

This observation key is stable across geometry/solve stages in one matching artifact.
Comparing two different matchers additionally requires a shared feature-artifact
identity and explicit frame mapping; matching-row numbers alone are insufficient.

### Artifact identities and ancestry

Every completed matching, geometry and reconstruction artifact gets `manifest.json`.
A minimal geometry example is:

```json
{
  "schema_version": 1,
  "artifact_id": "geometry-<uuid>",
  "artifact_type": "geometry",
  "parent_artifact_id": "matches-<uuid>",
  "producer_job_id": "job-<uuid>",
  "state": "complete",
  "revision": 1,
  "updated_at": "2026-09-20T21:00:00Z",
  "last_error": null,
  "payloads": [
    {"path": "geometry.sqlite3", "size_bytes": 12345, "sha256": "<digest>"},
    {"path": "tracks.npz", "size_bytes": 6789, "sha256": "<digest>"}
  ]
}
```

Artifact IDs are opaque IDs reserved before computation and immutable after
publication. They are not filesystem paths or hashes of configuration alone.
Payload hashes verify content; existing selection/configuration hashes remain useful
cache keys but do not replace artifact IDs. `schema_version` versions the manifest
envelope; capabilities declare optional payload fields such as reconstruction mappings.

Use the direct immutable chain: matches -> geometry -> reconstruction. Matching is the
self-contained root and records its feature-cache snapshot origin.
Paths are locators and may change without changing IDs. Readers validate ancestry
before joining data. Include `matching_artifact_id` in downstream metadata as a
convenient, validated reference to the observation namespace.

Resume may extend an unpublished matching or geometry draft. A published result is
immutable; changed inputs or settings require a new output and artifact ID. An identical
completed rerun validates and returns the existing artifact.

### Common lifecycle and solve progress

Use `state` for lifecycle and `phase` for stage-specific work. Shared states:

| State | Meaning |
| --- | --- |
| `starting` | Job allocated; input validation or worker startup in progress |
| `running` | Worker is doing work; inspect `phase` for detail |
| `paused` | Explicitly suspended with resumable stage state retained |
| `canceled` | User ended this job; partial data may remain but no result was published |
| `failed` | Job could not finish; retain structured error and partial data |
| `complete` | The declared result is published and all referenced payloads are readable |

Do not overload phases such as `verifying` or `associating_tracks` as lifecycle
states. Suggested solve phases are `loading_inputs`, `selecting_seed`,
`registering_frames`, `triangulating`, `bundle_adjusting`, `filtering`, and `publishing`.
Canceled, failed and complete jobs are terminal; retry creates a new job with a
`resume_from_job_id` or `retry_of_job_id` reference when appropriate. Paused jobs may
resume with the same job ID and increasing revision. Solving currently has no resume
checkpoint: stopping it must not claim a resumable `paused` state.

Each status has integer `revision`, UTC `updated_at` and nullable `last_error`.
Use a structured error (`code`, `message`, optional log reference). Revisions increase
on every published status change, including snapshot references and terminal states;
they never reset when resuming a paused job. Coalesced progress updates are allowed.
They are scoped to one job/status document, not comparable across jobs or with an
artifact's separate manifest revision. A watcher deduplicates by `(job_id, revision)`.

Example `jobs/<job-id>/status.json`:

```json
{
  "schema_version": 1,
  "job_id": "job-<uuid>",
  "output_artifact_id": "reconstruction-<uuid>",
  "state": "running",
  "phase": "registering_frames",
  "registered_frames": 74,
  "total_frames": 300,
  "points": 18342,
  "current_frame": 81,
  "bundle_adjustment_iteration": null,
  "latest_snapshot": null,
  "revision": 96,
  "updated_at": "2026-09-20T21:00:00Z",
  "last_error": null
}
```

`current_frame` is the original matching row, not video frame index or compact
solution row; use null when no single frame is active. `points` counts current live
landmarks, not all points ever allocated. Report BA iterations only if the optimizer
actually exposes them; do not relabel residual-function evaluations as iterations.
Null remains valid when that progress is unavailable. Provide a `solve-status` CLI
and reader `status()` using the same schema, independent of logs/progress bars.

Update status atomically at phase changes and bounded progress intervals. A heartbeat
can advance status during a long optimizer call. The supervisor must reconcile worker
exit, cancellation and crashes using process identity/locks and exit status; a stale
timestamp alone is not proof of failure. Existing `running`/`phase` status endpoints
need an explicit compatibility adapter rather than silently changing their meanings.

### Jobs, invocation and optional snapshots

Target layout for the remaining job-workspace work:

```text
workspace/
  jobs/<job-id>/
    job.json
    status.json
    stdout.log
    stderr.log
    draft/
    snapshots/
      000001.npz
      000002.npz
  artifacts/<artifact-id>/
    manifest.json
    ...published payloads...
```

Keep mutable work, logs and status in the job directory. Final artifacts are immutable.
`job.json` records the executed argument vector as a string array, resolved interpreter,
working directory, effective options including defaults, relevant allowlisted environment
and runtime provenance (Python, packages, GPU, matcher weights/revision), input artifact
IDs/locators and intended output ID/location. Record creation/start/end timestamps,
exit code or termination signal, and attempt lineage. Do not persist credentials or
the complete inherited environment. Update mutable job metadata by temporary-file
replacement; a rerun creates a new job record rather than editing the old invocation.

Snapshots are optional immutable previews, **not resume checkpoints**. Capture a
consistent solver state at safe boundaries, with configurable cadence and point-count
limits to avoid blocking computation on frequent large writes. Each NPZ carries the
job ID, input artifact IDs, snapshot revision, capture phase, K, frame mappings, poses,
registered mask, points, colors, stable point/source-track IDs, observations and
residuals when available. Include a residual-validity mask and NaN for unknown errors;
do not substitute zero. Any sampling must be declared, with remapped observation rows.

Write and close a temporary snapshot, atomically rename it to its immutable final
filename, and only then publish a status revision referencing its path, revision and
capture time. Filenames identify monotonically increasing snapshot revisions, which
can be less frequent than status revisions. Status exposes the latest reference;
the GUI never has to guess completeness by watching NPZ file size. Point positions
may change after BA while point IDs persist. An incomplete seed attempt is provisional
and must be marked as such rather than presented as the accepted reconstruction.

### Publication invariants

These are implementation requirements:

1. Finish and close payloads before a manifest/status references them. JSON updates
   use a temporary file in the same directory followed by atomic replacement.
2. Completed artifacts never change. Late optional exports such as Rerun recordings
   belong in a separate export location or derived artifact; do not append them to a
   sealed artifact. They need not block a core solve result from becoming complete.
3. Active SQLite drafts may append committed pairs, but existing pair payloads do not
   mutate. Final publication must use a consistent SQLite backup/checkpoint with no
   dependence on a live WAL/SHM sidecar. Copying only an active main DB file is invalid.
4. Stage the final payloads and manifest on the destination filesystem, validate their
   inventory/hashes, then publish the artifact directory atomically without overwriting
   an existing ID. Explicit flush/fsync handling is needed for crash durability;
   atomic visibility by itself is not a power-loss guarantee.
5. Publish the artifact before changing job status to `complete`. If a crash occurs
   between those steps, recovery validates the existing artifact and reconciles status.
   A published valid artifact cannot then be relabeled as a failed core computation.
6. Failed/canceled computations never publish a complete artifact. Retained draft
   results and snapshots stay under the job directory and remain labeled as partial.
7. `complete` guarantees all files referenced by the manifest are readable, even for
   a declared partial selection or a reconstruction with unregistered frames.

### Reader facade

Provide a versioned facade over current storage, with capabilities by artifact type:

| Method | Required behavior |
| --- | --- |
| `manifest()` | Identity, ancestry, schema/capabilities and payload inventory |
| `list_frames(after=None, limit=...)` | Page stable frame records with matching-row and original-frame IDs |
| `frame(row)` | Preview, coordinates and metadata using matching row |
| `list_pairs(after=None, limit=...)` | Page canonical pair keys/diagnostics with an opaque continuation cursor |
| `pair(first, second)` | Decode matches or geometry; document canonical ordering and pose direction |
| `status()` | Read one coherent status revision; expose worker liveness separately |
| `load_snapshot(revision=None)` | Load a specified immutable snapshot, or the latest referenced snapshot |
| `load_reconstruction()` | Decode final arrays with source mappings and schema validation |

For live pair readers, the cursor must track append/commit order, not lexicographic
`(first, second)` order: workers finish out of order and late smaller keys must not
be skipped. Return `items`, `next_cursor`, and a `data_revision` captured in the same
read transaction. Writers advance this draft-data revision transactionally with each
visible committed batch; job status carries the latest reported `data_revision`.
It is distinct from the JSON status revision and may advance between status updates.
Polling an empty page must preserve a cursor that can discover later commits. Frame
pagination can use the fixed frame order. Scope every cursor to its artifact/job,
and validate that scope on reuse. Keep read transactions short and never mutate data
from a reader.

`load_snapshot(None)` reads status first and opens the referenced file; if none has
been published it returns no snapshot. Readers reject unsupported schema versions,
missing manifests, invalid payload hashes and mismatched ancestry.

`ArtifactReader` is the supported GUI boundary for published matching, geometry and
reconstruction artifacts. SQL tables, zlib layouts, internal mapper objects and
progress-bar text are implementation details. Pagination of immutable published frames
and pairs is implemented; live draft cursors, `data_revision`, and snapshots remain planned.

```python
from pathlib import Path
from slam_lab.reader import ArtifactReader

reader = ArtifactReader(Path("recordings/osaka-lightglue-solver-v5"))
status = reader.status()
reconstruction = reader.load_reconstruction()
point_row = 0
stable_point_id = reconstruction["point_ids"][point_row]
source_track_id = reconstruction["source_track_ids"][point_row]
```
