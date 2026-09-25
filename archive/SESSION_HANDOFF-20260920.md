# SLAM Lab session handoff — 2026-09-20

This is the verified state at the end of the session, not a live status file.
Start with [WORKFLOW.md](WORKFLOW.md) for commands. The project directory is
`/home/nick/slam-lab`; all paths below are relative to it unless stated otherwise.

## Outcome

The lab now has an offline, modular monocular RGB pipeline:

```text
video -> cached SuperPoint features -> exhaustive appearance matches
      -> cached pairwise F/E/H verification -> verified tracks
      -> graph-based camera/point solver -> final joint BA and filtering
```

Both exhaustive Osaka matching runs are finished, including global appearance-track
association and Rerun export. CUDA LightGlue works on the RTX 2080 Ti. The first
modular solver has produced comparable 30-frame cosine and LightGlue experiments.

The backend remains file-based and Rerun is optional presentation. The first GUI
integration slice—strict artifact ancestry, stable selection lineage and structured
solver status—is implemented. Solve provenance and the published-artifact reader are
also implemented; job snapshots and live draft pagination remain planned.

## Data and environment

| Item | Verified value |
| --- | --- |
| Input | `/home/nick/Downloads/osaka.mp4`, 1920x1080, about 59.94 fps |
| Selected clip | 300 frames at stride 6, timestamps 0–29.9299 seconds |
| Preview/features | 1024x576, up to 2,048 SuperPoint features/frame |
| Observations | 612,352; matching row 0 is black and has no features |
| Feature cache | `.slam-cache/7065fae5d5e9bd54ae4e14d2d5f051960dfe2d14266d89d5e6771fa037786216/cache.sqlite3` |
| CPU environment | `.venv`, Python 3.12, PyTorch 2.14.0+cpu |
| GPU environment | `.venv-cuda`, Python 3.12, PyTorch 2.11.0+cu128 |
| GPU verified during execution | NVIDIA RTX 2080 Ti, 11 GB; host driver supports this runtime |
| Viewer | Rerun SDK/CLI 0.38.1 |
| Camera used by new solver | fx=fy=886.8100134752652, cx=511.5, cy=287.5; zero distortion |

The camera focal length is derived from an assumed 60-degree horizontal FOV, not
calibrated from this video. Camera intrinsics remain fixed while poses and points
are optimized. The result has arbitrary scale, with the seed baseline set to 1.

The feature cache has `complete=false` because only the requested first 300 selected
frames of a much longer source were extracted. This is intentional; the matching
runs are complete for that selection. Do not infer a failed extraction from that flag.

Model weights are already cached in `.slam-cache/models/checkpoints/`. LightGlue
implementation revision: `eb42fee2d71449efb0aa5c10549752b5d75384d8`.
SuperPoint-LightGlue weight SHA-256:
`6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a`.
Detailed feature/model provenance is stored in the cache and match databases.

## Completed runs

| Run directory under `recordings/` | State and result |
| --- | --- |
| `osaka-allpairs-cosine` | Complete: 44,850 pairs, 12,471,433 match edges, 28,778 tracks of length >=2 |
| `osaka-allpairs-lightglue` | Complete: 44,850 pairs, 22,111,491 match edges, 15,641 tracks of length >=2 |
| `osaka-cosine-geometry-step10` | Complete: 30 selected images, 435 pairs, 78,714 verified edges, 103 eligible seed pairs |
| `osaka-cosine-solver-v1` | Complete: 29 registered cameras, 1,751 points, 11,046 retained observations |
| `osaka-lightglue-geometry-step10` | Complete: 30 selected images, 435 pairs, 139,397 verified edges |
| `osaka-lightglue-solver-v1` | Complete: 19 registered cameras, 1,933 points, 10,483 retained observations |
| `osaka-lightglue-solver-v5` | Strict chain, exact solve provenance and unified artifact-reader validation |

Both matching status readers report `running=false`, `phase=complete`. Geometry
also reports complete/not running. There is no unfinished matching/geometry job
from this session to resume. Background PID/log files may remain as historical records;
the process lock/status is authoritative, not the existence of a `.job.json` file.

LightGlue preserved **2,807 CPU pairs** and computed **42,043 CUDA pairs**. Its last
reported matching rate was **47.8 pairs/s**. The CUDA resume began at 20:30:21 UTC and
completed track association at 20:45:24 UTC, about **15 minutes** later; the Rerun
export was also produced. A small warmed-up GPU benchmark measured 19–41 ms/pair.
The earlier multi-hour estimate applied to CPU execution, not this GPU run.

Cosine matching used unique mutual best normalized descriptor matches at similarity
>=0.8, without a ratio test. LightGlue used threshold 0.1 and all cached features.
Counts of appearance edges or tracks do not establish which reconstruction is better.

The solver sample uses matching rows 0, 10, ..., 290 (about 1 fps over 30 seconds),
not the first 30 consecutive frames. The blank row 0 is the only unregistered image.
The seed uses matching rows **210 and 230**, checked against row **220** as a third view.
BA converged in 29 evaluations. Final median reprojection error is **0.817 px**,
p95 **2.090 px**. These are image-fit statistics; the trajectory is not ground-truth
validated and must not be presented as a resolved accuracy problem.

Useful saved outputs:

* `osaka-allpairs-lightglue/correspondences.rrd` — complete appearance tracks (~58 MB).
* `osaka-allpairs-cosine/correspondences.rrd` — complete cosine tracks (~57 MB).
* `osaka-cosine-geometry-step10/pair-100-110.rrd` — geometry masks near 10 seconds.
* `osaka-cosine-solver-v1/reconstruction.rrd` — poses, points and solver diagnostics.
* `osaka-cosine-solver-v1/trajectory-check.png` — static trajectory inspection plot.
* `osaka-cosine-solver-v1/{metadata.json,poses.json,reconstruction.npz,points.ply,ransac.npz}`
  — numeric results, provenance and raw RANSAC diagnostics.

## Implementation and decisions

| Module | Responsibility |
| --- | --- |
| `cache.py`, `pipeline.py`, `extractor.py` | Timestamped previews and cached SuperPoint features |
| `matching.py` | LightGlue, cosine mutual NN and L2 mutual NN matcher adapters |
| `correspondence.py` | Resumable pair store, exhaustive scheduler, appearance tracks, runtime provenance |
| `match_jobs.py` | Resume argument recovery and chained background jobs |
| `verification.py` | F/E/H estimation, raw masks, pair diagnostics, verified tracks, geometry status |
| `solver.py` | Descriptor-free `ReconstructionProblem` and graph-based SfM backend |
| `geometry.py`, `bundle.py` | Pose/triangulation primitives and fixed-intrinsic sparse robust BA |
| `reconstruction.py` | Earlier video-order mapper plus reused state/final-filtering machinery |
| `reconstruction_io.py`, `ransac_view.py` | Reconstruction artifacts and inspection adapters |
| `match_view.py`, `geometry_view.py`, `rerun_support.py` | Pair/track viewers and environment-local Rerun discovery |

The supported command path is **`match-all -> verify-matches -> solve`**.

Verification separately fits F (USAC/MAGSAC), E (calibrated RANSAC) and H (RANSAC).
Raw masks are not overwritten by pose recovery or final filtering. OpenCV can throw
on degenerate inputs; those estimator failures are captured per pair. Verified
track edges require F and E agreement with enough pair support. Track association
enforces at most one feature from each image and orders edges by geometric residual.

Seed selection ranks geometric support, spatial coverage and parallax, rejects
homography-dominated seeds, and checks a third view. The graph solver uses all
available landmark tracks to register each next image with PnP, triangulates points,
then performs final joint BA and observation filtering. Unit translations from E
are never concatenated as equal-sized camera steps.

The output pose convention is `T_world_camera`, camera-to-world, with camera axes
right/down/forward. Internal poses map world-to-camera. Matching rows, original video
frame indices and compact solution rows are different IDs. The interface document
spells out their mappings and the proposed improvements.

## Earlier debugging findings worth preserving

The original bad trajectory was not proven to be simply insufficient match count.
The earlier RANSAC investigation found strong caption contamination in its seed
and a large pose-direction change when that region was excluded. The bend near
10 seconds existed before BA. See the historical [RECONSTRUCTION.md](RECONSTRUCTION.md)
and `recordings/osaka-ransac` diagnostics; they are not the new solver's results.

The apparent missing building matches in pair 100/101 were partly a display issue:
the original viewer capped links at 150. That LightGlue pair actually has 1,608
matches. The viewer now defaults to all links and has a separate matched/unmatched
coverage tab. Appearance coverage colors are not RANSAC inlier/outlier colors.

CUDA was initially hidden by the execution sandbox, while the CPU-only PyTorch
environment also could not use it. Checking on the host established that the GPU
and driver were available. A separate CUDA environment fixed inference without
replacing the CPU environment. GPU attention may use FP16 internally, so do not
promise bit-identical CPU/GPU matches or describe CUDA inference as entirely FP32.

## GUI handoff and next priorities

The GUI should read [WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md) for the current
artifact and status contract. No HTTP service is implemented. Core Python compute
functions do not launch Rerun. The supported compute workflow is
`match-all -> verify-matches -> solve`; `match-all` and `solve` accept `--no-rerun`.

The first P0 slice is implemented: matching, geometry and reconstruction manifests;
strict payload/parent validation; canonical observation lineage; stable point IDs; and
atomic `solve-status`. Unmanifested artifacts are rejected rather than adapted.

Next add full job invocation provenance, optional immutable live NPZ previews, crash
reconciliation and the versioned paginated reader facade. Snapshot previews are not
resume checkpoints. Live pair cursors must follow commit order so out-of-order worker
results are not missed.

For the next numerical experiment, compare LightGlue and cosine on the same
30-frame selection using the commands in [WORKFLOW.md](WORKFLOW.md). After visual
and geometric review, consider a mature solver adapter, calibration sensitivity,
local/periodic BA, track splitting and dynamic/caption masks. A new custom global
optimizer is not the immediate workbench dependency.

## Limitations and operational notes

* The prototype solves one seed-connected component with final-only BA. It has no
  track splitting, global SfM backend or automatic dynamic-object exclusion yet.
* Intrinsics are assumed; moving people, planar structures and captions can produce
  plausible inliers and low residuals with incorrect poses.
* `solve` has structured persistent progress but no checkpoint/resume. Its final artifact
  directory is published atomically and bound to its geometry parent.
* `--resume-after` inherits the launcher's Python interpreter. Launch from `.venv-cuda`
  if the queued job needs CUDA; automatic interpreter selection is not implemented.
* Existing solved outputs are not overwritten. Use a new directory for each solve.
* Phone remote access was discussed but not configured as part of this project work.
* This workspace exposes an empty read-only `.git` directory; no commit or PR was
  created in this session. Preserve the working files and local artifacts themselves.

## Validation and restart note

The last implementation validation passed **60 tests** and Ruff lint checks. It
covers synthetic known cameras and scale, outliers, pure rotation, planar and weak
baseline rejection, raw mask preservation, runtime migration, immutable source
matches, resume/input validation, subsampling IDs and self-contained Rerun export.
Subsequent changes were documentation only. This handoff rechecked saved run status,
metadata and output existence; it did not rerun inference or launch another solve.

Suggested next-session instruction:

> Read WORKFLOW.md, WORKBENCH_INTERFACE.md and SOLVER_TODO.md. Continue after completed
> P0 slice 1 with live snapshots, crash reconciliation and live draft pagination.
