# Initial monocular reconstruction

This document describes the **legacy `reconstruct` command** and the earlier Osaka
tracking investigation. It does not consume the exhaustive match stores. Use
[SOLVER.md](SOLVER.md) for the current `verify-matches -> solve` pipeline and
[SESSION_HANDOFF.md](SESSION_HANDOFF.md) for current results and remaining work.

The pipeline now reconstructs a sparse 3D map and camera poses directly from a
SuperPoint cache. It uses only the RGB camera. Depth, IMU, ground truth, and the
original video are not needed. This is an incremental structure-from-motion
baseline, with no loop closure yet.

## Run and inspect

```bash
source .venv/bin/activate

# Already-installed dependencies in this workspace; on another machine:
uv pip install -e '.[reconstruction]'

# Osaka: the cached 300-frame / 30-second preview.
# Limit BLAS threads to avoid overhead in sparse optimization on many-core CPUs.
OPENBLAS_NUM_THREADS=1 slam-lab reconstruct \
  .slam-cache/7065fae5d5e9bd54ae4e14d2d5f051960dfe2d14266d89d5e6771fa037786216 \
  --output recordings/osaka-reconstruction --fov-deg 60 --max-frames 300

rerun recordings/osaka-reconstruction/reconstruction.rrd

# Reopen or re-export a solved run without matching/optimization.
slam-lab view-reconstruction recordings/osaka-reconstruction
slam-lab view-reconstruction recordings/osaka-reconstruction --save recordings/another-view.rrd
```

Choose a new `--output` directory for each experiment; existing results are never
overwritten. `--max-frames` defaults to 300 selected cached frames. `--frame-step 3`
uses every third cached frame. This offline implementation loads selected features
into memory (roughly 2 MiB of descriptors per frame at 2,048 features), so use
bounded segments initially. Interrupted reconstructions can be rerun from the
unchanged feature cache; reconstruction itself does not yet checkpoint/resume.

Rerun shows the final colored map and trajectory, a moving camera frustum with its
image, current tracked observations, tracking-support curves, and per-frame status.
The final map is static: scrubbing shows the refined result, not the map as it
existed during incremental estimation. Missing poses create visible trajectory
gaps and clear the moving camera instead of retaining a stale transform.

The prepared Osaka run (`recordings/osaka-reconstruction`) uses the first 300
cached frames, covering approximately 30 seconds at 10 fps. It registered 299
cameras and retained 11,143 points with 142,673 observations. The black opening
frame has no pose. Median reprojection error is 0.75 px and the 95th percentile
is 1.99 px. All retained observations passed positive-depth and 3 px residual
checks; every point has at least 3 observations. Bundle adjustment improved the
fit but reached its 30-evaluation budget before convergence. These are internal
consistency measurements, not ground-truth trajectory accuracy; the initial
trajectory still needs calibration and drift evaluation. `validation.json` in
that run records an independent check of exported poses, projections, and scale.

## Fixed camera assumptions

The intrinsic matrix is

```text
K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]
```

By default, `fx = fy = width / (2 tan(horizontal_FOV / 2))`, with a **guessed**
60-degree horizontal field of view. The principal point is the image center,
`cx = (width - 1) / 2`, `cy = (height - 1) / 2`; all distortion coefficients,
including radial `k1` and `k2`, are zero. Intrinsics are held fixed throughout
reconstruction and bundle adjustment. This does not calibrate the camera.

`--fx`, `--fy`, `--cx`, and `--cy` override these parameters in **cached preview
pixels**, not original video pixels. `--fy` defaults to `--fx`. For intrinsics
calibrated at the original resolution, apply the cache resize convention:

```text
fx_cache = fx_original * cache_width / original_width
fy_cache = fy_original * cache_height / original_height
cx_cache = (cx_original + 0.5) * cache_width / original_width - 0.5
cy_cache = (cy_original + 0.5) * cache_height / original_height - 0.5
```

For Osaka's 1024×576 previews, the default is approximately
`fx = fy = 886.81`, `cx = 511.5`, `cy = 287.5`. Varying `--fov-deg` is a useful
experiment, but reprojection error alone cannot establish correct calibration or
metric accuracy. Zoom, digital stabilization/cropping, rolling shutter, and lens
distortion can violate the fixed pinhole assumption.

## Estimation loop

1. **Matching:** L2 nearest neighbors on cached SuperPoint descriptors, a ratio
   test in both directions (default 0.8), and mutual consistency. No descriptor
   equality or fixed keypoint index is assumed. References include the last
   registered frame and a small window of recent keyframes.
2. **Initialization:** five-point essential-matrix RANSAC, pose recovery, positive
   depth, and triangulation. Given fixed `K`, the calibrated relation is
   `E = Kᵀ F K`. We estimate `E` directly. A seed needs at least 30 good points,
   at least 25% surviving matches, and a median triangulation angle of at least
   2 degrees. Pairs explained almost entirely by a homography are rejected to
   avoid unreliable initialization from pure rotation or planar scenes.
3. **Tracking:** matches to existing landmarks yield 2D–3D correspondences.
   Conflicting landmark identities and many-to-one associations are rejected.
   PnP RANSAC estimates each new camera in the existing map's scale, followed by
   inlier-only pose refinement. Acceptance requires at least 20 inliers, 25%
   support, and features spanning at least 10% of each image dimension.
4. **Map extension:** triangulate previously unmapped matches using the solved
   cameras. Both depths must be positive, both reprojection errors at most 3 px,
   and the triangulation angle at least 1 degree. Existing landmarks acquire
   additional observations after the same reprojection/depth checks. Earlier
   frames are tracked backward through neighboring views, extending the seed map
   along the way; this avoids solving early frames only against distant seed views.
5. **Refinement:** sparse robust bundle adjustment jointly refines poses and
   landmarks with fixed intrinsics. The first seed pose is fixed and the seed
   baseline length is constrained to remove the coordinate/scale ambiguity. The
   result is rescaled to exactly preserve the seed baseline, without changing
   any image projections; this removes drift allowed by the soft constraint.
   A soft-L1 loss reduces outlier influence. Final filtering removes observations
   over the reprojection threshold or behind cameras and keeps landmarks with
   at least 3 observations (2 for a reconstruction containing only 2 cameras).

Descriptors can vary across a track because each observation retains its own
descriptor. A landmark ID links the observations. The first matcher is deliberately
simple. LightGlue is now implemented in the separate all-pairs stage; its saved
matches feed the modular solver, not this legacy command. No feature re-extraction
is needed. An appearance change or occlusion may still break a track;
there is no global place-recognition/relocalization stage yet.

RANSAC runs in both initialization and camera tracking. New triangulation uses
already-estimated cameras plus reprojection, depth, and angle gates. Pairwise unit
translations are never concatenated to make a trajectory: that would discard the
relative scale between successive camera movements.

## Output files

| File | Contents |
| --- | --- |
| `reconstruction.rrd` | Self-contained Rerun recording |
| `poses.json` | Every selected frame, timestamp, status, support, and camera-to-world pose or `null` |
| `points.ply` | Colored sparse point cloud in the same world coordinates |
| `reconstruction.npz` | Poses, points, colors, camera matrix, tracks, observations, residuals |
| `metadata.json` | Input identity, camera assumptions, settings, seed pair, quality and optimization report |
| `matching.json` | Attempted frame pairs and descriptor-match counts |
| `ransac.npz` / `ransac.json` | Exact solver masks, frozen PnP inputs, pre-BA poses, seed decisions |

In the NPZ, `T_world_camera[i]` transforms camera coordinates into world
coordinates: `X_world = R_world_camera @ X_camera + t_world_camera`. Camera axes
are right, down, forward. The translation column is the camera center in world
coordinates. Failed frames have `registered[i] = false` and a NaN matrix; JSON
uses `null`. `frame_indices` preserves the original video indices, and
`timestamps_ns` preserves the feature-cache timing.

Each row of `observations` is `(selected_frame_row, landmark_id, keypoint_index)`.
It aligns with `observation_xy` and `reprojection_errors`. Initialization and
matching frame identifiers also refer to selected-frame rows; use `frame_indices`
to convert to original video indices. Track lengths count retained observations.

**All geometry has arbitrary scale.** The seed baseline is one unit, not one
meter. A known distance or another source of scale is needed for metric output.

## Inspect RANSAC decisions

New reconstructions capture the actual inlier indices returned by
`solvePnPRansac`, before pose refinement or bundle adjustment. The default 2D
view now shows **green inliers and red rejected candidates**. Gray means the
candidate was not tested because no PnP solve ran. These are 2D–3D candidates,
not all SuperPoint detections or all descriptor matches. A seed frame uses the
essential-matrix solver instead, and is displayed in the two seed tabs.

```bash
rerun recordings/osaka-ransac/reconstruction.rrd
# Scrub video_time to approximately 10 seconds.
```

The tabs separate different stages:

- **PnP RANSAC:** the exact raw solver mask; click points for their raw/refined
  reprojection errors, feature index, incremental landmark ID, and final map ID.
- **Fit residuals:** observed-to-projected lines after inlier-only LM refinement.
  Segments with projections outside the image are omitted for legibility; the
  point attributes still retain those errors.
- **Retained tracks:** observations that survived final bundle adjustment and
  filtering. This was the only overlay in the original recording; it cannot be
  used to infer the earlier RANSAC mask.
- **Seed first / second:** the two static initialization images with the original
  essential-matrix RANSAC mask. A point's `triangulated` attribute shows whether
  it also passed the subsequent depth, parallax, and reprojection tests.

The 3D view compares **cyan before bundle adjustment** with **yellow after bundle
adjustment**, using the same seed coordinate frame and unit baseline. This helps
separate a tracking error from a later optimization change. Counts in
`metrics/ransac` report candidates, raw inliers, raw outliers, and refined inliers
separately. The older `pnp_inliers` field means the post-LM acceptance count;
`pnp_ransac_inliers` is the raw solver count.

`ransac.npz` uses keys such as `pnp/100/xy`, `pnp/100/ransac_inliers`,
`pnp/100/refined_inliers`, and `pnp/100/points`, indexed by selected-frame row.
The 3D inputs are frozen at pose-estimation time, so recomputing their projection
with `raw_pose` or `refined_pose` reproduces the recorded residuals. Their
incremental landmark IDs differ from the compacted final cloud IDs;
`final_landmark_ids` supplies that mapping, with -1 for a removed point.

Older runs remain viewable, but did not save RANSAC masks. They require a new
reconstruction from cached descriptors to recover actual solver decisions;
the viewer does not relabel final tracks as historical inliers. An inlier is
consistent with the fitted model, not proof that a feature is static. Moving
people, screen-fixed captions, bad landmark depths, and incorrect intrinsics
can all require investigation even when the raw inlier count is high.

### Osaka diagnostic finding

The instrumented `osaka-ransac` run exactly reproduces the original poses, points,
observations, and residuals. It is a diagnostic recording, not a trajectory fix.
At 10.01 s, the raw PnP mask accepts 426 of 644 candidates; 356 accepted candidates
are above image y=240, primarily in the building/background region. The bend is
already present in the trajectory before bundle adjustment.

Initialization occurs at 13.5135 and 14.014 s and reconstruction then tracks
backward through the earlier frames. The raw essential-matrix mask accepts 63
matches in the lower-left on-screen caption region. In a controlled seed-pair
experiment with fixed settings and RNG seed, excluding that region changes the
relative camera-center direction from approximately `[-0.903, 0.316, 0.290]` to
`[0.054, -0.061, 0.997]` (first camera's right/down/forward coordinates).
The excluded-caption pair no longer passes the initialization gates. This is
strong evidence of caption contamination and unstable seed geometry; it does not
establish the caption as the sole cause or supply a corrected trajectory.
`diagnosis.json` and `seed-ablation.json` in the run record these checks. Screenshots
`ransac-10s.png` and `seed-ransac.png` show the actual accepted/rejected observations.

## Iteration knobs and limits

| Option | Default | Purpose |
| --- | --- | --- |
| `--fov-deg` / `--fx --fy` | 60° / derived | Fixed camera focal length |
| `--ratio` | 0.8 | Descriptor ambiguity rejection |
| `--ransac-px` | 1.5 | Essential-matrix inlier threshold |
| `--reprojection-px` | 3 | PnP, triangulation, final observation threshold |
| `--min-parallax` | 1° | Minimum angle for triangulated points |
| `--min-track-length` | 3 | Minimum retained observations per point |
| `--bundle-evaluations` | 30 | Optimization budget; 0 disables it |
| `--seed` | 7 | OpenCV RANSAC random seed |

The optimization report distinguishes improvement from convergence: reaching the
evaluation budget can still improve a solution, but is reported as not converged.
The implementation assumes a mostly static scene. Moving crowds, repetitive
facades, video cuts, weak translation, and incorrect intrinsics can cause drift or
tracking failure. RANSAC cannot guarantee that its largest consensus is the static
background. A low residual is a consistency check, not ground-truth validation.

Useful next comparisons are LightGlue versus the descriptor baseline, focal-length
sensitivity, local bundle adjustment while mapping, wider track recovery, and
loop closure. This version provides the poses, tracks, errors, and map needed to
measure those changes.

Geometry follows [OpenCV's camera and reconstruction APIs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
Refinement uses [SciPy sparse robust least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).
