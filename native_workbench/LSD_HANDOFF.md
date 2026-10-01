# LSD-SLAM pipeline: handoff (2026-10-01)

Paused mid-way through step 4 (keyframe graph). Read this first, then
[LSD_SLAM.md](LSD_SLAM.md) for the design, the algorithm summary and every
measurement so far.

## What exists

A second, independent pipeline in the native workbench: direct monocular SLAM
after LSD-SLAM (Engel, Schöps, Cremers, ECCV 2014) with the semi-dense depth
filter of Engel, Sturm, Cremers (ICCV 2013). It shares the decoder, launcher,
3D renderer, depth network, SuperPoint engine and the Sim(3) pose-graph
optimiser with the SuperPoint workbench, and nothing else.

| Part | Files | State |
|---|---|---|
| Image pyramid (2×2 average, gradients, validity mask) | `src/lsd/image.*` | done |
| Undistortion (OpenCV k1 k2 p1 p2 k3 from `calibration.json`), eroded mask | `src/lsd/undistort.*` | done |
| SE(3) direct tracker, variance-weighted (paper eq. 12–14) | `src/lsd/se3_tracker.*` | done, tested |
| Semi-dense depth filter: epipolar stereo, Kalman fusion, propagation, regularisation, validity | `src/lsd/semi_dense_depth.*` | done, tested |
| Odometry: keyframes, depth-network prior + scale lock, Sim(3) keyframe poses, frames stored relative to their keyframe | `src/lsd/odometry.*`, `src/lsd/keyframe.hpp` | done |
| Sim(3) direct keyframe alignment with depth residual (eq. 17–19), adjoint, reciprocal check (eq. 20) | `src/lsd/sim3_tracker.*` | done, tested |
| Keyframe graph: constraint search (parent + nearby), covariance-weighted Sim(3) optimisation | `src/lsd/keyframe_graph.*`; `src/pose_graph.*` gained `sqrt_information` per edge | **implemented, makes fr2_desk worse — see below** |
| Loop candidates: SuperPoint on keyframes, pooled-descriptor shortlist, mutual-NN matching, 3D–3D similarity RANSAC initialisation | `src/lsd/place_recognition.*` | **implemented, never measured** |
| GUI `slam-native-lsd-workbench`: launcher (video-only mode), video + reprojected depth, keyframe depth views, 3D pose graph | `src/lsd_app.*`, `src/lsd_graph_view.*`, `src/lsd_main.cpp`, `../run-lsd-workbench.sh` | done (single-threaded) |
| Headless runner: screenshot, trajectory CSV, depth / edge / pair exports | `tests/snapshot_lsd.cpp` (`slam-native-lsd-snapshot`) | done |
| Evaluation | `tools/tum_eval.py` (ATE/RPE), `tools/lsd_depth_vs_gt.py`, `tools/lsd_edges_vs_gt.py` | done |
| Tests (CTest) | `tests/test_lsd_tracker.cpp`, `test_lsd_depth_filter.cpp`, `test_lsd_sim3.cpp`, scene `lsd_scene.hpp` | passing |

Shared-code changes: `Launcher::set_video_only()`, `read_calibration()` /
`read_distortion()` moved to `include/slam_native/calibration.hpp` (the
SuperPoint app uses it too), `PoseGraphEdge::sqrt_information` (default
identity, existing callers unchanged).

## Running and measuring

```bash
./run-lsd-workbench.sh                       # GUI, recent-video grid
native_workbench/build/app-debug/slam-native-lsd-snapshot VIDEO native_workbench/models FRAMES out.ppm traj.csv [DEPTH_VIEW]
.venv-cuda/bin/python native_workbench/tools/tum_eval.py data/datasets/tum/rgbd_dataset_<seq> traj.csv
```

`slam-native-lsd-snapshot` is deterministic (everything runs inline) and reads
environment switches:

| Variable | Effect |
|---|---|
| `LSD_STEREO=0` | network depth per keyframe only (no stereo, no propagation) |
| `LSD_GRAPH=0` | no Sim(3) constraints / optimisation |
| `LSD_LOOPS=0` | no SuperPoint loop candidates |
| `LSD_UNDISTORT=0` | ignore lens distortion |
| `LSD_SIGMA_L`, `LSD_SIGMA_I`, `LSD_PRIOR_SIGMA` | stereo noise model, prior sigma |
| `LSD_RECIPROCAL`, `LSD_CONSTRAINT_LEVEL` | eq. 20 threshold, Sim(3) alignment's first pyramid level |
| `LSD_EXPORT_DEPTH=f.csv` | every keyframe's depth (+ prior) for `lsd_depth_vs_gt.py`. **~100 MB per run: delete after use** (the scratch quota filled once) |
| `LSD_EDGES=f.csv` | accepted graph edges for `lsd_edges_vs_gt.py` |
| `LSD_PAIRS=f.csv`, `LSD_PAIR_DETAILS=f.csv` | every tried constraint pair: eq. 20 error and physical disagreement; ids and deviation from the initial guess |

Full sequences take ~4 min each (about 100 ms/frame).

## Where the numbers stand

ATE after sim3 alignment (`tum_eval.py`), 640×480, defaults:

| Run | ATE | Note |
|---|---|---|
| fr1_xyz 0–299 | 1.33 cm | graph on (1.46 off, 2.18 network depth only) |
| fr1_desk 0–199 | 2.20 cm | graph on (2.35 off, 3.13 network depth only) |
| fr2_desk full | 24.5 cm graph off; **29.2 cm graph on** | paper 4.52 cm (with loop closure) |
| fr2_desk full, stereo off | 90.4 cm | the depth filter is essential |
| fr3_long_office full | 41.4 cm graph off | graph-on run was stopped unfinished |

The fr2_desk graph-on run predates the loop-candidate code (the binary was
rebuilt while it ran), so loops have not been measured anywhere.

## The open problem: the graph hurts fr2_desk

With the graph on, fr2_desk got worse (24.5 → 29.2 cm ATE, RPE 5.6 → 7.4
cm/s and 2.3 → 3.8°/s): 89 alignment edges, 3 fallback parent edges, 372
pairs tried (eq. 20 error 10/50/90 %: 1.3k / 107k / 24M). Short clips
improved, so the mechanics work; something lets error in over a long run.
Next step, already prepared:

```bash
LSD_EDGES=edges.csv slam-native-lsd-snapshot .../rgbd_dataset_freiburg2_desk/rgb.mp4 native_workbench/models 100000 x.ppm traj.csv
.venv-cuda/bin/python native_workbench/tools/lsd_edges_vs_gt.py data/datasets/tum/rgbd_dataset_freiburg2_desk edges.csv
```

Hypotheses, most likely first:

1. **Wrong edges accepted.** The reciprocal threshold (5000) was calibrated
   on 62 pairs of fr1_desk only (consistent pairs ≤ ~5000, failed ones
   ≥ ~5700, LSD_SLAM.md). On fr2_desk the distribution is much wider. A
   bad edge with a huge information matrix drags the graph. Check with
   `lsd_edges_vs_gt.py`; if so, gate on physical disagreement (rotation,
   translation / depth, scale of S_ji·S_ij) in addition to eq. 20, or
   normalise the information by the number of points.
2. **Information scaling.** Edge information is the raw alignment Hessian
   (~700× overconfident, but uniformly so). Fallback edges reuse the last
   parent edge's information. Edges with many points dominate regardless of
   quality; consider normalising per edge (e.g. by used points) or robust
   kernels (Huber/Cauchy) on edges in `optimize_pose_graph`.
3. **Neighbour selection.** Nearby keyframes are chosen by distance < 1.5
   mean depths and optical axes < 50°; on fast motion many of these fail
   (fr1_desk: ~40 % of pairs diverge by 6–30°). The paper uses "the closest
   ten"; a frustum-overlap test (fraction of points that project inside the
   other keyframe) would be a better filter and saves alignment time.
4. **Half resolution.** Constraint alignment stops at level 1
   (`ConstraintConfig` sets `tracker.last_level = 1`); try level 0 for the
   parent edge at least.
5. **Optimiser.** `optimize_pose_graph` uses numerical Jacobians and an
   approximate residual (log R, t, log s), not the sim(3) log. Fine for
   small errors; check its cost actually decreases on fr2_desk
   (`ConstraintReport::initial_cost / final_cost`).

The local jumps at fr2_desk frames 2520–2524 and 2563 (13.6 cm steps for a
true 1.1 cm, weakly textured floor) are a tracking problem the graph cannot
fix without a loop. Ideas: reject a frame whose tracking energy or usage is
far outside its keyframe's history, and re-track it from the constant-velocity
guess, or skip it.

## Next steps after that

1. **Measure loops** on fr3_long_office (returns to its start) and
   fr1_room. The SuperPoint engine path is `models/superpoint-1024x576-k2048.engine`;
   the snapshot tool sets it automatically. `LoopConfig` defaults: 15-keyframe
   gap, shortlist 5, ≥ 20 matches, ≥ 12 RANSAC inliers (8 % of depth). These
   are untested guesses for LSD keyframes (the thresholds come from the
   SuperPoint VO's place-recall test).
2. **Threads** (designed, not started). Tracking stays on the caller's thread
   and tracks against an immutable `shared_ptr` snapshot of the current
   keyframe's tracking reference. A mapping thread owns the live
   `SemiDenseDepth`: stereo updates (drop old ones when behind) and keyframe
   creation (propagate, seed), publishing a new depth state (depth, reference,
   cloud, validity) by pointer swap. A graph thread finalises keyframes (loop
   candidates, constraints, optimisation) and writes keyframe poses under a
   mutex; world poses of frames are computed on read from (keyframe,
   relative). Keep a synchronous mode with the same job code for
   deterministic benchmarks. The GUI then reads snapshots instead of
   `Keyframe` internals (`lsd_app.cpp` reads `kf->filter`, `kf->reference`,
   `kf->depth`, clouds; `lsd_graph_view.cpp` reads poses, clouds, edges).
3. **Speed / CUDA**: tracking ~70 ms and mapping ~30 ms per frame at 640×480
   on one CPU thread; graph ~150–250 ms per keyframe. Per-pixel loops
   (`se3_tracker` linearise, `semi_dense_depth` observe/regularise) port
   directly.
4. **GUI**: the variance slider exists; missing are scrubbing, a constraint /
   loop log, and drawing which edges were rejected.

## Things that cost time

- Undistortion without masking the *frame* side made results worse; the
  pyramid carries the mask now (`Level::measured`). Keep it when adding code
  that samples frames.
- Stereo noise: σ_epipolar = 0.5 px was 2.5× overconfident on TUM; 2 px is
  calibrated (err/σ ≈ 1.2). The depth prior σ (25 %) is deliberately loose:
  tighter helps depth on fr1_xyz but hurts fr1_desk and ATE on both.
- The depth network's metric scale is off by 1.5–2.1× on TUM; only relative
  depth is used (scale lock).
- The reciprocal check's raw Hessian covariance is ~700× too small (paper:
  "only a lower bound"); thresholds in `ConstraintConfig` reflect that.
- Debug builds compile the LSD code with -O2 (set in CMake); without it
  tracking is far too slow.
