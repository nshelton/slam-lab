# LSD-SLAM pipeline: design and status

Current state and next steps: [LSD_HANDOFF.md](LSD_HANDOFF.md).

A second, independent pipeline in the native workbench: direct (feature-less)
monocular SLAM after Engel, Schöps & Cremers, *LSD-SLAM: Large-Scale Direct
Monocular SLAM* (ECCV 2014), with the semi-dense depth filter of Engel, Sturm
& Cremers, *Semi-Dense Visual Odometry for a Monocular Camera* (ICCV 2013).
It does not use SuperPoint tracks or the keypoint VO. It shares the decoder,
the geometry helpers (`vo_geometry`, `pose_graph`), the 3D views and the
evaluation tools.

The reference implementation (tum-vision/lsd_slam) is GPLv3. It may be read
for unpublished constants; no code is copied or linked.

## Decisions (2026-10-01)

- **CPU first, CUDA later.** The per-pixel core (tracking, stereo, filter)
  is Eigen/CPU under `src/lsd/`, namespace `slam_native::lsd`, testable
  headlessly. Hot loops move to CUDA once the numbers match.
- **Depth prior from the depth network.** The first keyframe (and later,
  regions that stereo has not reached) is seeded from Depth Anything with a
  wide inverse-depth variance, instead of the paper's random init. The prior
  is behind an interface so tests can use TUM ground-truth depth or random.
- **Input.** Grayscale = the NV12 luma plane, downloaded once per frame and
  undistorted to a pinhole image on the CPU. TUM `rgb.mp4` is h264 at
  ~19 Mbit/s for 640×480 (near-lossless), so the bench and the app use the
  same NVDEC path.
- **Output.** Trajectory CSV in the existing format (`track_io.hpp`), so
  `tools/tum_eval.py` scores it unchanged.
- **Loop candidates.** The paper's FAB-MAP candidate is replaced by the
  SuperPoint global place recall already measured on fr1_room.

## Algorithm summary (what we implement)

Notation: keyframe K_i = (I_i, D_i, V_i), D inverse depth, V its variance,
defined on pixels with enough gradient (semi-dense). Poses use the
left-perturbation convention of `vo_geometry` (`T <- exp(δ) T`, δ = (ω, v)),
which matches the paper and `pose_graph.cpp` (Sim3 adds log s).

**Tracking (SE(3), paper Sec. 3.3).** For each frame j and keyframe i,
minimise Σ_p ‖r_p / σ_rp‖_δ (Huber) with
`r_p = I_j(ω(p, D_i(p), ξ)) - I_i(p)` and
`σ_rp² = 2σ_I² + (∂r_p/∂D_i(p))² V_i(p)`. Levenberg-Marquardt / IRLS,
coarse to fine on a 2×2-average pyramid; the inverse of the final Hessian is
the pose covariance.

**Depth (ICCV 2013 Sec. 2).** Per tracked frame, for keyframe pixels whose
expected stereo accuracy is good: search the epipolar line (5-point SSD,
interval D ± 2σ when a prior exists), subpixel refinement, observation
variance `α² (σ_λ(ξ,π)² + σ_λ(I)²)` with geometric term `σ_l² / <g,l>²` and
photometric term `2σ_i² / g_p²`; fuse by the scalar Kalman update.
Regularise once per frame (inverse-variance-weighted neighbourhood mean,
excluding neighbours more than 2σ apart; variances unchanged). Keep a
validity score per pixel for outlier removal and hole filling.

**Keyframes (paper Sec. 3.4).** New keyframe when `ξᵀWξ` (relative to the
keyframe's mean inverse depth = 1) passes a threshold. Its depth is forward-
warped from the old keyframe (`σ₁² = (d₁/d₀)⁴ σ₀² + σ_p²`, occlusion on
collision), regularised, then rescaled to mean inverse depth 1; the factor
goes into the keyframe's Sim(3) pose.

**Map (paper Sec. 3.5–3.6).** Finished keyframes are aligned on Sim(3) to
nearby keyframes with photometric + depth residual
`r_d = [p']_z - D_j(p')`, `σ_rd² = V_j (∂r_d/∂D_j)² + V_i (∂r_d/∂D_i)²`,
Huber on the sum; reciprocal check by the Mahalanobis distance of
ξ_ji ∘ ξ_ij. Sim(3) pose graph with covariance-weighted edges in a
background thread. Loop alignment starts at a ~20×15 pyramid level.

## Running it

```bash
./run-lsd-workbench.sh                 # launcher (same recent-video grid as the SuperPoint workbench)
./run-lsd-workbench.sh --video data/datasets/tum/rgbd_dataset_freiburg1_xyz/rgb.mp4
# headless: screenshot + trajectory CSV, then score
native_workbench/build/app-debug/slam-native-lsd-snapshot VIDEO native_workbench/models FRAMES out.ppm traj.csv [DEPTH_VIEW]
.venv-cuda/bin/python native_workbench/tools/tum_eval.py SEQUENCE_DIR traj.csv
```

Binary `slam-native-lsd-workbench` (`src/lsd_main.cpp`, `src/lsd_app.cpp`,
`src/lsd_graph_view.cpp`). It shares `Launcher` (in video-only mode), the
NVDEC decoder, the GL presenter, `DepthEstimator` and `PointCloudRenderer`
with the SuperPoint workbench; its window layout is kept in `imgui-lsd.ini`.

- **LSD pipeline**: play/pause/step/restart, tracking state and statistics,
  depth model, live settings.
- **Video**: the frame with the current keyframe's semi-dense points warped
  in by the tracked pose, coloured by inverse depth.
- **Keyframe depth**: inverse depth, relative depth sigma, tracking residual
  r/σ and Huber weight of the last frame, the dense network prior, the image.
- **Pose graph**: keyframe frustums (current one yellow), parent edges,
  trajectory, the live camera, and every keyframe's semi-dense cloud.

The luma plane is downloaded per frame and halved until at most
`--max-width` (640) wide. Lens distortion is not corrected yet. Tracking runs
on the UI thread.

**Keyframe depth.** Each keyframe owns a `SemiDenseDepth` filter while it
is current (`src/lsd/semi_dense_depth`). Every tracked frame runs stereo on
it, fuses, regularises, and rebuilds the tracking reference (~30 ms at
640×480). A new keyframe gets the old one's hypotheses by propagation; the
depth network (run synchronously on keyframes, ~3.5 ms) seeds high-gradient
pixels that propagation left empty, with σ = 25 % of the inverse depth and
its scale locked to the previous keyframe (median ratio). The old filter is
then released; its smoothed map stays as the keyframe's final depth.
`LSD_STEREO=0` / the "Stereo depth filter" checkbox turns stereo and
propagation off (network depth per keyframe only), for comparison.

**Undistortion.** With `calibration.json` distortion (OpenCV k1 k2 p1 p2 k3)
the working image is remapped to pinhole (`src/lsd/undistort`). The mask of
pixels sampled inside the input (eroded by 2 px) is carried by the pyramid:
the tracker and stereo never compare samples outside it and keyframes never
place depth there. The video overlay re-distorts projected points. The depth
network still sees the distorted frame and is sampled through the map.

## Build order and status

| Step | Content | Validation | Status |
|---|---|---|---|
| 0 | `slam-native-lsd` library, image pyramid, undistortion, headless runner writing trajectory CSV | runs on TUM | done (`slam-native-lsd-snapshot`) |
| 1 | SE(3) direct tracker (`src/lsd/se3_tracker`) | synthetic scene done; TUM frames against keyframes with **ground-truth** depth next | synthetic passing |
| 2 | Depth filter: stereo, fusion, propagation, regularisation, outliers (`src/lsd/semi_dense_depth`) | synthetic test; per-keyframe audit against TUM depth (`tools/lsd_depth_vs_gt.py`) | done, in the app |
| 3 | Odometry: keyframe selection, scale normalisation, Depth Anything prior | ATE on fr2_desk (paper: 4.52 cm, 116 KF), fr1_xyz | interim: network depth per keyframe, scale lock (`src/lsd/odometry`) |
| 4 | Sim(3) keyframe alignment, reciprocal check, pose graph, SuperPoint loop candidates (`sim3_tracker`, `keyframe_graph`, `place_recognition`) | ATE on fr3_long_office, fr1_room loop | in progress, synchronous: helps short clips, hurts fr2_desk; loops unmeasured (LSD_HANDOFF.md) |
| 5 | New GUI binary: live frame + inverse-depth overlay, residual and Huber-weight images, keyframe graph, cloud with max-variance slider | headless snapshot | first version (no variance slider yet) |
| 6 | CUDA port of per-pixel loops | timing at source resolution | |

## Measurements

**Step 1, synthetic** (`tests/test_lsd_tracker.cpp`, 320×240, f = 300, wall at
4 m + box at 2.5 m, 5 levels down to 20×15, ~30k semi-dense points; 2026-10-01):

| Case | Translation error | Rotation error | Time |
|---|---|---|---|
| small motion (3.7 cm, 1°), from identity | 1.27 mm | 0.021° | 10 ms |
| large motion (22 cm, 5.4°), from identity | 1.17 mm | 0.016° | 5 ms |
| 40% of points with σ_d = 0.08 noise, variance-weighted | 6.8 mm | 0.10° | |
| same, constant variance (no eq. 14 weighting) | 31 mm | 0.43° | |

The ~1.2 mm floor is occlusion at the box edges: on the wall alone the error
is 0.04–0.3 mm, which also confirms the Jacobian. More iterations or a
tighter convergence test do not change it.

**Interim odometry on TUM** (network depth per keyframe, no stereo filter, no
undistortion, 640×480, 2026-10-01; `slam-native-lsd-snapshot` + `tum_eval.py`):

| Sequence (frames) | Keyframes | Lost | ATE (sim3) | RPE 1 s | Track time |
|---|---|---|---|---|---|
| fr1_xyz (0–299) | 13 | 0 | 2.59 cm | 2.62 cm, 1.57° | 36 ms/frame |
| fr1_desk (0–199) | 14 | 0 | 3.25 cm | 5.06 cm, 2.62° | 42 ms/frame |

The residual view shows paired red/blue fringes along edges: the network's
depth is slightly off at edges, which the stereo filter is meant to fix.

**Step 2, synthetic** (`tests/test_lsd_depth_filter.cpp`, exact poses, 12
frames moving 1.2 cm sideways per frame):

| Case | Pixels | Median error |
|---|---|---|
| prior: 20 % noise + 15 % bias, after regularisation | 36k | 9.2 % |
| prior refined by stereo | 46k | 1.1 % |
| stereo only (full-range search, no prior) | 39k | 0.27 % |
| refined map propagated to a new keyframe | 43k | 0.004 % |

With exact poses the variance is conservative (error/σ ≪ 1), as it should
be: the model includes pose and calibration error.

**Step 2 on TUM** (2026-10-01, first 300 / 200 frames, 640×480; ATE after
sim3; depth error = median |log z − log z_true| after one scale per keyframe,
`tools/lsd_depth_vs_gt.py`):

| fr1_xyz / fr1_desk | ATE | depth error | err/σ |
|---|---|---|---|
| network depth per keyframe, distorted | 2.29 / 3.23 cm | | |
| network depth per keyframe, undistorted (mask in tracking) | 2.18 / 3.13 cm | prior 2.8 / 4.8 % | |
| + stereo filter, σ_l = 0.5 px | 1.56 / 2.67 cm | 4.05 / 3.05 % | 2.3 / 2.9 |
| + stereo filter, σ_l = 2 px (default) | **1.47 / 2.38 cm** | 3.82 / 2.75 % | 1.2 / 1.3 |

- Undistorting without masking the frame made fr1_desk worse (4.11 cm): the
  smeared corners were compared. With the mask it helps.
- σ_l = 0.5 px made stereo overconfident (err/σ 2.3–2.9); 2 px calibrates it.
- Stereo improves depth on fr1_desk (2.75 % vs the prior's 4.8 %) but not on
  fr1_xyz (3.8 % vs 2.8 %), also away from depth edges (3.7 vs 2.8 %): there
  the network is unusually good. A tighter prior σ trades one for the other
  (σ = 2 %: fr1_xyz 2.6 %, fr1_desk 3.8 %) and raises ATE on both, so the
  default stays 25 %.
- The network's metric scale is off by 1.5× (fr1_xyz) to 2.1× (fr1_desk) on
  TUM; only relative depth is used.

**Full sequences, odometry only** (no pose graph or loop closure yet;
2026-10-01, default settings, 640×480; `slam-native-lsd-snapshot` FRAMES =
all):

| Sequence | Frames | KF | Lost | ATE | RPE 1 s | Time/frame (track + map) |
|---|---|---|---|---|---|---|
| fr2_desk | 2965 | 50 | 0 | 24.5 cm | 5.6 cm, 2.3° | 72 + 32 ms |
| fr2_desk, stereo off | 2965 | 58 | 0 | 90.4 cm | 32.8 cm, 20.0° | 48 ms |
| fr3_long_office_household | 2585 | 73 | 0 | 41.4 cm | 3.4 cm, 1.3° | 68 + 29 ms |

Paper (with loop closure): fr2_desk 4.52 cm, 116 KF.

- fr2_desk: within 300-frame windows the trajectory is accurate (0.4–1.3 cm
  after each window's sim3) and scale drifts ~10 % over 2200 frames; most of
  the ATE comes from jumps at frames 2520–2524 and 2563 (13.6 cm steps for a
  true 1.1 cm), where the view is mostly weakly textured floor with a plant.
- fr3_long_office: scale wanders between 0.85× and 1.10× of the first
  window's along the loop (local errors 3–6 cm in frames 602–1501). The
  sequence returns to its start: a Sim(3) loop edge (step 4) is what corrects this.
- Without stereo, network depth alone per keyframe drifts badly over a long
  sequence; the filter is essential, not a refinement.

**Step 4, synthetic** (`tests/test_lsd_sim3.cpp`): two keyframes, the second
in 1.3× depth units, started from scale 1 and ~1° rotation error: S_ji
recovered to 1.9 mm, 0.016°, 0.035 % scale; reverse direction 0.45 mm. Eq. 20:
13.5 for the consistent pair, 3.9·10⁶ when the reverse misses the scale.

**Step 4 on TUM** (2026-10-01, synchronous, no loop candidates yet):

| Run | Graph off | Graph on |
|---|---|---|
| fr1_xyz 0–299 | 1.46 cm | 1.33 cm (49 edges, 0 fallback) |
| fr1_desk 0–199 | 2.35 cm | 2.20 cm (29 edges, 0 fallback) |
| fr2_desk full | 24.5 cm | 29.2 cm (89 edges, 3 fallback, 12.7 s graph time) |

- Reciprocal check calibration (fr1_desk, 62 pairs): pairs whose
  forward/backward disagreement is physically small (< 1.5°, < 2 % of depth,
  < 2 % scale) score up to ~5000; failed alignments (6–30°) from ~5700. With
  the threshold at 100 every parent edge was rejected. At 5000 and alignment
  starting at 20×15: 29 of 32 consistent accepted, 0 of 30 failed.
- Starting the alignment at 20×15 vs 40×30…160×120 changed little; ~40 % of
  tried pairs on fr1_desk are genuine failures (keyframes 3+ apart with large
  viewpoint change).
