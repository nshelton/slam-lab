# Native workbench architecture

## Start here: the tracking and pose system as of 2026-09-30

This section is the handoff for work on robust camera tracking, track
re-association after occlusion, and a camera motion model. The dated
sections below keep the history and measurements; where they disagree, this
section is current. Design proposals: `CONFIDENCE_DESIGN.md` (uncertainty),
`DEPTH_INTEGRATION.md` (metric depth network → scale, dense map, occlusion).

### Pipeline in one picture

```text
NVDEC frame (native orientation, source px, lens-distorted)
  ├─► SuperPoint (TensorRT, 1024×576 letterbox, top-2,048) ─► detections + 256-d descriptors
  ├─► NVOF forward flow (4×4 grid, previous → current)
  └─► ColorSampler (3×3 RGB at each track; CUDA)
            │
            ▼
  FlowTracker (GPU, src/flow_association.cu)
    predict (flow) → gate (radius + descriptor cosine) → k rounds propose/accept
    → layout (matched | coasted | new, cap 1,000) → build: Kalman position fusion
            │  one D2H copy: TrackRecord per live track (id, x, y, prediction, correction, kind…)
            ▼
  tracked_frame_from(features, colors)   drops coasted tracks; attaches RGB
            │  TrackedFrame: (track_id, x, y, rgb) per supported track
            ├─► FocalEstimator (worker thread): FOV + k1 from F-matrix consistency
            ▼
  VisualOdometry (CPU, UI thread; src/visual_odometry.cpp)
    undistort (k1) → init (E/H RANSAC) → per-frame pose (robust LM, constant velocity)
    → keyframes: triangulate, sliding-window BA (8 KF), cull → retire landmarks
            ▼
  OdometryFrameResult (pose, inlier/outlier track IDs), trajectory, active + retired map
            ▼
  TrajectoryView (ImGui 3D + instanced GL map points), video overlay (pose-inlier view), CSV exports
```

### Contracts a new component must respect

- **Coordinates.** Tracks, detections and flow are in source-video pixels of
  the *native* (unrotated) frame, and they are lens-distorted. The VO
  undistorts on input (`undistorted()` / `undistort_point()`, division model,
  `VisualOdometryConfig::distortion_k1`). Landmark observations inside the VO
  are therefore undistorted pixels. The GUI's display rotation
  (`--rotate`, metadata display matrix) is display-only; nothing downstream
  of the decoder sees it.
- **Track IDs are unique for a run and never reused** (`VisualOdometry`
  keys landmarks by track ID). The GPU tracker allocates them in `build()`
  from `next_id_`; seeking or restarting creates a new tracker, so IDs
  restart and the VO/trajectory are reset (`Session::seek`). Re-association
  must not revive an old ID inside the tracker unless the VO contract is
  changed too. The cleaner route is aliasing at the VO level (below).
- **Only supported observations reach the VO.** Coasted positions (flow-only)
  are dropped in `tracked_frame_from()`.
- **Track positions are fused, not exact detections** (since 2026-09-30,
  default `flow_sigma_px = 0.3`). Never identify "the detection a track
  used" by position equality. The bench did that and reported continuation
  0.005 instead of 0.88. Use the tracker's `kind`/`superpoint_supported`,
  or plumb the detection index (below).
- **GPU frame lifetime.** A `GpuFrame` is valid until the next
  `VideoDecoder::next()`. Consumers on their own streams must
  `cudaStreamWaitEvent(frame.ready_event)` first.
- **CUDA context.** Create CUDA objects lazily, after the decoder has opened
  (it configures the primary context). An eagerly constructed stream caused
  FFmpeg's "incompatible flags" warning (`ColorSampler` is lazy for this
  reason).

### How a track lives and dies (GPU tracker)

Per frame, in `FlowTracker::associate` → kernels in `src/flow_association.cu`:

1. `predict`: each live track moves by bilinear-sampled forward flow.
   Optional forward-backward check marks unreliable flow.
2. `gather_candidates`: up to 8 detections within `association_radius`
   (6 px) of the prediction with descriptor cosine ≥ 0.7 to the track's
   last *observed* descriptor. Cost = d²/r² + (1 − cosine).
3. `propose`/`accept` × `assignment_rounds` (4): one-to-one, deterministic.
4. `layout`: slots are [matched | coasting | new]. A track with no match
   coasts if its flow is reliable and `misses < max_coast_frames` (5);
   otherwise **it dies here, silently**. Coasting tracks beyond the
   1,000 cap are also dropped. New tracks come from the strongest unmatched
   detections.
5. `build`: matched tracks run the Kalman update (predict P += σ_flow²;
   K = P/(P+σ_det²); x = pred + K(det − pred)); coasting tracks take the
   prediction and grow P; new tracks start at the detection with P = σ_det².
   `copy_descriptors`: matched/new tracks take the current detection's
   descriptor (half precision); coasting tracks keep the old one.

Death is currently final. A point that reappears after occlusion gets a
new ID, and in the VO a new landmark, so walls get duplicated in the map.

### How a landmark lives and dies (VO)

In `VisualOdometry::Impl` (`src/visual_odometry.cpp`):

- Created at initialization (both init frames) or in `triangulate_new()` at
  a keyframe, from the oldest window keyframe that saw the track
  (≥ 0.5° parallax, ≤ 3 px error in both views).
- Per frame, `track()` gathers the current observations of tracks that have
  landmarks, solves the pose, and splits them into inliers/outliers
  (`OdometryFrameResult::pose_inliers/pose_outliers`; the GUI's "Pose
  inliers" point view draws them green/red).
- At a keyframe, outlier landmarks are **deleted** (`landmarks.erase`,
  they're usually short-baseline depths and get re-triangulated).
- `cull()`: observations that fail 2× the reprojection threshold are
  dropped; a landmark with < 2 observations is **deleted**; a landmark whose
  track is gone and which no window keyframe observes is **retired**
  (`retire()` → `retired`, final, never refined again), or, with
  relocalization and a descriptor, first kept **dormant** for 90 frames.
- On loss, a segment with ≥ 3 keyframes is **suspended** for relocalization
  (below); otherwise, and on resolution change, `clear_segment()` retires the
  segment's landmarks. Either way the next segment starts with a new frame
  and scale unless the old one is resumed.
- `MapPoint` (public) carries: track ID, position, mean RGB of keyframe
  observations, segment, keyframe count, first/last frame.
  `active_map()`, `retired_map(first)`, `retired_count()`;
  `write_map_csv()` exports both.

### Relocalization and landmark persistence (implemented 2026-09-30)

Tracks still die at occlusions (the tracker's unique-ID contract is
unchanged); the VO re-associates at the landmark level instead.

- **Descriptors reach the VO.** `TrackRecord::detection` (the former `pad`)
  is the matched detection's index; `FlowTracker::associate` fills
  `FrameFeatures::track_descriptors` (the matched detection's 256-d
  descriptor per track, zero rows for coasted tracks) whenever the raw
  descriptors are on the host (live SuperPoint and cached frames both are).
  `tracked_frame_from()` copies them into `TrackedFrame::descriptors`.
  Tracks CSV input has no descriptors: relocalization is then inactive and
  the VO behaves exactly as before.
- **Landmarks keep their latest descriptor** (updated at every inlier
  observation) and their segment. Measured on `dreamworks.MOV`, the latest
  descriptor beats the first, a mean, a per-landmark variance/PCA model and a
  constant-velocity Kalman prediction of the descriptor; drift is a random
  walk (`tools/descriptor_*.py`).
- **Dormant landmarks.** With relocalization on, a landmark whose track ended
  and which left the window is kept for `relocalization_max_frames` (90)
  before it is retired. Dormant and suspended landmarks are listed by
  `active_map()` (not final); `retired` stays append-only and final.
- **On loss** of a segment with ≥ `relocalization_min_keyframes` (3), its
  landmarks (active + dormant), last pose and velocity are suspended. Every
  following frame (until 90 frames after its last pose, even while a new
  segment initializes) tries `relocalize()`: constant-velocity prediction →
  mutual-best descriptor matches within 256 px (cosine ≥ 0.7, grid-bucketed)
  → P3P RANSAC (`vo::p3p`, Grunert; `vo::ransac_pnp`, 4 px) → `optimize_pose`
  → guided search within 16 px → `optimize_pose`; ≥ 30 inliers accepts.
  `resume_lost()` retires the young segment, restores the old segment's
  keyframes and landmarks, re-keys matched landmarks to the new track IDs,
  adds a keyframe and runs the local BA. Segment IDs come from `next_segment`
  and are never reused; a resumed segment keeps its ID, frame and scale.
- **Before declaring loss**, `track()` retries a failed pose with P3P RANSAC
  on the same correspondences (sudden motion).
- `OdometryFrameResult::relocalized` / `relocalization_candidates`; the GUI
  sidebar shows the count (tooltip: last event); bench key `reloc 0|1`,
  `SLAM_VO_VERBOSE=1` prints each attempt.

Result on `dreamworks.MOV` (30 fps, 882 frames, walking behind columns;
hfov 67.1, k1 −0.031, defaults): relocalized at 3 of 4 column events
(frames 573, 680, 786; 13–14 frames after the loss, i.e. once the column has
passed), largest segment 557 → 826 poses, 7 → 4 segments. The camera centre
steps across each gap at the walking speed seen before and after it. The
unrecovered loss (855) is a partial column in front of repetitive railings
with few recognisable landmarks. `disney_04` is unchanged (0 losses).
Constant-velocity drift measured on this clip (`tools/pose_propagation.py`):
≈ 8 px at 0.17 s, 20 px at 0.33 s, 40 px at 0.5 s, 140 px (5.7°, mostly
rotation) at 1 s — hence the wide first gate plus RANSAC rather than tight
per-landmark gates.

**Re-association while tracking** (`reassociate()`, every tracked frame,
after the pose and before the keyframe step): every landmark without an
observation this frame — active ones whose track vanished and dormant ones
(kept `dormant_max_frames` = 900, at most 20,000) — that projects into the
image is reported in `OdometryFrameResult::untracked_landmarks` (distorted
input pixels, via `distort_point`). Each takes over a track that has no
landmark yet if that track lies within `reassociation_radius_px` (2 px) of
the projection and the descriptors are mutual best matches (cosine ≥ 0.7);
the landmark is re-keyed to the new track, so no duplicate is triangulated.
The GUI's video view draws them (checkbox "Untracked landmarks"): magenta
ring = untracked (faint: dormant), filled magenta + white ring = re-found
this frame. The 2 px gate is measured, not guessed: on `dreamworks.MOV`,
re-associations within 2 px of the projection were pose inliers in the next
frames 92% of the time, 2–3 px 72%, 3–5 px 19%, beyond 5 px 3% (bench prints
this breakdown). Relocalization only matches landmarks seen within 90 frames
of the loss (older dormant ones have drifted and steal mutual-best matches in
its 256 px window).

Effect (bench, `reassoc 0|1`): synthetic walk with broken tracks 5,603 →
1,420 landmarks, 9,346 re-associations, 0 joining different points.
`disney_04`: 8,550 → 7,677 landmarks, 1,543 re-associations (90% confirmed),
still 0 losses, inlier ratio 0.898 → 0.894. `dreamworks.MOV` is fragile
around frames 250–300: with fusion off re-association raised the largest
segment 538 → 814 poses, with defaults it lowered it 826 → 523 (a different
early trajectory led to a collapse at 299) — sensitivity, not wrong merges
(0–2 re-associations per frame there). VO cost ≈ +1 ms/frame.

Not done yet: what to do with landmarks that stay untracked in view (the
magenta rings: occluded, detector misses, or stale/drifted positions);
relocalization against segments older than the last lost one; loop closure.

### Where to add a camera motion model

- **Current model.** `track()` predicts the pose with constant velocity
  (`velocity * last_pose`, velocity = last × previous⁻¹), falls back to
  constant position if that yields too few inliers, then runs
  `optimize_pose()` (Huber LM, 10 iterations × 2 stages). There is no
  covariance, no IMU, and no feedback to the tracker.
- **Pose covariance.** `optimize_pose()` builds the 6×6 Gauss-Newton
  Hessian `H = Σ w JᵀJ` internally (`src/vo_geometry.cpp`); its inverse
  at convergence is the pose covariance (up to the noise scale). Return it
  to feed a filter.
- **A filter on SE(3)** (constant-velocity EKF / error-state on the left
  perturbation used everywhere: `perturb(T, δ) = exp(δ) · T`, δ = (ω, v))
  would replace the ad-hoc velocity and give an uncertainty for gating.
- **Feeding the pose back to the tracker.** For tracks with landmarks, the
  predicted pose gives a predicted pixel (`K.project(T_pred * X)`, then
  *distort* with k1, since the tracker works in distorted pixels). This
  helps where flow fails (occlusion, blur, fast motion) and is the natural
  way to search for re-appearing retired landmarks. The tracker's `predict`
  kernel takes only flow today; add an optional per-track prediction array
  (uploaded from the host, indexed like the tracks) and blend or use it as
  a second gate. Frame-to-frame, flow is usually better than a pose
  prediction (it includes per-pixel depth effects), so treat the pose as a
  fallback or a prior, not a replacement.
- **Outlier feedback.** The tracker never learns that the VO rejected a
  track (`pose_outliers`). Down-weighting or killing persistently rejected
  tracks is a cheap robustness win.
- Threading: the VO and keyframe BA (≈10–12 ms) run synchronously on the UI
  thread; the lens estimator shows the worker pattern
  (`BackgroundFocalEstimator`).

### Lens model

`VisualOdometryConfig`: `horizontal_fov_degrees` (square pixels, centred
principal point) plus `distortion_k1` (division model, r normalised by the
half-diagonal, negative = barrel), or `intrinsics` (FX FY CX CY). The FOV is
the central focal length's. `FocalEstimator` (`src/focal_estimation.cpp`)
estimates both from the tracks: k1 by minimising the epipolar residual of
refitted fundamental matrices over undistorted tracks (in image pixels;
measuring in undistorted units biased k1 by +0.02), then the FOV at which
KᵀFK is a valid essential matrix (Mendonça–Cipolla). The GUI's **Camera
lens** window shows both curves and applies them. Pure translation informs
k1 but not the FOV; pure rotation informs neither.

### Live metric depth (added 2026-09-30)

`DepthEstimator` (`include/slam_native/depth_estimator.hpp`,
`src/depth_estimator.cu`) runs a monocular metric-depth TensorRT engine on its
own CUDA stream. It is display-only so far: nothing in tracking or the VO reads
it yet.

- **Asynchronous.** `Session::submit_depth()` queues a frame whenever the
  previous one has finished. `submit()` returns after preprocessing, so the
  decoder may recycle the surface. `poll()` (every UI iteration) picks up the
  result, so the map lags the video by about a frame.
- **Upright input.** Depth networks are trained on upright images, so the
  preprocessing kernel applies the display rotation (NV12/P010 → BT.709 RGB,
  up to 4×4 taps per input pixel, letterbox, per-channel normalisation).
  Each model has a landscape and a portrait engine, `<stem>-<W>x<H>.engine`.
  Changing the model or the rotation rebuilds the estimator.
- **Coordinates.** A `DepthMap` holds metres on the network's output grid
  (upright, letterboxed, 0 = padding). `DepthMap::at_source(x, y)` samples it
  at *native source pixels*, the same coordinates as tracks, so a consumer
  (VO scale, depth-initialised landmarks) needs no extra mapping. Points are
  lens-distorted, like the frame the network saw.
- **Models are data.** A `DepthModelSpec` holds the tensor names, mean/std on
  0–255 RGB, letterbox fill, clamp and an optional canonical focal length
  (Metric3D predicts depth for f = 1000 px. It is scaled by the VO lens's
  focal length in network pixels, so the depth follows the applied FOV).
  Inputs and outputs may be float32 or float16, outputs `[1x[1x]]HxW` at any
  resolution, and engines with dynamic shapes run at their profile's optimum.
  Extra outputs (Metric3D's normals) are bound and ignored. To add a model,
  export an ONNX, add it to `tools/build_depth_engines.sh`, and add a preset
  to `depth_model_presets()`.

| Preset (`--depth-model`) | Engine input | trtexec GPU, 2080 Ti | In-app (incl. pre/post) |
|---|---|---|---|
| Depth Anything V2-S (indoor / outdoor) | 518×294 / 294×518, fp16 | 2.3 ms | ~4 ms |
| Metric3D v2 ViT-S | 1064×616 / 616×1064, fp16 | 23 ms | — |

DAv2 checkpoints: Hypersim (indoor, clamp 20 m) and VKITTI (outdoor, 80 m).
The ONNX is exported in fp16 (`tools/export_depth_anything_onnx.py`) because
TensorRT 11 builds strongly typed networks, so the ONNX types set the
precision. Metric3D is the onnx-community fp16 export.

### Measuring changes

No ground truth exists for the real clips. Use these, in order of fairness:

1. **Epipolar RMS** of tracks over frame pairs (`slam-native-focal
   tracks.csv`, the minimum of the k1 curve): track noise against the
   best two-view geometry, independent of the VO. Smoothed tracks cannot
   game it without being geometrically consistent.
2. **VO inlier ratio / median reprojection** (bench "camera pose" line).
   Slightly flattering for temporally smoothed tracks.
3. **Track continuation / median length** (bench). Continuation is now the
   share of previously live tracks supported by a detection.
4. **Map sanity**: `export-map` + a top-down plot; duplicated walls mean
   missed re-association.

Baseline, `disney_04` (1920×1080, 60 fps, 1,798 frames; lens 87.9° /
k1 −0.054; defaults, flow sigma 0.3): continuation 0.883, median track
length 15, VO inlier ratio 0.898, median reprojection 1.12 px, epipolar RMS
1.082 px, 0 losses, ≈8,700 map points (≈8,000 retired). With fusion off:
0.881 / 17 / 0.840 / 1.32 px / 1.215 px. Reproduce:

```bash
B=native_workbench/build/app-debug
$B/slam-native-tracking-bench ~/Downloads/chopped/disney_04.mp4 \
  native_workbench/models/superpoint-1024x576-k2048.engine 0 1800 \
  hfov 87.9 k1 -0.054 export-tracks tracks.csv export-map map.csv export-trajectory traj.csv
$B/slam-native-focal tracks.csv          # epipolar RMS curve, FOV, k1
$B/slam-native-vo-tracks --tracks tracks.csv --hfov 87.9 --k1 -0.054 --map map.csv --output traj.csv
$B/slam-native-two-view --tracks tracks.csv --a 1200 --b 1230 --hfov 87.9 --k1 -0.054 --points pair.csv
.venv-cuda/bin/python native_workbench/tools/plot_two_view.py pair.csv VIDEO pair.png
$B/slam-native-snapshot VIDEO ENGINE START_S FRAMES out.ppm   # headless render of the real UI
```

ORB-SLAM3 is the independent reference (`scripts/run_orbslam3.sh`); on
`disney_04` at 60° it traced the same path shape as this VO did.

### Tools and tests added 2026-09-30

- `slam-native-two-view`: the VO's own two-view code on one frame pair,
  every intermediate printed; `--sweep-hfov`, `--k1`, `--points`.
- `slam-native-focal`: FOV + k1 estimate from a tracks CSV.
- Bench keys: `flow-sigma`, `det-sigma`, `hfov`, `k1`, `export-map`.
- Tracks CSV: optional `r,g,b` columns. Map CSV: see `track_io.hpp`.
- CTest: `native_focal_estimation` (synthetic FOV/k1 recovery, background
  worker parity), retired-map + colour check in `native_visual_odometry`,
  Kalman fusion check in `native_gpu_flow_tracker`.
- Relocalization: bench keys `reloc`, `export-features PATH` (raw detections
  + descriptors in the feature-cache format); CTest checks for P3P (exact
  recovery, RANSAC with 60% outliers), relocalization after a synthetic full
  occlusion with new track IDs, and per-track descriptors in the tracker.
- Offline analysis (`tools/`, run with `.venv-cuda/bin/python` on a bench run
  with `flow-sigma 0 export-tracks … export-features …`):
  `descriptor_stability.py` (drift, recall vs search radius, database scale),
  `descriptor_models.py` (per-landmark descriptor models),
  `descriptor_dynamics.py` (MSD / Kalman on descriptors),
  `pose_propagation.py` (motion-model drift, projection + descriptor
  association, PnP on real events; needs the VO trajectory and map).

## Tracking and keypoint decisions — 2026-09-24

This is the current native exploratory pipeline; it does not change the Python
offline matching and reconstruction workflow. It supersedes the 2026-09-23
baseline (single-pass spatial association, unmatched tracks die immediately).

- SuperPoint supplies every observation. NVIDIA optical flow (4×4 grid)
  supplies motion only; grid cells never seed tracks.
- Flow predicts, SuperPoint corrects. (Until 2026-09-30 a track's observed
  position was always an exact SuperPoint detection; it is now a Kalman
  fusion of the flow prediction and the detection, see "Start here".)
  When no detection supports a track, it may
  **coast**: carried to its flow prediction for up to `max_coast_frames`
  (default 5), flagged unsupported and drawn hollow. It keeps its ID and is
  reacquired when a gated detection returns. `--max-coast 0` restores the old
  die-immediately rule.
- Association is gated by both a radius around the flow prediction (default
  6 source px) and descriptor cosine similarity to the track's last observed
  SuperPoint descriptor (default ≥ 0.7). Cost = d²/r² + (1 − cosine).
- One-to-one assignment runs several propose/accept rounds (default 4). Each
  unassigned track proposes to its cheapest still-free candidate (up to 8 kept
  per track). Each detection keeps the cheapest proposal via a packed 64-bit
  `atomicMin` (cost bits, track index), so ties go to the lower index and the
  result does not depend on thread scheduling. One round reproduces the old
  single-pass rule, where losers never tried a second choice.
- Live-slot priority under the cap (default 1,000): matched continuations in
  detection order, then coasting tracks, then the strongest unmatched
  detections by (score, index). Raw detections (up to 2,048) are unaffected
  by the cap.
- Descriptors are quantized to half precision on the GPU with an exact copy of
  the cache's `float_to_half`. Live TensorRT output (float32) and cached
  float16 replay therefore give bit-identical costs and identical track IDs.
  `slam-native-hybrid-test` checks this.
- Forward-backward flow consistency is optional (`--flow-backward PX`). It
  computes a second NVOF pass (current → previous). Tracks that fail the check
  cannot coast; they can still match a detection. It is off by default: it
  doubles NVOF time (≈3 → 6 ms at 1080p), and on Osaka it cut coasting and
  shortened tracks without improving match similarity. The descriptor gate
  already protects reacquisition.

### Diagnosis of the 2026-09-23 tracker

Measured with `slam-native-tracking-bench` (600 frames of Osaka per segment).
The flow conventions were already correct. NVOF with `inputFrame = previous`
and `referenceFrame = current` yields forward flow. S10.5 values are divided
by 32. The 4×4 grid maps correctly to source pixels, where SuperPoint
coordinates also live. A synthetic-shift test now pins these down. The failures
were elsewhere:

1. **Stream race (bug).** FFmpeg copies each NVDEC surface on its CUDA stream
   (the legacy default stream). Optical flow and SuperPoint read the surface on
   `cudaStreamNonBlocking` streams, which do not synchronize with it. About 30%
   of flow cells differed between two identical decode passes, which degraded
   predictions at random. The decoder now records a CUDA event per frame
   (`GpuFrame::ready_event`), and consumers `cudaStreamWaitEvent` on it.
   Repeated passes are now bit-identical.
2. **Attrition from detector misses.** More than 95% of lost tracks had no
   SuperPoint detection within the radius of an accurate flow prediction
   (median nearest detection ≈ 11 px, i.e. an NMS neighbour). The engine always
   saturates its fixed top-2,048. Weaker tracked corners drop out of the top-k
   for a frame, and the old rule killed them immediately. Coasting with
   descriptor-gated reacquisition addresses this.
3. **No second choices.** A detection proposed only to its nearest prediction,
   and a losing track was never offered its next candidate. This was a small
   effect compared with 2. The multi-round assignment fixes it.
4. Nearest-cell flow lookup is replaced by bilinear interpolation between
   4×4 cell centres (`(4c + 1.5, 4r + 1.5)`), identical on host and device.

Result (median track length in observations, and survivors at frame 300 of
the first frame's 1,000 tracks; 2,048-point engine, radius 6):

| Osaka segment | 2026-09-23 | current defaults | defaults + radius 8 |
| --- | --- | --- | --- |
| 30 s (slow) | 18 / 312 | 34 / 404 | 45 / 443 |
| 120 s | 10 / 132 | 27 / 176 | 37 / 185 |
| 240 s (fast) | 6 / 105 | 12 / 167 | 16 / 185 |

Median descriptor cosine of accepted matches is unchanged (0.98–0.99, p10
≥ 0.95). There is no ground truth, so correctness is supported only by that
proxy and by the synthetic tests: zero ID switches under 12% dropout with
distractors, and exact agreement with a CPU reference assignment.

## Frame processing (GPU-resident hot path)

```text
NVDEC surface ──event──► TensorRT SuperPoint ─► GPU compaction ─► DeviceDetections ┐
      │                                        (threshold, letterbox → source px) │
      └──event──► NVOF (prev→curr) ─► S10.5→float kernel ─► DeviceFlowField ──event──┤
                                                                                   ▼
              FlowTracker (one CUDA stream): predict → gather candidates → k× propose/accept
              → rank unmatched → layout (block scan) → build next state + descriptors
                                                                                   │
                          one D2H copy of compact per-track records + counts ◄─────┘
```

- Images, flow fields, detections, descriptors and track state (positions,
  half descriptors, IDs, lengths, coast counters) never leave the GPU in the
  tracking path. Track state is double-buffered on the device.
- Per frame, the host receives one fixed-size copy of compact track records
  (72 B × cap) and a counts struct, for display, the length histogram and
  inspection. The tracker synchronizes once per frame.
- SuperPoint still copies its compacted keypoints and descriptors to the host
  for the feature cache and the red raw-detection view. That copy is
  persistence and display, not tracking input. Cached frames upload their
  detections instead.
- The host flow copy (`OpticalFlow::download`) happens only while "Show flow
  vectors" is on. Toggling it while paused takes effect on the next frame.
- Kernel cost on the RTX 2080 Ti (Debug build, 1,000 tracks × 2,048
  detections): ≈0.38 ms of tracker GPU time per frame (`tracking_gpu_ms`),
  NVOF ≈3.0 ms (forward only), SuperPoint ≈16.5 ms including host readback.
  The tracker's wall time (`tracking_ms`) includes waiting for flow. The
  sidebar's optical-flow time is GPU time measured with CUDA events.

Only raw SuperPoint data is persisted. Raw inference results go to the bounded
asynchronous SQLite writer before the tracker mutates them. IDs, associations,
flow, coasting state and diagnostics are in-memory only; restarting or seeking
starts a new tracking history.

### Tests and tools

- `slam-native-gpu-tracker-test` (CTest): rules, coasting, descriptor gate,
  multi-round assignment, cap priority, CPU-reference agreement on a dense
  1,000 × 3,000 case, and synthetic translation with dropouts and distractors.
- `slam-native-optical-flow-test [VIDEO]` (CTest runs the synthetic part):
  known-shift direction and units, backward direction, and bit-exact
  repeatability over a streamed real sequence.
- `slam-native-hybrid-test VIDEO ENGINE`: live versus cached replay give
  identical flow and identical track IDs.
- `slam-native-tracking-bench VIDEO ENGINE [START_S] [FRAMES] [key value]...`:
  track-length distribution, cohort survival, loss classification,
  flow-direction sanity, and timing. Keys: `radius`, `min-sim`, `weight`,
  `coast`, `fb`, `rounds`, `tracks`, `quality`, `backward`, `k`,
  `flow-sigma`, `det-sigma`, `hfov`, `k1`, `export-tracks`,
  `export-trajectory`, `export-map`.

## Camera pose: live monocular visual odometry — 2026-09-24

```text
FlowTracker output ─► tracked_frame_from() ─► TrackedFrame ─► VisualOdometry ─► pose, trajectory, map
CSV / other trackers ─► read_tracks_csv() ──┘   (track_id, x, y) per frame        └─► TrajectoryView (ImGui)
```

**Tracker-agnostic input (the hook for other tracks).** `VisualOdometry`
(`include/slam_native/visual_odometry.hpp`) consumes `TrackedFrame`s: frame
index, timestamp, image size, and `(track_id, x, y)` observations in that
frame's pixels. Two rules apply. IDs are unique and never reused after a
track ends. Only real observations are included: the flow tracker's coasted
positions are dropped by `tracked_frame_from()`. Any tracker can drive it:
the GPU flow tracker, the old descriptor tracker (via the same
`FrameFeatures` conversion), or tracks from another program through the
CSV format in `track_io.hpp`:
`frame_index,timestamp_ns,track_id,x,y`, optionally with a first line
`# size W H`. `slam-native-vo-tracks` runs the solver on such a file and
writes a trajectory CSV (camera centre plus camera→world quaternion).
`slam-native-tracking-bench ... export-tracks PATH` writes the native
tracks in that format.

**Algorithm** (CPU, Eigen 3.4 pinned via FetchContent; `src/visual_odometry.cpp`,
`src/vo_geometry.cpp`):

1. *Initialization.* Keep a reference frame. Once enough tracks are shared
   with it and their median displacement is at least 1% of the image width,
   run essential-matrix RANSAC (8-point, Sampson error) and homography RANSAC
   on normalized coordinates.
   - Wait if the homography explains ≥ 85% of the essential inliers (pure
     rotation or a plane), or if the best of the four (R, t) decompositions
     is ambiguous.
   - Keep only points with ≥ 0.5° individual parallax. Near the focus of
     expansion depth is unobservable and those points pass any reprojection
     test. Require ≥ 60 such points with median parallax ≥ 1°.
   - Scale is set so the first keyframe's median depth is 1.
2. *Tracking.* Each frame gets a robust pose-only Levenberg–Marquardt solve
   (Huber, 3 px) on tracks that have landmarks. It starts from a
   constant-velocity prediction, falling back to constant position. Too few
   inliers means lost: a new segment starts and reinitializes. Rotations are
   re-projected onto SO(3) after every composition. Without that, the
   prediction `A·B⁻¹·A` (with Rᵀ as inverse) amplified round-off shear
   about 2.4× per frame, which `exp(ω)·R` can never remove. This was a real
   bug found by the noise-free regression test.
3. *Keyframes* (6–20 frames apart; earlier if inliers fall below 75% of the
   last keyframe's landmarks):
   - Add inlier observations.
   - Discard landmarks of live tracks that became outliers. They are almost
     always short-baseline depths, and they are re-triangulated.
   - Triangulate tracks from the oldest window keyframe that saw them, with
     ≥ 0.5° parallax and ≤ 3 px error in both views.
   - Run a sliding-window bundle adjustment over the last 8 keyframes: Schur
     complement, Levenberg–Marquardt, Huber loss; the two oldest keyframes
     are fixed to hold the gauge and scale.
   - Cull points with bad observations. Points that can never be observed
     again are retired into the persistent map (since 2026-09-30; they
     used to be deleted, so the map only showed the last second or two).
4. Per-frame poses are stored relative to their keyframe, so the displayed
   trajectory follows bundle-adjustment corrections.
   `stable_prefix()` lets the viewer refresh only the part that can still
   change.

Intrinsics default to a 60° horizontal field of view with a centred
principal point, in source-video pixels (the same assumption as the Python
solver). `--hfov` or `--intrinsics FX FY CX CY` override it. Since
2026-09-30 there is a one-parameter radial distortion model (`--k1`) and a
live FOV/k1 estimator; the 60° default was far off for `disney_04`
(≈88°), which made the path collapse into a straight forward line.

**Results.**
- *Synthetic* (`slam-native-vo-test`, CTest): a 240-frame 60 fps walk with
  0.7 px noise, 8% dropouts that break track IDs, 80 moving points and 1%
  gross outliers. One segment, ATE 0.15% of path length, median rotation
  error 0.34°, ≈1.6 ms/frame.
  - Noise-free data is reproduced to 6e-7% ATE.
  - Pure rotation never initializes.
  - A CSV round trip reproduces the trajectory exactly.
- *Osaka, live GPU tracks* (600 frames per segment):

| Segment | Posed frames | Losses | Mean inlier ratio |
| --- | --- | --- | --- |
| 30 s | 446 / 600 (camera nearly static at first) | 0 | 0.90 |
| 120 s | 570 / 600 | 0 | 0.78 |
| 240 s | 566 / 600 | 0 | 0.73 |

  Median reprojection error ≈ 1.2 px.
- *Against the offline solves* (0–30 s, one pose per second, 1485 / 1800
  frames posed, 0 losses):
  - One-second relative rotations agree with `osaka-lightglue-solver-v5`
    to 0.36° median. The two offline solves agree with each other to 0.40°.
  - Translation directions agree to 9.5° median; offline vs offline is
    19.6°.
  - Worst stretch is 18–24 s (up to 7.8° / 95°), where the offline solves
    also show a 4–6× jump in step length.
  - Relative to the offline solves, monocular scale drifts about 1.5–2×
    over 20 s.
  - Neither side is ground truth.
- *Cost.* ≈0.2–0.8 ms per frame; keyframes up to ≈10–12 ms (bundle
  adjustment), synchronously on the UI thread.

**UI.** The **Camera trajectory** window (`src/trajectory_view.cpp`, toggled
by *Trajectory* in the Pipeline panel) shows:
- state, inliers, map size, keyframes and timing;
- the current segment's path (keyframes as squares), the current camera
  frustum and the active map points;
- active and retired map points (optionally in image colour), a grid on the
  X–Z plane at the first keyframe's height (monocular VO knows no floor);
  points are solid, unlit (constant-colour) icosahedrons drawn by `PointCloudRenderer`
  (`src/point_cloud_renderer.cpp`): one instanced GL draw per set into a 4×
  MSAA, depth-tested offscreen target, composited between the grid and the
  path/frustum overlays. The shader applies the display rotation, segment
  filter and world-space size (slider; segment units, ≥ ~1 px), so
  retired points are uploaded once (append-only) and active points each frame;
- orbit (perspective, fixed 50° lens), top (X–Z), side and front
  (orthographic) views. The orbit pivot and radius come from the 3D
  trajectory, so orbiting is a true sphere (the old fit used the projected
  2D bounding box and zoomed/slid with the angle). Left-drag orbits, wheel
  dollies toward the pivot (orthographic: zooms), right-drag pans the pivot
  in the view plane, double-click fits; a faint trackball shows while
  controlling. *Follow* pivots on the current camera. *All segments*
  overlays earlier segments; each has its own arbitrary frame and scale.

Seeking or restarting resets the trajectory, because track IDs restart.
`slam-native-snapshot VIDEO ENGINE START FRAMES OUT.ppm` renders the real UI
headlessly for inspection.

**Weaknesses and next steps.**
- No relocalization or loop closure. After a loss, a new segment starts
  with a new frame and scale.
- Frames before initialization have no pose.
- Scale drift (above) grows over long runs.
- Intrinsics come from the self-calibration estimate (or a guess). Refining
  focal length and k1 inside bundle adjustment is the next accuracy lever.
- Rolling shutter and moving people are only handled by the robust losses.
- Keyframe mapping runs on the UI thread. It could move to a worker thread,
  and PnP/RANSAC could use the GPU-resident tracks.
- Descriptors (already cached per SuperPoint observation) are the route to
  relocalization, loop closure and re-association; see "Start here" for
  the plumbing.
- No re-association after occlusion: a returning point is a new track and a
  duplicate landmark.

## Cache boundary

Schema 3 stores frame identity, source dimensions, float32 coordinates/scores,
float16 or float32 descriptors, and extraction timings. It has no landmarks
table or tracking columns. Enqueue rejects derived tracking data. The read-only
connection loads observations by timestamp and PTS; WAL permits reads alongside
the writer. Reopening always rebuilds tracks from frame zero.

Cache metadata validates source path/size/mtime, engine path/size/mtime/content
fingerprint (FNV-1a), extractor version, input dimensions, keypoint count,
threshold, and encoding. Source identity avoids hashing multi-gigabyte videos
at startup; content changes preserving size and mtime are not detected.
Tracking parameters are excluded. Legacy schema versions are rejected before
any writes, and old default database paths are not reused automatically.

## Transport and diagnostics

A separate decoder seeks and decodes to the target, then becomes the playback
decoder. The tracker, flow reference, trails and selection reset, and the target
frame is processed immediately. Play and Step continue from that frame. Restart
seeks to the beginning and starts playing. Cache writes are flushed before seeking;
new rows use monotonically increasing cache IDs instead of seek-local decoder
ordinals. Timestamp and PTS remain the observation lookup key. The UI
draws SuperPoint-supported tracks filled and coasting tracks hollow, at their
flow-predicted position. It removes dead tracks' trails immediately. The
sidebar shows matched, coasted and new counts.
Correction distances and lifetime counts are runtime diagnostics. Flow
predictions and corrections are available for later analysis.
The Point view selector can instead show all raw SuperPoint detections in red,
before the tracking cap, with trails and flow arrows hidden. It changes display
only; tracking continues in the background.

## Remaining weaknesses and experiments

- Detector repeatability is now the dominant loss. A k=3,840 engine (the
  TensorRT TopK limit) lengthened median tracks at 120 s from 27 to 37, but
  SuperPoint time rose from ≈16.5 to ≈46 ms. A cheaper route is track-guided
  re-detection: export the dense score map and search it near each flow
  prediction.
- SuperPoint positions are integer pixels at 1024×576 (1.875 source px).
  Position fusion (flow sigma 0.3) removes much of that jitter; subpixel
  refinement in the ONNX export (quadratic fit on the 3×3 score
  neighbourhood) would stack with it and tighten the radius gate.
- Coasted positions come from flow only. The descriptor gate limits
  reacquisition errors, but a coasting track can drift for up to 5 frames on
  unreliable flow. Enable `--flow-backward` if that matters more than speed.
- Radius 8 gave longer tracks than 6 at the same match-similarity p10. The
  default stays at 6 until this is checked against geometric verification.
- Display, inference and flow still execute sequentially on separate streams;
  they could overlap across frames.
