# Native workbench architecture

## Start here: the tracking and pose system as of 2026-10-01

This section is the handoff for work on robust camera tracking, track
re-association after occlusion, and a camera motion model. The dated
sections below keep the history and measurements; where they disagree, this
section is current.

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
    undistort (k1) → init (E/H RANSAC) → per-frame pose (Huber LM from constant velocity
    / constant position; P3P RANSAC recovery) → keyframes (incl. emergency): triangulate,
    sliding-window BA (8 KF), cull (landmarks leave the local map, never the map); no pose →
    coast on the motion model + relocalize in the local map, then by descriptor over the whole
    map (PlaceIndex) → resume that segment; nothing found → new segment
            ▼
  OdometryFrameResult (pose, covariance, confidence, inlier/outlier track IDs), trajectory, map
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
  changed too. The cleaner route is aliasing at the VO level (re-keying a
  landmark to the new track ID).
- **Only supported observations reach the VO.** Coasted positions (flow-only)
  are dropped in `tracked_frame_from()`.
- **Track positions are fused, not exact detections** (since 2026-09-30,
  default `flow_sigma_px = 0.3`). Never identify "the detection a track
  used" by position equality. The bench did that and reported continuation
  0.005 instead of 0.88. Use the tracker's `kind`/`superpoint_supported`,
  or the record's `detection` index (the supporting detection, −1 when
  coasted).
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

Death is final in the tracker: a point that reappears after occlusion gets
a new ID. The VO re-keys the old landmark to the new track (re-association
within the local map; relocalization after an occlusion).

### How a landmark lives and dies (VO)

In `VisualOdometry::Impl` (`src/visual_odometry.cpp`):

- Created at initialization (both init frames) or in `triangulate_new()` at
  a keyframe, from the oldest window keyframe that saw the track, i.e. the
  widest pair (≥ 0.5° parallax, ≤ 3 px error in both views).
- Per frame, `track()` gathers the current observations of tracks that have
  landmarks, solves the pose, and splits them into inliers/outliers
  (`OdometryFrameResult::pose_inliers/pose_outliers`; the GUI's "Pose
  inliers" point view draws them green/red). The pose starts from a
  constant-velocity prediction, then constant position; if neither keeps
  half the correspondences, P3P RANSAC (`vo::p3p`, Grunert;
  `vo::ransac_pnp`, 4 px) re-estimates it from the same correspondences.
  Without a pose (fewer than 20 inliers) the frame is coasted, see the
  motion model below.
- At a keyframe, outlier landmarks are **deleted** (`landmarks.erase`,
  they're usually short-baseline depths and get re-triangulated).
- **Nothing is retired** (since 2026-10-01). `landmarks` holds every landmark
  of the run, all segments, with its sightings and latest descriptor. The
  **local map** (`local`, a set of ids) is the part being tracked: the current
  segment's landmarks that a live track observes or that one of the last
  `local_map_keyframes` local keyframes saw. Only the local map is projected
  into frames, re-associated at a pose and bundle-adjusted, so the per-frame
  cost does not grow with the map.
- `cull()` (local map only): observations that fail 2× the reprojection
  threshold are dropped; a landmark with < 2 observations is **deleted**; a
  landmark whose track is gone and which no local keyframe observes **leaves
  the local map** and stays in the map, where the search without a pose prior
  finds it again (below).
- When coasting runs out (`lose()`), and on resolution change,
  `end_segment()` empties the local map; the segment's landmarks stay. The
  next segment starts from the current frame with a new frame and scale,
  unless an earlier segment is found first (`resume()`).
- `MapPoint` (public) carries: track ID, position, mean RGB of keyframe
  observations, segment, keyframe count, first/last frame, `local`.
  `map()` returns every landmark by id, `landmark_observations()` the
  keyframes that observe each, `tracked_landmarks()` the few hundred tracks
  that observe a landmark now (for per-frame use); `write_map_csv()` exports
  the map.

### Camera motion model, pose uncertainty, coasting and relocalization

Added 2026-10-01 (`track()`, `relocalize()`, `coast()` in
`src/visual_odometry.cpp`; `pose_covariance()` and `MotionModel` in
`src/vo_geometry.*`).

- **Pose error convention.** δ = (ω, v) with `T_true = exp(δ) · T`
  (`perturb`): ω rotates about the camera centre, v moves the camera, both in
  camera axes; rotation in radians, translation in the segment's map units.
- **Tracked pose.** Covariance = s² (Σ JᵀJ)⁻¹ over the inliers, s² their
  residual variance (Σ|e|² / (2n − 6)). Landmarks are taken as exact, so it is
  optimistic. On synthetic data the normalized squared error averages 5.97
  (6 when consistent; `pose_covariance_is_consistent`).
- **Motion model.** Constant velocity with white-noise acceleration, per
  frame: `T[k+1] = exp(a) V T[k]`, `V[k+1] = exp(a) V`, a ~ N(0, Q). The
  prediction is `velocity * last_pose` as before; `MotionModel` carries its
  covariance (pose, velocity and their cross term). n frames without a
  measurement give P + n(C + Cᵀ) + n²Pv + Q·n(n+1)(2n+1)/6, so the pixel
  uncertainty grows like n^1.5 once Q dominates.
- **Q is estimated per segment** from the one-step prediction errors e of
  tracked frames. Pose jitter S makes consecutive errors negatively
  correlated (E[e eᵀ] = Q + 6S, E[e_k e_{k−1}ᵀ] = −4S), so
  Q = E[e eᵀ] + 1.5·E[e_k e_{k−1}ᵀ], exponentially weighted over about 60
  frames and projected to positive semidefinite. Without the correction the
  jitter is extrapolated as acceleration.
- **Confidence.** `OdometryFrameResult::pose_covariance` (6×6),
  `pose_sigma_px` (RMS per-axis standard deviation of the landmarks'
  projections under that covariance) and `confidence` =
  1 − exp(−τ²/(2σ²)) with τ = `reprojection_threshold_px`: the probability
  that a landmark projects within the inlier threshold of where it is.
  `TrajectorySample` and the trajectory CSV carry `predicted` and
  `confidence`.
- **Coasting.** A frame without a pose reports the prediction
  (`state == coasting`, `predicted`), keeps the local map and makes no
  keyframe. `relocalize()` projects the local landmarks that have no live
  track with the predicted pose and matches them by descriptor (mutual best,
  cosine ≥ 0.7) to tracks without a landmark, within 3σ of the predicted
  uncertainty (clamped to 16–256 px), then runs P3P RANSAC over those matches
  plus the live correspondences. With ≥ 30 inliers the matched tracks take
  over their landmarks (`associate()`), a keyframe ties the frame in, and the
  coasted frames' poses are replaced by the constant-velocity path between
  the tracked poses on either side. The segment ends when σ exceeds 256 px or
  after 90 frames; the unconfirmed coasted frames are then removed from the
  trajectory.
- **Calibration on real clips.** See `baselines/README.md` (v4). The model
  is conservative on handheld video and too tight for a sustained smooth
  acceleration (the synthetic occlusion test: 14.8 px off at σ = 4.3 px after
  20 frames; the 16 px floor covers it).

### Search without a pose prior (place recognition)

Added 2026-10-01 (`src/place_index.*`; `place_matches()`, `resume()`,
`search_places()` and the second half of `relocalize()` in
`src/visual_odometry.cpp`).

- **`PlaceIndex`** holds the latest descriptor of every landmark (updated
  whenever its track is a pose inlier). `match()` is brute force and exact:
  queries × entries cosine in blocks of 2,048 entries on up to 8 threads,
  mutual best, cosine ≥ `min_descriptor_similarity`, cosine distance ≤
  `place_ratio` × the runner-up's. It is the seam for an approximate index or
  a different descriptor.
- **A pose from matches alone**: P3P RANSAC (`place_ransac_iterations`, 4 px)
  per segment, most matches first, refined by `optimize_pose` on the RANSAC
  inliers only (over all matches the robust fit is pulled off: 95 raw inliers
  became 48); accepted with ≥ `place_min_inliers` (20). On fr1_room the
  count is 3–7 where the place is new and 14–95 on the revisit.
- **Frame without a pose from its tracks** (`track()` → `relocalize()`): the
  search around the predicted pose runs first, as before. When it fails, the
  frame's tracks without a landmark are matched against the whole map. A pose
  in the current segment relocalizes as the prior-gated search does, but the
  ≥ 20 inliers must be descriptor matches: the live correspondences agree with
  any pose near the last one (fr1_room frame 848 passed with 22 inliers
  counting them, kept one segment and doubled the ATE). A pose in another
  segment ends the current one and calls `resume()`.
- **While a new segment initializes** (`initialize()`): the same search on
  every frame, over all landmarks; a pose resumes that segment instead of
  starting another.
- **`resume()`**: the frame becomes a keyframe of the old segment. The local
  keyframes become the old keyframes with the most sightings of the landmarks
  found (up to `local_map_keyframes`, most last), so the bundle-adjustment
  window ties the new keyframe to keyframes that share landmarks with it, and
  the local map is what those keyframes saw. The matched tracks take over
  their landmarks; the motion model restarts with zero velocity.
- **While tracking** (`search_places()`, every `place_search_interval`
  frames): all tracks against the landmarks outside the local map. Matches in
  this segment that project within `reassociation_radius_px` of the track
  under the tracked pose are re-associated like local ones
  (`place_reassociated`). A P3P pose from the remaining matches means the
  place was mapped before, in this segment with drift or in another segment.
  It is reported (`place_segment`, `place_inliers`, `place_offset_px`) and
  **not corrected**: that needs a pose graph or a wider bundle adjustment,
  and segment merging (next step).

Measured 2026-10-01 (TUM + dreamworks, three start frames each, against
`baselines/v4`; scratch run, not yet a saved baseline):

- With the search off (`place-interval -1`) the trajectories are byte-identical
  to v4 in all 12 runs compared, and the VO costs the same: keeping every
  landmark is free.
- TUM ATE is unchanged within the spread (fr3 5.3 ± 2.4 → 6.0 ± 1.4 cm,
  fr1_xyz 1.7 → 1.8, the others equal): nothing is corrected yet.
- dreamworks from frame 300: 2 segments → 1. The loss at frame 675 is now a
  relocalization at 679 by descriptor (25 inliers of 86 matches). The
  relocalizations after the columns come 2 frames earlier (571 instead of 573).
- Landmarks re-associated from outside the local map: 0–51 per run (most on
  fr1_xyz and fr3). Drift is usually above the 2 px gate.
- Revisits found while tracking: fr1_room 7 (segment 0 seen from segment 1),
  fr3 2–13, fr2_desk 0–9, fr1_xyz 5–11, at a median 7–114 px from where the
  tracked pose puts those landmarks. That offset is the drift a correction
  step would remove.
- Cost: one search is about 50 ms at 20,000 landmarks (800 queries, 8
  threads). The VO goes from 2.1 to 3.5 ms/frame on TUM, with a 30–100 ms
  frame every `place_search_interval` frames on the UI thread.

Not done:

- **Loop correction and segment merging.** `search_places()` finds revisits
  and reports them; nothing moves the old keyframes or joins two segments.
- **Feeding the pose back to the tracker.** For tracks with landmarks, the
  predicted pose gives a predicted pixel (`K.project(T_pred * X)`, then
  *distort* with k1, since the tracker works in distorted pixels). The
  tracker's `predict` kernel takes only flow today; add an optional per-track
  prediction array and use it as a second gate. Frame-to-frame, flow is
  usually better than a pose prediction, so treat the pose as a fallback.
- **Outlier feedback.** The tracker never learns that the VO rejected a
  track (`pose_outliers`).
- Threading: the VO, keyframe BA (≈10–12 ms) and the place search (≈50 ms
  every 30 frames) run synchronously on the UI thread; the lens estimator shows the worker pattern
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
   points that returned as new tracks (no re-association).

Baseline, `disney_04` (1920×1080, 60 fps, 1,798 frames; lens 87.9° /
k1 −0.054; defaults, flow sigma 0.3): continuation 0.883, median track
length 15, VO inlier ratio 0.897, median reprojection 1.12 px, epipolar RMS
1.082 px, 0 losses, ≈8,500 map points (≈7,700 retired). With fusion off:
0.881 / 17 / 0.840 / 1.32 px / 1.215 px. Reproduce:

```bash
B=native_workbench/build/app-debug
$B/slam-native-tracking-bench ~/Downloads/chopped/disney_04.mp4 \
  native_workbench/models/superpoint-1024x576-k2048.engine 0 1800 \
  hfov 87.9 k1 -0.054 export-tracks tracks.csv export-map map.csv export-trajectory traj.csv
$B/slam-native-focal tracks.csv          # epipolar RMS curve, FOV, k1
$B/slam-native-snapshot VIDEO ENGINE START_S FRAMES out.ppm   # headless render of the real UI
```

To compare versions of the whole pipeline, `tools/baseline.py OUT --build
BUILD_DIR` runs the bench from frame 0 over every TUM sequence and the
dreamworks clip and writes `trajectory.csv` + `run.log` per sequence,
`summary.csv` (`tum_eval` metrics: posed frames, segments, ATE, RPE) and
`manifest.json` (commit, binary, command lines). Runs are deterministic, so
two snapshots differ only by the code or the `--set KEY VALUE` options.

ORB-SLAM3 is the independent reference (`scripts/run_orbslam3.sh`); on
`disney_04` at 60° it traced the same path shape as this VO did.

### Tools and tests added 2026-09-30

- `slam-native-focal`: FOV + k1 estimate from a tracks CSV.
- Bench keys: `flow-sigma`, `det-sigma`, `hfov`, `k1`, `fx fy cx cy`,
  `kf-min`, `kf-max`, `kf-emergency-ratio`, `window`, `place-interval`,
  `place-ratio`, `place-inliers`, `export-map`,
  `export-features PATH` (raw detections + descriptors in the feature-cache
  format).
- Tracks CSV: `frame_index,timestamp_ns,track_id,x,y[,r,g,b]`. Map CSV:
  `track_id,segment,x,y,z,r,g,b,has_color,keyframe_observations,first_frame,last_frame,local,landmark_id`
  (`track_io.hpp`).
- CTest: `native_focal_estimation` (synthetic FOV/k1 recovery, background
  worker parity), map + colour check, segment resume without a pose prior,
  re-association outside the local map and P3P checks (exact
  recovery, RANSAC with 60% outliers) in `native_visual_odometry`, Kalman
  fusion and per-track descriptor checks in `native_gpu_flow_tracker`.

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
  `flow-sigma`, `det-sigma`, `hfov`, `k1`, `fx`/`fy`/`cx`/`cy`, `kf-min`,
  `kf-max`, `kf-emergency-ratio`, `window`, `export-tracks`,
  `export-trajectory`, `export-map`, `export-features`.

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
`# size W H`, loaded with `read_tracks_csv()`.
`slam-native-tracking-bench ... export-tracks PATH` writes the native
tracks in that format, and `export-trajectory PATH` the poses (camera
centre plus camera→world quaternion).

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
   constant-velocity prediction, falling back to constant position; when
   neither keeps half the correspondences, P3P RANSAC (4 px) re-estimates
   the pose from the same correspondences. Too few inliers after that means
   lost: a new segment starts and reinitializes. Rotations are
   re-projected onto SO(3) after every composition. Without that, the
   prediction `A·B⁻¹·A` (with Rᵀ as inverse) amplified round-off shear
   about 2.4× per frame, which `exp(ω)·R` can never remove. This was a real
   bug found by the noise-free regression test.
3. *Keyframes* (6–20 frames apart; earlier if inliers fall below 75% of the
   last keyframe's landmarks; an emergency keyframe, ignoring the minimum
   interval, when they fall below 40%, so the map refills in fast motion
   before tracking is lost):
   - Add inlier observations.
   - Discard landmarks of live tracks that became outliers. They are almost
     always short-baseline depths, and they are re-triangulated.
   - Triangulate tracks from the oldest window keyframe that saw them (the
     widest pair), with ≥ 0.5° parallax and ≤ 3 px error in both views.
   - Run a sliding-window bundle adjustment over the last 8 keyframes: Schur
     complement, Levenberg–Marquardt, Huber loss; the two oldest keyframes
     are fixed to hold the gauge and scale.
   - Cull points with bad observations. Points no longer observed leave the
     local map but stay in the map (see "Search without a pose prior").
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
  frustum and the map points;
- map points in and outside the local map (*Not local*, optionally dimmed), coloured plain,
  by image colour or by age (frames since the landmark's first keyframe:
  yellow new, blue at the *Age span* slider's value or older), a grid on the
  X–Z plane at the first keyframe's height (monocular VO knows no floor);
  points are solid, unlit (constant-colour) icosahedrons drawn by `PointCloudRenderer`
  (`src/point_cloud_renderer.cpp`): one instanced GL draw per set into a 4×
  MSAA, depth-tested offscreen target, composited between the grid and the
  path/frustum overlays. The shader applies the display rotation, segment
  filter and world-space size (slider; segment units, ≥ ~1 px), so
  both sets are uploaded each frame;
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
- No loop closure, and relocalization only around the motion model's
  prediction (see "Camera motion model" above). When coasting runs out, a
  new segment starts with a new frame and scale.
- Frames before initialization have no pose.
- Scale drift (above) grows over long runs.
- Intrinsics come from the self-calibration estimate (or a guess). Refining
  focal length and k1 inside bundle adjustment is the next accuracy lever.
- Rolling shutter and moving people are only handled by the robust losses.
- Keyframe mapping runs on the UI thread. It could move to a worker thread,
  and PnP/RANSAC could use the GPU-resident tracks.
- Descriptors (`TrackedFrame::descriptors`, from
  `FrameFeatures::track_descriptors`) drive re-association and
  relocalization; loop closure would use them too.

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
