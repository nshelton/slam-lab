# Native workbench architecture

## Tracking and keypoint decisions — 2026-09-24

This is the current native exploratory pipeline; it does not change the Python
offline matching and reconstruction workflow. It supersedes the 2026-09-23
baseline (single-pass spatial association, unmatched tracks die immediately).

- SuperPoint supplies every observation. NVIDIA optical flow (4×4 grid)
  supplies motion only; grid cells never seed tracks.
- Flow predicts, SuperPoint corrects. A track's observed position is always an
  exact SuperPoint detection. When no detection supports a track, it may
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
  `coast`, `fb`, `rounds`, `tracks`, `quality`, `backward`, `k`.

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
   - Cull points with bad observations, and points that can never be
     observed again.
4. Per-frame poses are stored relative to their keyframe, so the displayed
   trajectory follows bundle-adjustment corrections.
   `stable_prefix()` lets the viewer refresh only the part that can still
   change.

Intrinsics default to a 60° horizontal field of view with a centred
principal point, in source-video pixels (the same assumption as the Python
solver). `--hfov` or `--intrinsics FX FY CX CY` override it. There is no
distortion model.

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
- orbit, top (X–Z), side and front views. Wheel zooms, left-drag orbits,
  right-drag pans, double-click fits. *Follow* keeps the camera centred.
  *All segments* overlays earlier segments; each has its own arbitrary
  frame and scale.

Seeking or restarting resets the trajectory, because track IDs restart.
`slam-native-snapshot VIDEO ENGINE START FRAMES OUT.ppm` renders the real UI
headlessly for inspection.

**Weaknesses and next steps.**
- No relocalization or loop closure. After a loss, a new segment starts
  with a new frame and scale.
- Frames before initialization have no pose.
- Scale drift (above) grows over long runs.
- Intrinsics are guessed and lens distortion is ignored. Calibration or
  focal-length self-calibration in bundle adjustment is the biggest
  accuracy lever.
- Rolling shutter and moving people are only handled by the robust losses.
- Keyframe mapping runs on the UI thread. It could move to a worker thread,
  and PnP/RANSAC could use the GPU-resident tracks.
- Descriptors (already cached per SuperPoint observation) are the route to
  relocalization and loop closure.

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
  Subpixel refinement would tighten the radius gate.
- Coasted positions come from flow only. The descriptor gate limits
  reacquisition errors, but a coasting track can drift for up to 5 frames on
  unreliable flow. Enable `--flow-backward` if that matters more than speed.
- Radius 8 gave longer tracks than 6 at the same match-similarity p10. The
  default stays at 6 until this is checked against geometric verification.
- Display, inference and flow still execute sequentially on separate streams;
  they could overlap across frames.
