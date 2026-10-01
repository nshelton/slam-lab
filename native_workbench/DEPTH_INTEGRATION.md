# Depth inference integration: design proposal

Status: **draft for review** (2026-09-30). Builds on the existing
`DepthEstimator` (`include/slam_native/depth_estimator.hpp`) and on
`CONFIDENCE_DESIGN.md`; read `ARCHITECTURE.md` "Start here" first.

Goal: use the monocular metric depth network to make the camera pose and map
better (metric scale, faster and more robust initialization, occlusion
reasoning, depth priors) and to show the scene densely, without making the
tracker-agnostic VO depend on any particular depth model.

## Where we start

**Depth inference exists but feeds only the Depth window.**

- `DepthEstimator` runs a TensorRT engine (Depth Anything V2-S indoor/outdoor,
  Metric3D v2 ViT-S) on its own CUDA stream. `submit(frame, focal_px)` copies
  the frame and returns; `poll()` hands back a `DepthMap` a UI frame or so
  later. While busy, `submit()` ignores frames, so not every frame gets depth.
- The network sees the *upright* frame (display rotation applied).
  `DepthMap::at_source(x, y)` samples bilinearly at **native source pixels**,
  the same lens-distorted coordinates as tracks and detections, so sampling at
  a track needs no extra plumbing.
- Cost on the 2080 Ti (trtexec): DA-V2-S 518×294 ≈ 2.6 ms; Metric3D 1064×616
  ≈ 23 ms. The per-frame budget at 30 fps is 33 ms, of which SuperPoint takes
  ≈ 16 ms, NVOF ≈ 3.5 ms and tracking ≈ 0.4 ms of GPU. DA-V2-S fits on every
  frame; Metric3D does not, alongside SuperPoint.
- Metric3D predicts depth for a canonical 1000 px focal length and is
  rescaled with our FOV estimate. DA-V2 ignores intrinsics, so its metric scale
  is whatever its training camera implied.

**The VO is monocular.** Each segment has an arbitrary scale (median depth of
the first keyframe = 1), fixed in BA by a second fixed keyframe.
Initialization needs ≥ 1° median parallax and ≥ 60 well-conditioned points. On
`dreamworks.MOV` re-initializing after a loss takes 7–20 frames ("waiting for
parallax / motion"). Points near the focus of expansion (< 0.5° parallax) are
never triangulated, which is most of the view when walking forward. Since
today, landmarks persist (dormant, relocalization) and every frame reports the
landmarks that project into view but are untracked (the magenta rings), with
no way yet to tell *why* they are untracked.

## What depth is good for here, in priority order

| # | Use | Problem it addresses | Needs depth at | Changes the solver? |
|---|---|---|---|---|
| 1 | **Measure the network against the VO** | We don't know the model's bias or noise on our footage | keyframes | no |
| 2 | **Metric scale per segment** | Arbitrary, unrelated scale per segment; scale drift is invisible | keyframes | no (output only) |
| 3 | **Dense keyframe point clouds** in the 3D view | The map is sparse (≈ 10k points) and hard to read | keyframes | no |
| 4 | **Occlusion test for untracked landmarks** | Magenta rings: is the point behind the column, or just missed? | current frame | no (bookkeeping) |
| 5 | **Depth gate** on re-association and relocalization matches | Wrong merges in repetitive texture (railings) | current frame | gating only |
| 6 | **Single-frame (re)initialization** | 7–20 lost frames after every loss; pure rotation never initializes | init frame | yes |
| 7 | **Depth priors in BA**, low-parallax landmarks | No landmarks near the focus of expansion; scale drift | keyframes | yes |

Items 1–3 are read-only consumers of the VO: safe, and they produce the
numbers the later items need (how far to trust the network). Items 4–5 only
change bookkeeping and gates. Items 6–7 change the estimator and come last,
behind the confidence work.

## Contracts

**Coordinates.** Sample depth at the track's *native source pixel*
(distorted), exactly where the tracker reports it: `DepthMap::at_source`.
Inside the VO, a pixel with depth becomes a 3D point as
`X_cam = z · K⁻¹ · undistort(p)`. This assumes the network outputs **z-depth**
(distance along the optical axis), not range along the ray. That is the
convention for DA-V2 and Metric3D, but it should be verified once (Phase 0:
residuals vs image radius). With k1 ≈ −0.03 the distortion effect on depth is
second order, but the ray direction must still come from the undistorted
pixel.

**Depth is an optional observation attribute, never required.** Like
descriptors and colours:

```cpp
struct TrackObservation {
  ...
  float depth_m{0};        // network z-depth at (x, y), metres; <= 0: unknown
  float depth_sigma_m{0};  // 1-σ; <= 0: use config (relative) default
};
```

Tracks CSV gains an optional `depth` column (written by the bench and the app
export, read by `read_tracks_csv`). `slam-native-vo-tracks` can then replay
depth-aided runs offline and deterministically. CSVs without it behave as
today.

**Sampling policy (app/bench side, not VO).** At each track position:

- Bilinear `at_source`; reject NaN/0 (padding, outside, `max_depth_m` clamp).
- **Edge test**: SuperPoint corners sit on depth discontinuities (column edges,
  railings) where networks smear depth. Take the 3×3 neighbourhood on the
  output grid. If `max/min > 1 + edge_ratio` (e.g. 0.1), report the
  **minimum** (the foreground) with an inflated σ, or drop it. Phase 0 decides
  which.
- Sky, glass and water (the pond in `dreamworks.MOV`) give confident nonsense.
  Rely on the VO-side consistency checks below, not on a classifier.

**Timing and frame identity.** A `DepthMap` carries `frame_index`. A depth
sample may only be attached to observations of **that exact frame**; never
reuse the previous map for the current frame (the camera moved).
Two modes:

- *Live (app):* run the VO one frame behind. Frame *f*'s `TrackedFrame` is
  held until depth for *f* has arrived or *f* was skipped (estimator busy),
  bounded at 2 frames. That costs one frame of pose latency and no extra
  copies, since tracks are host data. Alternatively, run the VO immediately
  and attach depth late through `VisualOdometry::add_depth(frame_index,
  samples)`, used only if *f* became a keyframe. Recommendation: delay by one
  frame. It is simpler, and items 4–5 need depth for the current frame anyway.
- *Bench/tests:* synchronous (submit, then wait for each frame), so runs are
  deterministic. GPU timing must never change a result in the bench.

Which frames get depth: with DA-V2-S, every frame. With Metric3D, the VO
needs depth mainly at keyframes, but the keyframe decision happens after the
frame's GPU surface is gone. Either keep a copy of the last N decoded frames
on the GPU (1920×1080 NV12 ≈ 3 MB each), or accept every-other-frame depth and
let a keyframe use depth only if it has some.

**IDs and the VO contracts are unchanged.** Depth never creates, merges or
removes tracks.

## Phase 0: measure the network against the VO (no behaviour change)

The VO's own landmarks with good parallax are a reference *up to one scale per
segment*, which is exactly what is needed to calibrate a metric-depth network.

For every landmark observation at a keyframe that has a depth sample:

```
r = log(z_network) − log(z_vo)        z_vo = (T_kf · X).z
```

Per segment, `median(r)` estimates the log scale s (VO units → metres), and
the spread of `r − s` is the network's relative error. Break it down by:

- **depth** (near / far), **image radius** (z-depth vs range, lens),
  **edge flag**, **landmark parallax** (to make sure the reference itself is
  good: use only landmarks with ≥ 2° parallax and ≥ 3 keyframe observations);
- **time within a segment**: s(t) per keyframe. A trend means VO scale drift,
  which the network can now *see* (scale drift measurement comes for free);
- **across segments** of the same walk: with metric scale, the camera speed
  should be continuous across a relocalization-free segment break. Walking
  ≈ 1.2–1.5 m/s is 4–5 cm per frame at 30 fps, a ground truth we can check by
  eye;
- **model**: DA-V2-S indoor vs outdoor vs Metric3D, on `dreamworks.MOV`
  (covered walkway, outdoor background, pond) and `disney_04`.

Deliverables: `tools/depth_vs_vo.py` (needs the tracks CSV `depth` column and
the map/trajectory exports, like `pose_propagation.py`), a bench line
`depth: scale per segment, relative error p50/p90, edge share`, and a choice of
`depth_relative_sigma` (expect 0.1–0.2) and of the edge policy. Decide the
model here, not before.

## Phase 1: metric scale per segment (output only)

The VO keeps solving in its own units (the solver and its tests are
unchanged), and estimates a per-segment scale online:

- At each keyframe, add that keyframe's robust `median(r)` (edge-free samples
  of good landmarks) to the segment's estimate: a running weighted median, or
  a 1-D Kalman filter on log scale with process noise for drift.
- Expose it: `OdometryFrameResult::metric_scale`, `metric_scale_sigma`,
  `VisualOdometry::segment_scale(segment)`. `MapPoint`/`TrajectorySample` are
  unchanged; consumers multiply. Exports gain `--metric` (trajectory and map in
  metres, with an `m_per_unit` column) and the trajectory view shows a metre
  grid and the camera speed (a sanity check).
- On relocalization the resumed segment keeps its scale (already the same
  frame). Segments that never reconnect now at least share units, so they can
  be shown side by side at the same scale.

## Phase 2: dense keyframe clouds in the 3D view

Back-project each keyframe's depth map, coloured from the frame, into the
trajectory view, so the scene is visible and not just the corner cloud.

- **Data**: per keyframe, a `KeyframeCloud { keyframe id, points in the camera
  frame (metres), rgb }`, subsampled on the output grid (every 4th pixel of
  518×294 ≈ 9.5k points), edge-filtered (drop discontinuity pixels, the
  "flying pixels" between column and background) and depth-limited (e.g.
  < 15 m, or < 3× the segment's median landmark depth).
- **Colour**: sample the frame with a `ColorSampler`-style kernel at the
  back-projected pixels, on the GPU, while the frame is alive. Upright vs
  native: build the cloud on the depth output grid and map through
  `DepthMap`'s transform.
- **Pose**: keyframe poses change while in the BA window (`stable_prefix()`).
  Store clouds in the *keyframe's camera frame*; the renderer draws each with
  the keyframe's current pose and the segment's metric scale inverted
  (cloud metres → VO units). One instanced draw per keyframe in
  `PointCloudRenderer`, with a per-instance-set model matrix. Re-upload only
  when a cloud is created.
- **Budget**: 150 keyframes × 9.5k points ≈ 1.4 M instances. That is too many
  icosahedra, so this needs a point-sprite path in the renderer, or a voxel
  downsampling of the accumulated cloud (keep a 5 cm voxel hash; insert new
  keyframes; old keyframes outside the BA window are final). TSDF fusion is a
  later, separate project.
- **Consistency check for free**: dense clouds from overlapping keyframes must
  coincide. Visible doubling means bad scale, bad poses or bad depth, which
  makes it a debugging view too.

## Phase 3: occlusion test for untracked landmarks; depth gate

For each `untracked_landmarks` entry the VO already has the predicted depth
`z_pred = (T · X).z · scale`. With the depth sample `z_img` at its projection:

| Test | Meaning | Action |
|---|---|---|
| `z_img < z_pred · (1 − k·σ)` | something is **in front** (column, person) | **occluded**: keep dormant, never cull for being unseen, don't let it re-associate |
| `|log z_img − log z_pred| ≤ k·σ` | the surface is there and **visible**, yet untracked | **missed**: detector miss or stale landmark; count consecutive misses → cull after N, or hint the tracker (below) |
| `z_img > z_pred · (1 + k·σ)` | we see **through** where it should be | the landmark is wrong (or is a thin structure); count against it |

- Add `ProjectedLandmark::visibility { unknown, occluded, visible, behind }`
  and colour the magenta rings by it in the 2D view (e.g. occluded = dashed or
  blue). This answers the open question from the re-association work.
- **Depth gate** for re-association and relocalization matches: reject a match
  whose landmark depth disagrees with the depth at the matched pixel (in log
  space, k·σ). Relocalization's 256 px window and the railings are where wrong
  mutual-best matches come from. Before the segment has a scale (relocalizing
  early), use the lost segment's scale.
- **Tracker hint** (optional, later): "visible but untracked" landmarks can be
  sent to the GPU tracker as predicted positions with the stored descriptor,
  so a SuperPoint detection there is claimed by the old landmark instead of
  seeding an anonymous track. This needs a per-track prediction input on the
  tracker (`ARCHITECTURE.md`, "Feeding the pose back to the tracker").

Measure with the bench: the share of untracked-in-view landmarks per class
over the column events (occluded should dominate during a column and vanish
after it), and the re-association / relocalization inlier rates with and
without the gate.

## Phase 4: single-frame (re)initialization

With depth, a segment can start from **one** frame: back-project the tracked
observations (edge-free, depth-limited) as landmarks at metric scale, set the
pose to identity, and track from the next frame. No parallax wait, and pure
rotation stops being fatal.

- Use it only after a loss with no successful relocalization within K frames
  (relocalization into the old segment is still preferred: it keeps the
  frame), and at the start of a video that has depth.
- The landmarks start with depth uncertainty σ_z = `depth_relative_sigma · z`
  and are *provisional*: they are replaced by triangulated positions once
  they have parallax (Phase 5), and the segment's scale is the network's
  rather than the median-depth convention.
- Expected effect on `dreamworks.MOV`: the 7–20 frames after each loss without
  a pose drop to ~1. The first poses are only as good as the network depth, so
  the inlier rate right after init will show whether that is good enough.

## Phase 5: depth as a prior in BA

This is where depth changes the estimate, so it waits for the confidence work
(`CONFIDENCE_DESIGN.md` Levels 2–3: weighted solvers and landmark
covariance).

- Add a residual per landmark observation with depth:
  `(log z_net − log (s · z_vo)) / σ_rel`, with the segment log-scale *s* as a
  BA variable. The second fixed keyframe (the scale gauge) is then no longer
  needed: depth observes scale.
- Landmarks below the parallax threshold (near the focus of expansion: today
  never created) can be created from depth with a wide prior, and refined as
  parallax accumulates. That gives more landmarks in exactly the forward-walk
  case.
- Keep the prior weak (σ_rel from Phase 0, floored) and robust (Huber in log
  space): network depth has correlated, scene-dependent bias, so it must not
  override geometry where geometry is good.

## API summary

```cpp
// visual_odometry.hpp
struct TrackObservation { ...; float depth_m{0}; float depth_sigma_m{0}; };
struct VisualOdometryConfig {
  ...
  bool use_depth{true};                 // master switch; no effect without depth samples
  double depth_relative_sigma{0.15};    // default σ_z / z (Phase 0 sets it)
  double depth_gate_sigmas{3.0};        // occlusion / gate threshold k
  bool depth_initialization{false};     // Phase 4
  bool depth_priors{false};             // Phase 5
};
struct OdometryFrameResult {
  ...
  double metric_scale{};                // metres per VO unit for this segment; 0 = unknown
  double metric_scale_sigma{};          // relative
  // ProjectedLandmark gains: enum Visibility { unknown, occluded, visible, behind } visibility;
};
class VisualOdometry { ... double segment_scale(int segment) const; };

// app side
struct DepthSample { float metres; float sigma; bool edge; };
DepthSample sample_depth(const DepthMap&, float x, float y, float edge_ratio, float relative_sigma);
```

`DepthMap` stays an app/bench type. The VO only ever sees numbers on
observations, so other depth sources (a stereo or LiDAR CSV) can drive the
same code.

## Risks

- **Scale bias per model and scene.** DA-V2's metric head assumes its training
  camera (and indoor vs outdoor), and Metric3D relies on our FOV estimate.
  Both can be 10–30% off in absolute scale, and the bias changes when the
  scene changes (walkway vs courtyard). Phase 1 therefore *estimates* a scale
  rather than trusting metres directly, and Phase 0 measures whether it is
  stable within a segment.
- **Correlated errors.** Depth errors are spatially and temporally
  correlated (whole surfaces off together), so averaging many samples doesn't
  shrink the error as 1/√n. Use medians, floor the uncertainty, and treat
  depth as a weak prior.
- **Edges and thin structures.** Columns, railings and foliage are exactly
  where tracks live and where depth is worst. The edge test is essential, not
  optional.
- **Flicker.** Per-frame inference is not temporally consistent. Consumers
  that compare frames (occlusion classes, scale per keyframe) must tolerate
  it, which favours robust statistics over single samples.
- **GPU contention.** Metric3D at 23 ms does not fit alongside SuperPoint at
  30 fps. Use DA-V2-S live; use Metric3D at keyframes only, or offline.
- **Determinism.** "Skipped while busy" makes live runs timing-dependent.
  Bench and tests must run depth synchronously.
- **Water, glass, sky.** These are confidently wrong. The VO consistency check
  (Phase 0 residuals, the `behind` class) has to catch them. They should show
  up in Phase 0 on the pond in `dreamworks.MOV`.

## Open questions

1. Which model and resolution? Decided by Phase 0 on our clips, not by
   benchmarks.
2. Do we want metres in exports and the UI by default, or as an option
   (`--metric`) until the scale estimate has proven stable?
3. Dense view: point sprites in `PointCloudRenderer`, or a voxelized
   accumulated cloud? (Both, eventually: sprites for the BA window, voxels for
   final keyframes.)
4. Should depth go into the feature cache (f16 518×294 ≈ 300 KB/frame), so
   cached replays have it, or only per-track samples in the tracks CSV?
   Per-track samples are enough for Phases 0–1 and 3–5; Phase 2 needs maps at
   keyframes.
5. Depth-based initialization makes the network's scale the segment's scale.
   Should that replace the median-depth convention everywhere, so that every
   segment is metric from the start?
