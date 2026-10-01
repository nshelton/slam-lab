# Depth inference integration: design proposal

Status: **draft for review**, revision 2 (2026-09-30). Builds on the existing
`DepthEstimator` (`include/slam_native/depth_estimator.hpp`) and on
`CONFIDENCE_DESIGN.md`; read `ARCHITECTURE.md` "Start here" first. Next step:
**dense depth at keyframes** (Phase 2, with the minimum of Phases 0–1 it
needs), then provisional landmarks.

### Revision 2: what changed since the first draft

- **Landed in the VO (same day):** landmarks have their own IDs
  (`MapPoint::landmark_id`, a `track → landmark` index; tracks attach to and
  leave landmarks), and re-association can **merge** duplicate landmarks. Any
  per-landmark depth state below lives on the landmark, never on a track.
- **Landed from the confidence work** (`CONFIDENCE_DESIGN.md` phases 0–1):
  observations carry `TrackObservation::sigma_px`, and
  `use_observation_sigma` is now **on by default**, so trajectories differ
  from before. Compare with `--set use_observation_sigma=0`. Each landmark has
  a geometry-only covariance (`update_quality`, called in `cull`), and
  `MapPoint::confidence` = precision × a view-count factor
  (`confidence_two_view`), so 2-view, low-parallax points already score low.
- **New sections by the confidence session:** Phase 0 cues, Phase 0b
  (self-calibrated per-pixel σ map), the circularity rule, and a proposed
  answer to Q5. They are kept below as written.
- **Decisions for the first implementation (user, 2026-09-30):**
  DA-V2-S on **every frame** (indoor/outdoor weights per clip); live app
  **delays the VO by one frame** so depth always matches its frame; dense view
  as **point sprites per keyframe** (voxels later); **this session** does
  Phase 2 with the minimal Phases 0–1 (per-track sampling, tracks CSV `depth`,
  per-segment scale, `KeyframeDepthStore`, dense view). The confidence session
  builds Phase 0b on the sampled data. Provisional landmarks follow,
  coordinated with it.
- **New here:** the rotation failure on `castle.mp4` (below), and
  "Provisional landmarks: inverse depth", the building block that Phases 4–5
  and the rotation fix share. It includes the constraints agreed with the
  confidence session.

Goal: use the monocular metric depth network to make the camera pose and map
better (metric scale, faster and more robust initialization, occlusion
reasoning, depth priors) and to show the scene densely, without making the
tracker-agnostic VO depend on any particular depth model.

## Implementation status (2026-09-30)

Done (Phase 2 with the minimal Phases 0–1):

- **Observation depth**: `TrackObservation::depth_m / depth_sigma_m`; tracks
  CSV `depth,depth_sigma` columns (optional columns now read by header name;
  `TrackColumns` on the writer).
- **Sampling** (`depth_sampling.hpp`): the policy above, including the 3×3
  edge test (foreground depth, 3× σ, or dropped) and clamp/padding rejection.
- **Bench**: `depth <preset or engine stem>` runs the model synchronously
  on every frame (≈ 3.9 ms/frame for DA-V2-S); `depth scale segment N` and
  `depth:` report lines.
- **VO scale** (output only): per-keyframe `keyframe_log_scale` /
  `keyframe_scale_samples` / `keyframe_scale_spread`, and a per-segment
  log-scale filter (`metric_scale`, `metric_scale_sigma`, `segment_scale()`).
  Reference landmarks: ≥ 3 keyframes, ≥ 2° parallax, non-edge sample.
- **Live app**: the VO runs one frame behind (pending queue, max 2 frames;
  drained in `advance()` too). Depth is sampled at the tracks, and colours
  for the cloud grid are sampled while the frame is alive.
- **Dense keyframe clouds**: `KeyframeDepthStore` (grid = every 4th output
  pixel, flying pixels and far points dropped, camera frame), aligned per
  pixel by the keyframe's `ScaleGrid` (provisional, then settled), and
  cross-checked with the last 4 keyframes in both directions (agree within
  6% / seen through / occluded = no vote; hidden when checked and never
  confirmed). Drawn by `PointCloudRenderer` as point sprites with each
  keyframe's *current* pose, re-uploaded when a cloud's version changes.
  Trajectory view: "Dense depth", size, "Consistent only". Bench: `dense 1`.
- CTest `native_depth`: grid/transform inversion for all rotations, the
  sampling policy, back-projection and edge dropping.

Measured (bench, DA-V2-S; the confidence session's Phase 0b agrees):

| Clip | network error on reference landmarks (log, p50) | keyframe log-scale p10/p50/p90 | trend per 100 frames |
|---|---|---|---|
| TUM fr3 long_office | 0.083 | 1.15 / 1.31 / 1.48 | 0.000 |
| disney_04 | 0.167 | 0.71 / 1.11 / 1.32 | −0.025 |
| castle 95–115 s | 0.148 | 0.88 / 1.12 / 2.04 | −0.12 |
| dreamworks | 0.077 | 0.34 / 1.68 / 1.82 | −0.20 |

- **Absolute scale is biased**: on TUM the ground truth gives 2.40 m/unit,
  the network 3.46 (+44 %, log +0.36), stable over the sequence. So metres
  stay an *estimate* (Phase 1), never trusted directly.
- **Cloud consistency** (`tools/depth_consistency.py`: neighbouring keyframes'
  depth projected into each other; median / p90 log disagreement at 1 and 4
  keyframes apart, DA-V2-S):

  | | disney_04 gap 1 | disney_04 gap 4 | fr1_room gap 1 | fr1_room gap 4 |
  |---|---|---|---|---|
  | per-keyframe scale (first version) | 0.034 / 0.105 | 0.065 / 0.231 | 0.052 / 0.206 | 0.089 / 0.538 |
  | **VO grid, settled** (current) | **0.023 / 0.082** | **0.046 / 0.178** | **0.034 / 0.151** | **0.057 / 0.318** |
  | offline best (landmark grid, final geometry) | 0.022 / 0.077 | 0.041 / 0.155 | 0.032 / 0.139 | 0.054 / 0.284 |

  The network's errors are low-frequency *shape*, not just scale: a 4×3
  log-scale grid per keyframe (`ScaleGrid`, fitted by the VO to the same
  reference landmarks, refitted when the keyframe leaves the BA window:
  `settled_*`) removes about a third of the disagreement, as much as the best
  single scale per pair. No radial (FOV) trend; edges are about 2% of pixels.
  Metric3D is worse after alignment (0.034 at gap 1) and 8× slower. Then the
  multi-view check hides points that a neighbouring keyframe sees through:
  disney_04 has 95% of points confirmed and 2.9% hidden; fr1_room 77% / 12.7%.
  fr1_room's remaining disagreement is the VO's own scale collapse in
  segment 0 (keyframe log-scale 0.5–5.4), which needs Phase 5, not depth
  processing.
- **The VO's scale drifts within a segment** on some clips (dreamworks:
  monotonic, ≈ 5× over the clip; TUM: none). One scale per segment is
  therefore wrong for output; clouds use per-keyframe scale. Fixing the drift
  itself is Phase 5's job (the depth prior), which this data motivates.

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

**Rotation starves the map (`castle.mp4`, ≈ 108.5–113 s, 60 fps drone).** The
camera pitches up towards the horizon at ≈ 13°/s while barely translating
(baseline / depth ≈ 0.14–0.49° per 0.5 s). Pose tracking stays good: inliers
≈ correspondences, median reprojection < 1 px. But the tracks that *have a
landmark* fall from ≈ 850 to ≈ 30. Old landmarks rotate out of view, and the
new content can't be triangulated: there's no keyframe pair in the window with
≥ 0.5° parallax. Faster keyframes make it worse, because they shorten the
window in time. The bench run scraped by at 29 correspondences
(`min_tracked_points` = 20); the app run was lost. Thresholds only move the
edge: pure rotation carries no depth information. Landmarks whose depth is
unknown but bounded (inverse depth) fix it, and depth from the network makes
them good immediately.

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

> **Addition (2026-09-30, confidence work).** Record per-sample *cues*, not
> only the edge flag, so Phase 0 also produces the data for the confidence
> map (next section): log-depth gradient magnitude on the output grid, local
> image texture, pixel distance to the nearest good landmark, and (when the
> previous keyframe has depth) the disagreement after warping it with the VO
> pose. The constant `depth_relative_sigma` stays as the fallback, but the
> map replaces it once it passes its calibration test.

## Phase 0b: self-calibrated confidence map (addition)

Networks give no usable per-pixel confidence (DA-V2 none at all), and their
errors are structured, not white. We build our own from the VO's landmarks.
Everything is in **log depth**, so it is scale-free: the arbitrary VO
normalization is a constant offset that the fit absorbs.

**Who owns what.** Scale and shape are owned by different sides:

```
network ──scale (one per segment, slowly varying)──► VO       (Phase 1, Q5)
VO ──────shape accuracy where parallax is good────────► network σ map (this section)
```

Never pull individual landmarks onto their network depth. One scale per
segment, and per-point fusion only as a weighted prior (Phase 5).

**Model.** Per keyframe, after the Phase 1 scale fit, each reference
landmark's residual `r = log z_net − log(s · z_vo)` is a sample of the
network's error with known geometric variance σ_geo² (`CONFIDENCE_DESIGN.md`
Level 3, as `depth_sigma_ratio²`). Predict the network's variance per pixel
from cheap cues:

```
σ_net²(u) = exp( w0 + w1·|∇ log D(u)| + w2·texture(u)
                 + w3·dist_to_landmark(u) + w4·temporal(u) )
```

and fit `w` by maximum likelihood, `r ~ N(0, σ_net²(u) + σ_geo²)`: a few Newton
steps per keyframe, weights carried over as a prior from keyframe to keyframe.
This is the edge test made continuous and self-tuning per sequence (sky,
glass and the pond come out with large σ if they disagree with geometry).
Optional refinement: edge-aware interpolation of the actual residuals near
dense landmarks.

**Circularity rule.** Once Phases 4–5 create or pull landmarks from depth,
those landmarks partly *are* the network. Only landmarks whose
**geometry-only** covariance is small may train the map. The VO therefore
keeps two covariances per landmark (without / with the depth prior) or at
least a `from_depth` flag, and the "≥ 2° parallax, ≥ 3 observations" filter
uses the geometry-only one. Depth-initialized (provisional) landmarks never
qualify until they are re-triangulated.

**Calibration test (gate before any solver use).** Hold out 20 % of the
eligible landmarks per keyframe. Their normalized residuals
`r / sqrt(σ_net² + σ_geo²)` must have std ≈ 1 and ≈ 68 % / 95 % coverage at
1σ / 2σ. On TUM, also compare against the dataset's real depth images.
Show the map next to the video and the reliability plot in the bench.

**First results (2026-09-30, `tools/depth_confidence.py`, DA-V2-S hypersim,
bench exports: dreamworks 0–900, castle 95 s + 1200 frames).** These are
offline measurements only; nothing feeds back into the VO.

- **Scale per keyframe dominates.** With one scale per segment, the
  per-keyframe median offset has std 0.64 (dreamworks) / 0.37 (castle) in log
  depth (p5/p95 up to −1.6 / +0.9), and 22 % / 14 % of residuals are gross
  (factor > 2). Aligning per keyframe cuts gross residuals to 5.3 % / 2.5 %.
  Either the VO's monocular scale drifts within a segment, or the network's
  scale flickers between frames; this data cannot tell which. **Implication
  for Phase 1:** the scale estimator must track at keyframe rate, or the
  depth prior inherits large errors. To separate the two causes, run the
  bench with depth on TUM (real depth images and ground-truth poses).
- **Shape error after per-keyframe alignment:** median |Δ log z| 0.054
  (dreamworks) / 0.144 (castle). Constant model σ_net = 0.20 / 0.23, not the
  0.15 assumed in `depth_relative_sigma`.
- **Cue model (held-out 20 % of landmarks):** on dreamworks it beats the
  constant clearly (NLL −0.150 → −0.316; normalized std 1.04, 79 % / 93 % at
  1σ / 2σ). On castle it ties (−0.098 vs −0.094; constant already calibrated:
  std 0.96, 72 % / 95 %). Strongest cues: log depth (×2.0 per e-fold on
  dreamworks), distance to the nearest reference landmark (sparse regions are
  worse), edge flag (×1.7, dreamworks), and image radius. Per-sequence fitting
  is therefore worthwhile: the same model is right for both clips with
  different weights.
- **Tails are heavier than Gaussian** (79 % within 1σ on dreamworks). Use a
  Student-t (or Gaussian + outlier mixture) for the prior's robust loss.
- Not yet tried: image gradient / texture cues (they need the depth maps, not
  per-track samples) and the TUM ground-truth check.

**Online implementation (2026-09-30).** `DepthConfidenceModel`
(`depth_confidence.{hpp,cpp}`, test `native_depth_confidence`) fits
σ_net² = exp(w·(1, edge, log depth, radius, nearest landmark)) by Fisher
scoring (ridge toward the constant prior σ = 0.2) over a rolling window of
20 000 references. `KeyframeDepthStore` owns one model. At each keyframe the
app passes the frame's observations of reference landmarks
(`KeyframeDepthInput::references`, circularity rule: ≥ 2° parallax, ≥ 3
keyframes, geometry-only σ). Residuals use the same scale-grid alignment as the
clouds and the multi-view check. Then every new cloud point gets
`KeyframeCloud::sigma`. The edge cue is continuous here (3×3 log max/min), and
the radius and nearest-landmark cues are as in the offline tool. Trajectory
view: *Max depth sigma* filters dense points, and colour mode *Confidence*
colours them on the map points' ramp. On fr3_long_office (600 frames) the
window fills after ~65 keyframes, and near surfaces come out confident and
far/peripheral ones less so. Not used for the multi-view tolerance: σ_net
describes errors largely *shared* by neighbouring views (median ~0.15–0.2),
while inter-view disagreement after alignment is 2–4 %. A per-point
tolerance would need a model of the *difference* between views.

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

## Phase 2: dense keyframe clouds in the 3D view (next step)

Back-project each keyframe's depth map, coloured from the frame, into the
trajectory view, so the scene is visible and not just the corner cloud.

**Prerequisites from Phases 0–1, the minimum:** per-track depth samples on
`TrackObservation` (with the sampling policy above) and a per-segment robust
log-scale estimate. Without the scale, a metric cloud cannot be drawn in VO
units. The full Phase 0 study (models, cues) can follow.

**Ownership.** The VO stays depth-map-agnostic: it sees only per-observation
numbers. Dense maps live app/bench side in a `KeyframeDepthStore` keyed by the
keyframe's frame index, filled when the VO reports `keyframe = true` for a
frame whose depth exists (hence "delay the VO by one frame" in live mode). It
holds the downsampled metric depth, colour and, later, Phase 0b's σ map.
Poses come from the VO's trajectory (keyframe samples), which follows BA.

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

## Building block: provisional landmarks (inverse depth)

Phases 4–5 and the rotation fix need the same thing: a landmark whose depth
is **not yet constrained by geometry**, usable for the pose at once and
refined as parallax arrives. Only the source of its initial depth differs.

**State.** Anchored to the keyframe that created it: a unit bearing **b** in
the anchor's camera frame, and inverse depth ρ = 1/z with variance σ_ρ².
`X = C_anchor + R_anchorᵀ · b / ρ`.

- ρ = 0 is a point at infinity, an ordinary value. Sky and horizon points sit
  near 0 with a wide σ_ρ and still constrain rotation, which is all a rotating
  camera needs (castle).
- Triangulation error is close to Gaussian in ρ and nearly depth-independent:
  σ_ρ ≈ σ_px / (f · baseline). So the geometry is well modelled in ρ. That is
  why ρ, not z or log z, is the stored state.
- ρ must stay > 0: clamp in the solver and rely on the prior. A landmark whose
  ρ is driven below 0 by reprojection is an outlier.

**Initial depth, two sources.** Flag which on the landmark (`depth_source`:
`network | scene_median`):

| Source | ρ₀ | σ | When |
|---|---|---|---|
| network depth at the track (Phase 0 sampling policy) | 1 / (z_net / s) in VO units | log-space σ_rel (Phase 0, later σ_net from Phase 0b) | depth present, sample not rejected (edge, clamp, sky) |
| none | 1 / median landmark depth of the keyframe | wide in ρ, covering [0, 1/z_min] | no depth (CSV replay, depth off, estimator busy, rejected sample) |

**Prior residual (in log space, on ρ).** Network error is multiplicative, so
the prior is Gaussian in log depth:
`r = (log z_net + log ρ − s) / σ_rel`, with dr/dρ = 1/ρ, defined for ρ > 0. Here
*s* is the segment's log scale (Phase 1), so scale stays additive. Points
whose sample was rejected get **no** prior: their ρ stays near the scene
median or 0 with its wide σ until geometry fixes it.
*Storage in ρ, comparison in log z:* geometry is Gaussian in ρ, the network is
Gaussian in log z.

**Life cycle.**

1. *Create* for tracks that `triangulate_new` rejects for low parallax (and,
   in Phase 4, for every track at a single-frame initialization). Not for
   tracks that failed the reprojection test: those are outliers, not
   low-parallax.
2. *Use* in pose optimization at once, weighted by the projected uncertainty
   (this composes with the confidence session's `pose_landmark_uncertainty`
   inflation). **Count them toward `min_tracked_points`**, or a rotating
   segment still dies with good tracking.
3. *Refine* in BA (Phase 5) as ρ with the prior term.
4. *Promote* to a normal XYZ landmark when its **geometry-only** covariance
   becomes small (parallax ≥ the triangulation threshold and
   `depth_sigma_ratio` below a bound). Civera's linearity index is the
   textbook test. Clear the provisional flag; the prior may stay as a weak
   term.
5. *Re-anchor* when the anchor keyframe leaves the window (convert to XYZ
   and back at the new anchor, propagating σ_ρ), or *retire* like any
   landmark.

**Constraints agreed with the confidence session.**

- Keep the depth prior **out** of `update_quality`'s information matrix.
  With fewer than 2 views or near-zero parallax, that matrix is singular, so
  `depth_sigma_ratio` = ∞ and `confidence` = 0. That is intended: the depth is
  not geometrically constrained. Display, calibration and the "good landmark"
  filters read this geometry-only covariance.
- Flag provisional / depth-sourced landmarks explicitly on `Landmark` (a
  `provisional` flag plus `depth_source`), so the Phase 0b calibration and
  Phase 0/1 scale fits exclude them until they are re-triangulated (the
  circularity rule).
- In BA the prior is **its own residual type** in `bundle_adjust`, not a fake
  `BundleObservation`.
- No viewer special case: the view-count factor in `MapPoint::confidence`
  already scores 2-view, low-parallax points low.

**Acceptance test.** `castle.mp4` 95–115 s (hfov 71.2, k1 −0.011):
correspondences stay ≥ ~200 through the pitch-up, with no loss, both with
network depth and with the scene-median fallback (depth off). `disney_04` and
`dreamworks.MOV` don't regress (losses, inlier ratio, largest segment).
Synthetic: a pure-rotation segment after initialization keeps tracking.

## Phase 4: single-frame (re)initialization

With depth, a segment can start from **one** frame: back-project the tracked
observations (edge-free, depth-limited) as landmarks at metric scale, set the
pose to identity, and track from the next frame. No parallax wait, and pure
rotation stops being fatal.

- Use it only after a loss with no successful relocalization within K frames
  (relocalization into the old segment is still preferred: it keeps the
  frame), and at the start of a video that has depth.
- The landmarks are *provisional* (building block above, source `network`):
  inverse depth with the log-space prior, promoted once they have parallax.
  The segment's scale is the network's rather than the median-depth
  convention.
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
- Landmarks below the parallax threshold (near the focus of expansion, and
  everything during a rotation: today never created) become provisional
  landmarks (building block above), refined as parallax accumulates. That
  gives more landmarks in exactly the forward-walk and pitch-up cases.
- Keep the prior weak (σ_rel from Phase 0, floored) and robust (Huber in log
  space): network depth has correlated, scene-dependent bias, so it must not
  override geometry where geometry is good.
- *(Addition)* Use the per-pixel σ_net from Phase 0b, not the constant, once
  it passes its calibration test. Apply the circularity rule: depth residuals
  feed the "with prior" landmark covariance only; calibration reads the
  geometry-only one.

## API summary

```cpp
// visual_odometry.hpp
struct TrackObservation { ...; float depth_m{0}; float depth_sigma_m{0}; };
// internal Landmark (visual_odometry.cpp), provisional landmarks:
//   bool provisional; enum DepthSource { triangulated, network, scene_median } depth_source;
//   int anchor_keyframe; Vec3 bearing; double inverse_depth, inverse_depth_sigma;
struct VisualOdometryConfig {
  ...
  bool use_depth{true};                 // master switch; no effect without depth samples
  double depth_relative_sigma{0.15};    // default σ_z / z (Phase 0 sets it)
  double depth_gate_sigmas{3.0};        // occlusion / gate threshold k
  bool depth_initialization{false};     // Phase 4
  bool depth_priors{false};             // Phase 5
  bool provisional_landmarks{false};    // building block; scene-median prior without depth
  double promote_depth_sigma_ratio{0.1};  // geometry-only bound to leave provisional
};
struct OdometryFrameResult {
  ...
  double metric_scale{};                // metres per VO unit for this segment; 0 = unknown
  double metric_scale_sigma{};          // relative
  // ProjectedLandmark gains: enum Visibility { unknown, occluded, visible, behind } visibility;
};
class VisualOdometry { ... double segment_scale(int segment) const; };
// Shared with CONFIDENCE_DESIGN.md: TrackObservation::sigma_px and the tracks
// CSV `sigma` column have landed; the `depth` column is added here.

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

1. Which model and resolution? *For the first implementation:* DA-V2-S
   (518×294) on every frame. The final choice still comes from Phase 0 on
   our clips.
2. Do we want metres in exports and the UI by default, or as an option
   (`--metric`) until the scale estimate has proven stable?
3. Dense view: point sprites in `PointCloudRenderer`, or a voxelized
   accumulated cloud? *Decided:* point sprites per keyframe first; voxels
   for final keyframes later.
4. Should depth go into the feature cache (f16 518×294 ≈ 300 KB/frame), so
   cached replays have it, or only per-track samples in the tracks CSV?
   Per-track samples are enough for Phases 0–1 and 3–5; Phase 2 needs maps at
   keyframes.
5. Depth-based initialization makes the network's scale the segment's scale.
   Should that replace the median-depth convention everywhere, so that every
   segment is metric from the start?
   *Proposed answer (2026-09-30):* **yes, early**. The VO's scale is a
   convention, and the network is the only grounded source, so make it the
   gauge as soon as Phase 0 shows the per-segment scale is stable. Don't
   wait for Phase 4. Anchor to the running robust estimate (Phase 1's log-scale
   filter), never to a single frame's depth: absolute scale is 10–30 % off and
   scene-dependent.
6. *Answered (2026-09-30):* how to store depth for landmarks without parallax?
   Inverse depth as the state; log depth for the network prior and scale (see
   "Provisional landmarks").
7. *Answered (2026-09-30, with the confidence session):* does the depth prior
   enter the landmark covariance? No: the geometry-only covariance stays
   prior-free (confidence 0 for unconstrained depth). Provisional landmarks
   are flagged and excluded from calibration, and the BA prior is its own
   residual term.
8. Keyframe-trajectory jitter (in-between frames anchored only to the
   previous keyframe, so BA corrections to the next keyframe are not
   propagated): known, deferred. Dense clouds are anchored to keyframes, so it
   doesn't affect them.
