# Depth fusion and depth-aided tracking: design proposal

Status: **draft, agreed in discussion 2026-09-30, to implement next session.**
Builds on `DEPTH_INTEGRATION.md` (network depth, scale grid, keyframe clouds),
`CONFIDENCE_DESIGN.md` (uncertainty everywhere) and the depth confidence model
(`depth_confidence.hpp`).

## Why

Per-frame network depth is inconsistent between frames, and per-keyframe
clouds disagree. Measured: after scale-grid alignment, neighbouring keyframes
differ by 2–4 % (disney_04). Network vs geometry is off by 15–20 % (log), and
most of that error is **shared** by neighbouring views. Against TUM ground
truth, 28 % of shown dense points are more than 10 % off after a per-keyframe
scale, and the multi-view check hides only 6 % of the bad ones.

So:

- **Temporal fusion** (carry depth forward, update with each new map) fixes the
  *flicker*: it removes the 2–4 % independent part.
- **Accuracy** needs a geometric measurement: averaging the same biased network
  answer does not fix it. Dense optical flow + pose gives one at almost every
  pixel.
- With a consistent dense depth + σ, the next frame's **tracking** can use it:
  predicted track positions, and direct (photometric) alignment.

## 1. Depth filter: per-pixel Kalman in log depth

**State** per pixel of the depth-map grid (518×294 for DA-V2-S): log depth
(VO units) and σ (log). VO units, so the warp uses VO poses directly. The network is
converted with the keyframe scale grid on the way in. Plus an age/observation count.

**Predict (push the state to the next frame):**

- Forward-warp with the VO pose (relative motion k → k+1), by splatting with a
  z-buffer: nearest surface wins, disocclusions become empty pixels.
- σ² += process noise: the pose uncertainty projected per pixel
  (`OdometryFrameResult` pose σ), plus a drift term per frame, plus a resampling
  term near depth edges.
- Lens: undistort/distort as the VO does (k1).

**Update (the new network map is just an observation):**

- Measurement: network log depth − scale-grid L(pixel), σ from
  `DepthConfidenceModel` (cues: edge, depth, radius, landmark density).
- Standard scalar Kalman update in log depth. Empty (disoccluded) pixels take the
  measurement as is, so "the network fills occlusions" falls out as the limit.
- **Reset test:** if |innovation| > k·sqrt(σ_pred² + σ_meas²) for several
  frames (moving object, wrong prior), reinitialise the pixel from the
  measurement. Count resets per frame; a spike means a pose failure.

**Further observations (same update, more measurements):**

- **Flow triangulation:** NVOF gives dense flow k → k+1. With the relative pose,
  every pixel with parallax yields a triangulated depth, with σ from parallax and
  flow noise (as `depth_sigma_ratio` for landmarks). This is strong where
  parallax is good, and weak near the focus of expansion and under pure rotation
  (the network carries those pixels). This is the measurement that improves
  *accuracy*.
- **Landmarks:** sparse anchors through the scale grid (already in place).

**Hooks:**

- Rescale when a keyframe's grid settles or a loop closure changes scale.
- Clear on tracking loss / new segment, and on seeks.
- Keyframe clouds become snapshots of the fused map (consistent by
  construction). The multi-view check may become redundant; measure it.

**Implementation:** CUDA kernels next to `DepthEstimator` (warp/splat,
update), state double-buffered on the GPU, host copy only for clouds and display.
At 518×294 this is well under 1 ms.

**Measure** (tools exist: `dense_vs_gt.py`, `depth_consistency.py`, TUM suite):

- Error vs TUM ground-truth depth: raw network vs fused (expect little change
  until flow triangulation is added).
- Temporal flicker: per-pixel |Δ log z| of the same surface frame to frame
  (expect a large drop).
- Neighbouring-cloud disagreement (expect a drop).
- Calibration of the fused σ (normalised error ≈ 1).

## 2. Depth-aided tracking

The fused depth (+σ) for frame k, the image of frame k and a predicted pose
for k+1 give an expected position for every pixel in k+1.

**2a. Tracker prediction (cheap, do first).** Predict each live track's
position in k+1 from depth + predicted pose, as a second prediction next to
the optical-flow one in the GPU tracker. Uses: a tighter association gate,
re-acquisition after coasting, and fast rotation (where flow struggles). This is
"Feeding the pose back to the tracker" in `ARCHITECTURE.md`. Measure: track
continuation and coasting (bench), re-association rate.

**2b. Direct pose initialisation (semi-direct).** Photometric alignment of
k+1 to k over high-gradient pixels with converged depth (σ-weighted), on a
coarse-to-fine pyramid, starting from the constant-velocity prediction.
Feature-based `optimize_pose` then starts from it. Start *sparse*: patches
around landmarks (known depth, SVO-style), then add dense pixels if it pays.
Needs a per-frame affine brightness fit (phone auto-exposure) and robust
weights (blur, rolling shutter). Measure: "pose re-estimated (P3P)" events,
losses, TUM suite ATE/RPE.

**2c. Joint cost (only if 2b pays off).** Feature reprojection + photometric
terms in one pose optimisation.

**Guard against the feedback loop.** Depth comes from poses, and the pose would
then come from that depth. Feature reprojection stays the anchor: the direct
result is an initialisation or prior, accepted only if the feature inliers agree
(inlier count not worse, reprojection RMS within bounds). Photometric terms
only use pixels whose fused σ has converged, never freshly network-filled ones.

## Order

1. Filter with the network observation only (+ σ, reset test, hooks). Measure.
2. Tracker prediction from depth + pose (2a). Measure.
3. Flow-triangulation observation. Measure accuracy vs GT.
4. Sparse direct pose initialisation (2b). Measure with the TUM suite.
5. Dense / joint direct (2c), if 4 pays off.

## Risks

- Too little process noise locks errors in; too much reproduces the network.
  Tune on TUM (GT depth + poses).
- Pure rotation: no parallax, no new geometry (the filter holds; the network carries).
- Moving objects: the reset test; they also corrupt the photometric terms (robust weights).
- Photometric assumptions break on exposure changes, blur and rolling shutter.
- Ownership: the depth estimator (session slam-lab-5d), keyframe clouds and
  the confidence models (slam-lab-de), the VO and loop-closure rescaling
  (slam-lab-86). Coordinate before starting.
