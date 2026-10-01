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

**Built (2026-09-30), host version:** `depth_filter.hpp` / `src/depth_filter.cpp`,
`tests/test_depth_filter.cpp`, bench `fuse-depth 1` (with `dense 1` the keyframe
clouds are snapshots of the fused map). Changes from the plan below, all from
the synthetic test:

- **Two states per pixel, z and the network bias b** (2×2 covariance). The
  network measures z + b. A scalar filter treats the shared 15–20 % error as
  independent and becomes overconfident: on the synthetic scene (bias 0.15
  fixed to the surface, 3 % independent noise) its normalised error RMS is 4.9.
  The 2-state filter scores 0.97. b is an Ornstein–Uhlenbeck process
  (`bias_sigma`, `bias_correlation_frames`). Measurement noise is the
  independent part (`independent_fraction` of the network σ), and the rest is
  b's prior.
- **b is warped with the same Jacobian as z** (d log z′ / d log z = 1 − t_z/z′).
  If b is held fixed in log depth while z scales, the depth ratios between
  frames triangulate absolute depth from network maps alone, and z's gain stays
  near 0.5 (flicker drops only 45 %). Real networks don't support that; only a
  geometric measurement (step 3) should separate z and b.
- **Resampling noise floor:** the gradient × resampling-distance term uses
  (|∇ log z| − 0.03)₊. Otherwise the state's own pixel noise reads as slope and
  feeds back as process noise.
- **Splat:** to the 4 nearest pixels (closes the cracks of a magnifying warp).
  Nearest surface wins; among candidates within 2 % of it, the closest
  sub-pixel position wins (order-independent, so it ports to CUDA).
- **CPU first:** about 80 ms/frame in the debug build, fine for the bench. The
  CUDA port comes after the numbers.

**Result: per-frame warping hurts accuracy (2026-09-30).** Flicker drops 5–8×,
but against TUM ground-truth depth (`dense_vs_gt.py`, median error of shown
points / share more than 10 % off), fused is worse than raw everywhere:

| Sequence | Raw | Fused |
|---|---|---|
| fr1_xyz | 0.034 / 15 % | 0.041 / 20 % |
| fr1_desk | 0.045 / 22 % | 0.065 / 32 % |
| fr1_room | 0.057 / 29 % | 0.069 / 36 % |
| fr2_desk | 0.049 / 26 % | 0.114 / 55 % |
| fr3_long_office | 0.054 / 28 % | 0.098 / 49 % |

- It isn't the VO: with TUM ground-truth poses (`fuse-gt SEQDIR`), fr1_desk
  is still 0.076 vs 0.046.
- It isn't the scale grid (`fuse-grid 0`: same) or the bias model (a
  near-scalar filter: same).
- Shorter memory helps (`fuse-drift 0.03`: 0.055 with ground-truth poses).

The cause is resampling the whole state every frame. A pixel snaps to the
nearest centre after each warp and is back-projected from there next frame, so
the surface random-walks sideways (up to ~0.7 px per frame). The 4-neighbour,
nearest-wins splat also grows the foreground. The slow, long sequences (fr2,
fr3), where under 1 % of pixels are ever re-initialised, suffer most.

**Keyframe-anchored version (built 2026-09-30, the current code).** The state
stays in the anchor keyframe's pixels. Each frame samples the network map at
the anchor pixels' projections (z-buffer visibility, bilinear except across
edges) with an EKF row H = [J, 1]. The state is splatted once per keyframe
(a nearer surface wins only the pixels whose centres it covers). Exact on the
synthetic tests, with and without rotation (1e-7 / 1e-4). Further findings on
fr1_desk:

- **Geometry is right.** With TUM ground-truth depth as the measurement and
  ground-truth poses (`fuse-truth-depth DIR fuse-gt SEQ`), the fused map is
  within 0.012 of the truth (3 % of points over 10 %).
- **VO units drift.** The keyframe's log metres per unit falls from 0.53 to
  0.24 within 50 frames. A state in VO units goes stale against the poses,
  and the bias absorbs it. The state is now in network metres; translations
  are converted with a smoothed scale, and the grid only corrects the shape.
- **Network maps alone still don't improve accuracy.** Median error is 0.065
  with per-frame scale alignment (`fuse-align`) and 0.074 without, against
  0.046 raw. With ground-truth poses, the earlier version scored 0.076.
- **The network's error is mostly shared between consecutive keyframes**
  (cell correlation 0.78 in image coordinates), so averaging maps can't
  improve accuracy by much. Part of the error is fixed in the image rather
  than tied to the surface: the top rows read 7–14 % too near and the lower
  centre 3 % too far (`position_bias` on 224 keyframes). Fusing carries
  values measured at one image position to another, which the
  surface-attached bias model doesn't represent.

**Correctness audit (2026-10-01, after trails and smearing in the app).** On
fr1_desk with dumps of the fused state at each keyframe (`fuse-dump DIR`)
next to the raw map. Four defects, all fixed:

1. **Resampling** at re-anchoring and render took the nearest source's depth,
   whose true position can be up to ~0.7 px from the target pixel. On slanted
   surfaces that's a per-pixel error at every keyframe (speckle, compounding).
   Now the depth is corrected along the local plane through the neighbours'
   projections. Synthetic, turning: 3e-4 → 3e-7.
2. **Gaps** under magnification let hidden surfaces show through. The splat
   footprint now grows with the local spacing.
3. **Measurement noise was modelled far too low.** After transport, network
   maps differ by 0.054 median frame to frame; the model said 0.02–0.04. So
   14 % of updates were gated, with hundreds of resets per frame: trails and
   speckle. Now it's floored at each frame's measured innovation spread (MAD;
   `adaptive_noise`). Gated 14 % → 3 %.
4. **Per-pixel z/b split.** Network maps can't observe b, so each pixel split
   innovations by its own history, and z came out 5× rougher than the raw
   map. b's mean is no longer estimated (`estimate_bias` off): the mean is a
   scalar filter's, and the covariance still carries b, so σ stays honest.

Also: the anchor pose is refreshed from the VO's bundle-adjusted trajectory
each frame (`set_anchor_pose`), and per-frame alignment is now off by default
(with the fixes above it cost accuracy).

| fr1_desk, 300 frames | Median error | > 10 % off | Roughness | Flicker vs raw |
|---|---|---|---|---|
| raw network | 0.050 | 27.8 % | 0.0038 | 1 |
| fused before the audit | 0.077 | 39 % | 0.043 | 0.13 |
| fused after | 0.053 | 28.4 % | 0.0030 | 0.41 |

The earlier large flicker reductions were partly the bug: the state was
frozen. Ground truth through the filter: ground-truth depth with ground-truth
poses gives 0.011, and with VO poses 0.035. The remaining gap from VO poses is
real pose and scale error on a fast sequence.

**Full TUM after the audit** (whole sequences, default VO; median error of
shown points / share more than 10 % off; flicker = fused / raw):

| Sequence | Raw | Fused | Flicker | Fused, before the audit |
|---|---|---|---|---|
| fr1_desk | 0.047 / 24.7 % | 0.050 / 25.3 % | 0.44 | 0.065 / 32 % |
| fr1_room | 0.057 / 28.4 % | 0.062 / 31.3 % | 0.46 | 0.069 / 36 % |
| fr1_xyz | 0.033 / 15.1 % | 0.036 / 15.8 % | 0.55 | 0.041 / 20 % |
| fr2_desk | 0.049 / 25.6 % | 0.049 / 26.0 % | 0.59 | 0.114 / 55 % |
| fr3_long_office | 0.054 / 28.3 % | 0.055 / 28.8 % | 0.51 | 0.098 / 49 % |

Accuracy is now at parity with raw (within 0.001–0.005), with half the
flicker. As expected, network maps alone can't buy accuracy; that needs a
geometric measurement (step 3).

**Origin of the anchored design (done, see above):** anchor the state to the keyframe, as LSD-SLAM does (Engel et al.
2014; semi-dense VO, ICCV 2013). Each new map is read by projecting keyframe
pixels into the frame and sampling the measurement there (bilinear). The state
is splatted into a new keyframe only at keyframe creation. Duplicates are
fused when consistent, nearest otherwise. Add inverse-variance neighbour
regularisation. CNN-SLAM (Tateno et al. 2017) is the closest prior work, with
network depth per keyframe refined by small-baseline stereo.

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
