# Confidence: design proposal

Status (2026-09-30): **phase 0 implemented** (VO side, diagnostics only;
trajectories bit-identical). See "Phase 0: what landed" below. Phases 1+ are
still proposals.

Goal: give every observation, landmark and pose an uncertainty, and use it
(a) to weight the solvers, (b) to make gating/keyframe/culling decisions, and
(c) to show and filter the map. Today all of these are binary pass/fail gates
with fixed pixel thresholds, and all surviving observations weigh the same.

## Principles

1. **Uncertainty, not a score.** Carry variances / information matrices in
   physical units (pixels², world²). Weights are inverse variances, so
   "weighting by confidence" is ordinary weighted least squares and composes
   with the existing Huber kernel. A 0–1 `confidence` exists only as a derived
   display/filter value at the API edge.
2. **Three levels, propagated in order.** 2D observation noise (tracker) →
   3D landmark covariance (triangulation / BA) → pose covariance (pose
   optimization). Each level is computed from the one below.
3. **Monocular scale is arbitrary, so confidences must be scale-free.** Any
   user-facing number is relative (depth σ / depth, angles, pixels), never
   absolute world units.
4. **Noise vs outliers stay separate.** Variance describes how well a correct
   match is localized. Descriptor similarity and forward–backward flow error are
   mostly *outlier* evidence; they stay as gates (or later, as prior inlier
   probabilities), not as variances.
5. **Default-identical.** Every new field defaults to "unknown", which maps to
   today's behaviour (σ = 1 px, unit weights). Phase 0 must reproduce current
   trajectories bit-for-bit.

## What exists today

| Signal | Where | Reaches the VO? |
|---|---|---|
| Kalman position variance per track (px²) | `DeviceTrack::variance`, `flow_association.cu:50` | No (not even in `TrackRecord`) |
| Descriptor cosine of the match | `TrackRecord::similarity` → `FrameFeatures::landmark_similarities` | No |
| Forward–backward flow residual | `TrackRecord::fb_error` | No |
| \|detection − flow prediction\| | `TrackRecord::correction` → `correction_distances` | No |
| SuperPoint score, track length, misses | `TrackRecord` | No |
| `LandmarkState::confidence`, `FrameFeatures::track_confidences` | `types.hpp` | Vestigial: never set / always cleared |
| Triangulation parallax | `triangulate_new`, `visual_odometry.cpp:894` | Computed as a gate, discarded |
| Per-observation reprojection error | `cull`, `visual_odometry.cpp:947` | Computed as a gate, discarded |
| Point information matrix `Hpp` (3×3) and its inverse | `bundle_adjust`, `vo_geometry.cpp:430–455` | Built every iteration, discarded |
| Pose information `H` (6×6) | `optimize_pose`, `vo_geometry.cpp:350` | Built every iteration, discarded |
| Keyframe observation count, first/last frame | `MapPoint` | Yes (exposed) |

`TrackObservation` (the tracker-agnostic VO input) carries only id, x, y and
colour, so nothing from the tracker's quality model survives the hand-off.

## Level 1: observation uncertainty (tracker → VO)

**API.** Add to `TrackObservation`:

```cpp
float sigma_px{0};  // 1-σ localisation noise per axis; <= 0: unknown (config default)
```

and `VisualOdometryConfig::default_observation_sigma_px{1.0}` and
`min_observation_sigma_px{0.5}` (floor: see Risks). CSV-driven runs and other
trackers simply leave it at 0.

**Source (FlowTracker).** Add `variance` to `TrackRecord` and set
`sigma_px = sqrt(variance)` in `tracked_frame_from`. That is the posterior of
the existing per-track Kalman filter, so it already encodes "freshly detected
(σ = detection_sigma) vs fused over many frames (smaller)". Only supported
(non-coasted) observations reach the VO, so the coast inflation never shows up
there; that is intended.

Open option (phase 1b, measure first): inflate σ for low SuperPoint score.
Do **not** fold similarity / fb_error into σ (Principle 4).

Remove the vestigial `track_confidences` / `LandmarkState::confidence`, or
repurpose them explicitly. They should not be left half-meaning something.

## Level 2: weighted solvers

All three solvers whiten residuals: `r = (project(X) − u) / σ`.

- `PoseObservation` and `BundleObservation` gain `float sigma{1}`.
- `optimize_pose` / `bundle_adjust`: weight = `huber_weight(|r|², δ) / σ²` in
  the normal equations; cost uses `|r|²`. With σ = 1 this is the current code.
- **Thresholds.** Phase 1 keeps inlier/Huber thresholds in *pixels* (only the
  weights change), so gating behaviour is unchanged and effects are
  attributable. Phase 3 switches to χ² gates in whitened units
  (2 DoF, 95 %: |r| < 2.45, comparable to today's 3 px at σ ≈ 1.2).
- `ransac_pnp`: unchanged at first (it counts inliers). Later: PROSAC-style
  sampling ordered by confidence.

## Level 3: landmark uncertainty (3D)

**Covariance.** For a point with cameras held fixed, `Σ_X = Hpp⁻¹` where
`Hpp = Σ_obs Jpᵀ Jp / σ²`. `bundle_adjust` already builds `Hpp` and inverts
it (damped) every iteration. Store the undamped inverse from the final
iteration. For points outside the BA problem (just triangulated, or with one
windowed view), compute the same 3×3 from their observations in
`triangulate_new` / on demand. Cost: negligible.

This ignores camera uncertainty and gauge freedom, so it **underestimates**
(see Risks). It is still correctly *ranked*: low parallax, few views and
large σ all inflate it, and the depth direction dominates as expected.

**Stored per `Landmark`:**

```cpp
Mat3 covariance;            // world², cameras fixed; zero = unknown
double max_parallax_rad;    // over all keyframe observation pairs (updated on new observations)
double reprojection_rms;    // whitened, over windowed observations at the last cull
```

**Exposed on `MapPoint` (scale-free):**

```cpp
float depth_sigma_ratio{NAN};       // σ along the viewing ray / distance, from the
                                    // last keyframe that saw it; NAN = unknown
float max_parallax_degrees{};
float reprojection_rms{NAN};        // σ units
float confidence{NAN};              // 0..1 = 1 / (1 + (depth_sigma_ratio / r0)²), r0 in config
std::array<float, 6> covariance{};  // xx xy xz yy yz zz, for ellipsoid display
```

`depth_sigma_ratio` is the headline number: dimensionless, monocular-safe, and
it captures parallax, view count and pixel noise together. Retired points keep
their values from retirement time (they are final, so their uncertainty is too).

## Level 4: pose uncertainty

`optimize_pose` returns its final 6×6 information `H`. Expose on
`OdometryFrameResult`:

```cpp
float rotation_sigma_deg{NAN};
float translation_sigma_ratio{NAN};  // translation σ / median scene depth (scale-free)
```

Use: a continuous tracking-health signal for loss detection and for the UI,
next to the inlier count.

## Where confidence gets used

In rough order of expected payoff:

1. **Pose tracking weights (biggest win).** Weight each 2D–3D match by the
   *combined* pixel variance `σ_obs² I + J Σ_X Jᵀ` (landmark covariance
   projected into the image). Low-parallax and young landmarks then stop
   pulling the pose as hard as well-triangulated ones. Today they weigh the same.
2. **Bundle adjustment weights:** Level 1 σ (Level 3 is an output of BA, not an input).
3. **Search regions sized by uncertainty.** Re-association
   (`reassociation_radius_px = 2`) and the relocalization guided search use a
   fixed radius. Replace it with the projected 2×2 covariance ellipse
   (Mahalanobis gate, with the px value as cap). This matches the measured
   fit rates (92 % within 2 px, under 20 % beyond 3 px): the right radius depends on
   the landmark.
4. **Keyframe decision:** `keyframe_track_ratio` counts *information*
   (Σ weights of inliers) instead of raw inlier count, so losing a few
   well-constrained points triggers a keyframe sooner than losing many weak ones.
5. **Culling / retirement:** do not retire (freeze) points whose
   `depth_sigma_ratio` exceeds a threshold. Drop them, or keep them flagged
   `uncertain`. This fixes the "smear" of low-parallax points along viewing
   rays in the map.
6. **Triangulation acceptance:** replace the fixed `min_triangulation_parallax`
   gate with `depth_sigma_ratio < max` (parallax is a proxy for it).
7. **Viewer:** colour by confidence (colormap), minimum-confidence slider, and
   an **ellipsoid mode**. The instanced icosahedron renderer takes a per-instance
   3×3 transform `chol(Σ)·k` instead of a uniform scale, so uncertainty is
   drawn directly as an elongated blob along the viewing ray.
8. **Exports:** map CSV gains `depth_sigma_ratio, max_parallax_deg,
   reprojection_rms, confidence` (append columns; the reader tolerates them).

## Phases

| Phase | Content | Behaviour change | Exit criterion |
|---|---|---|---|
| 0 | Plumbing and diagnostics: `sigma_px` field (unused), Level 3 bookkeeping, `MapPoint` fields, pose σ, viewer colour-by-confidence, CSV columns | None | Trajectories bit-identical on test suite + TUM fr1_xyz / fr1_desk / fr3_long_office; confidence visibly separates the low-parallax smear |
| 1 | Whitened residuals with tracker σ in pose opt and BA; px thresholds unchanged | Weights only | ATE/RPE on TUM (ground truth) no worse anywhere and better on average; synthetic tests unchanged |
| 2 | Landmark covariance propagated into pose-tracking weights | Weights only | As above, plus fewer "pose re-estimated (P3P)" events |
| 3 | Covariance-shaped gates: re-association, guided search, χ² inliers, triangulation acceptance, retirement filter | Decisions | Re-association precision ≥ current (dreamworks measurement) at higher recall |
| 4 | Ellipsoid rendering, information-based keyframes, PROSAC | Mixed | Case by case |

Each phase should land separately with its numbers recorded in
`ARCHITECTURE.md`, following the existing measured-default convention.

**Evaluation harness needed:** a small `tools/` script (or bench mode) that runs
`slam-native-vo-tracks` on the TUM sequences and reports Umeyama-aligned
(sim3) ATE and RPE against `groundtruth.txt`. `test_visual_odometry.cpp`
already has the alignment code for synthetic data.

## Risks

- **Overconfidence from the Kalman σ.** Long tracks converge to small
  variance, but their errors are temporally correlated (a feature that slides
  on an edge), not white. Hence the `min_observation_sigma_px` floor
  (proposal: 0.5 px, tune in phase 1).
- **`Hpp⁻¹` ignores camera uncertainty**, so absolute values are optimistic,
  especially for points seen only by fixed or poorly constrained keyframes.
  Mitigation: use it for ranking and relative thresholds; calibrate `r0` and
  gate thresholds empirically rather than treating them as true probabilities.
- **Robust kernel interaction.** Whitening changes which residuals hit the
  Huber knee. Keeping px thresholds in phase 1 isolates this to phase 3.
- **Determinism.** Weighted sums must keep the existing sorted iteration
  orders (pose observations are sorted by id today).
- **Concurrent work.** `visual_odometry.cpp`, `vo_geometry.*`,
  `flow_association.cu` and `types.hpp` are under active change. Phase 0
  touches all of them, so land it in one coordinated change or split by file
  owner: tracker side (Level 1 plumbing) vs VO side (Levels 2–4).

## Phase 1: what landed (2026-09-30)

- **Tracker σ reaches the VO.** `TrackRecord::variance` (Kalman posterior)
  → `FrameFeatures::track_sigmas_px` (replaces the vestigial
  `track_confidences`) → `TrackObservation::sigma_px` in `tracked_frame_from`
  → optional per-row `sigma` column in the tracks CSV (written by the bench).
- **Weighted solvers.** `PoseObservation` / `BundleObservation` carry σ:
  weight `huber_weight(e²) / σ²`, Huber knee and inlier tests still in pixels.
  Inside the VO each keyframe stores the resolved σ per track, and landmark
  observations are `Sighting{keyframe, u, sigma}` (σ travels with the
  observation through re-association). `update_quality` uses per-observation σ.
- **Switch:** `VisualOdometryConfig::use_observation_sigma` (σ floored at
  `min_observation_sigma_px` = 0.5). Off = uniform `observation_sigma_px` =
  1, which is bit-identical to the unweighted solver (verified on the TUM
  replays). `slam-native-vo-tracks --set use_observation_sigma=1`.
- **Evaluation harness:** `tools/tum_eval.py` (sim3-aligned ATE per segment,
  1 s RPE vs `groundtruth.txt`) and `tools/tum_suite.py` (all 5 TUM sequences,
  each from start offsets 0/150/300, mean ± spread; tracks cached in
  `out/tum/`). Monocular VO is chaotic: a uniform σ of 1.001 (same problem,
  different rounding) leaves 4 sequences unchanged, but moves fr1_room's ATE
  from 36 to 64 cm in a single run. Hence several offsets, and only trust
  differences larger than the spread.

**Tracker σ distribution** (fr3_long_office): p5/p25/p50/p75/p95 =
0.51/0.51/0.51/0.55/0.74 px, 4.5 % at 1.0 (fresh detections). The filter
settles at ≈ 0.51 px (detection σ 1, flow σ 0.3), so in practice the weights
separate fresh (1 px) and young (0.55–0.75 px) tracks from settled ones.

**Results** (true intrinsics, mean over offsets; RPE cm per 1 s):

| sequence | RPE base → σ | ATE cm base → σ | ATE spread base → σ |
|---|---|---|---|
| fr1_desk | 2.71 → 2.65 | 2.42 → 2.42 | 0.62 → 0.47 |
| fr1_room (chaotic) | 14.45 → 13.98 | 24.7 → 26.2 | 26.9 → 6.3 |
| fr1_xyz | 3.55 → 3.18 | 4.88 → 3.94 | 6.01 → 6.15 |
| fr2_desk | 1.82 → 1.69 | 14.45 → 12.34 | 6.05 → 2.25 |
| fr3_long_office | 1.42 → **1.24** | 5.60 → 5.19 | 2.68 → 0.84 |
| **mean** | **4.79 → 4.55** | **10.41 → 10.02** | |

RPE improves on all five, and the spread over start offsets shrinks 2–4×
(the weighted VO is more repeatable). A floor of 0.7 px is *worse* than both
(mean RPE 5.45, ATE 12.3): it erases the settled vs young distinction. A floor
of 0.3 equals 0.5 (no σ is below 0.51).

`use_observation_sigma` is now **on by default** (synthetic tests carry no σ
and are unaffected).

## Phase 2 attempt: landmark covariance in pose tracking (off)

`pose_landmark_uncertainty` adds each landmark's covariance, projected
through the predicted pose, to its observation's σ (isotropic:
`σ² = σ_obs² + tr(J Σ_X Jᵀ)/2`), optionally capped (`pose_landmark_max_inflation`).

| variant (with observation σ) | mean RPE | mean ATE | fr1_xyz ATE | fr1_room RPE |
|---|---|---|---|---|
| observation σ only (default) | **4.55** | **10.02** | 3.94 ± 6.15 | 13.98 |
| + landmark covariance | 5.33 | 12.25 | 1.80 ± 0.29 | 18.88 |
| + covariance, cap 2× | 4.95 | 11.07 | 2.63 | 16.31 |
| + covariance, cap 4× | 5.23 | 11.92 | 1.87 | 18.46 |

A large, consistent win on fr1_xyz, but worse on fr1_room (fast motion) and
fr2_desk, so it stays **off**. Likely causes: fresh landmarks get almost no
weight, so in fast motion the pose rests on a few old, possibly stale ones;
and the isotropic approximation inflates both image axes, while depth error
projects along one direction (the epipolar line). Next try: the full 2×2
projected covariance as an anisotropic weight in `optimize_pose` (needs a 2×2
information per `PoseObservation`).

### Phase 2, anisotropic (2026-09-30)

`PoseObservation::landmark_covariance` (px²) carries `J Σ_X Jᵀ`.
`optimize_pose` measures the residual as a Mahalanobis distance in σ-pixels,
`d² = eᵀ M e` with `M = σ²(σ²I + C)⁻¹`. `d` replaces |e| in the IRLS weight
and the Huber knee, so a landmark stops pulling along its uncertain direction
(the epipolar line) but still constrains the pose across it. With
`pose_landmark_pixel_gate` (default), inliers are still classified by plain
pixel error. With a zero covariance the code takes the old path exactly.

| variant | fr1_desk | fr1_room | fr1_xyz | fr2_desk | fr3_long | mean ATE | mean RPE | RPE° |
|---|---|---|---|---|---|---|---|---|
| default (obs σ) | 2.42 | 26.2 | 3.94 ± 6.2 | 12.3 | **5.19 ± 0.84** | **10.02** | **4.55** | 1.60 |
| isotropic | 2.50 | 37.2 | 1.80 | 14.7 | 5.09 | 12.25 | 5.33 | 1.54 |
| anisotropic, Mahalanobis gate | 3.39 | 34.6 | 2.08 | **10.6** | 6.58 | 11.45 | 4.89 | **1.46** |
| **anisotropic, pixel gate** | **2.30** | 31.2 ± 24 | **1.88 ± 0.26** | 13.6 | 6.31 ± 0.98 | 11.07 | 4.75 | 1.53 |

(ATE cm, mean over start offsets 0/150/300.)

- The Mahalanobis gate is worse than the pixel gate. Covariance describes
  precision, not correctness: a wrongly matched young landmark is uncertain
  along its epipolar line and slides there without being rejected.
- The pixel-gated version beats the default on fr1_desk and fr1_xyz
  (3.94 ± 6.15 → 1.88 ± 0.26: much more repeatable) and on rotation drift,
  is noise on fr1_room (chaotic) and fr2_desk, and is **consistently worse on
  fr3_long_office** (ATE 5.19 → 6.31, RPE 1.24 → 1.36, tight spreads). The
  mean is still worse than the default, so it stays **off**.
- Likely reason, and the next fix: `Σ_X` is computed with the keyframes held
  fixed. A long-lived landmark anchored in old keyframes looks precise, but
  relative to the *current* camera it carries those keyframes' drift. Weighting
  by `Σ_X` therefore favours old landmarks over fresh ones, the opposite of
  what accumulated drift calls for (fr3 is a long, slow loop). Candidates:
  (a) marginal covariance including the window's camera uncertainty (Schur
  complement from BA); (b) an age term growing with keyframes since the
  landmark left the window; (c) the stale-observation fix, so old landmarks
  are re-checked against the current window.

### Stale observations and Phase 2 retest (2026-09-30)

**Diagnostic fix (no behaviour change).** Each `Sighting` stores its
whitened reprojection error as last measured while its keyframe was in the
BA window, and `reprojection_rms` uses those values. Before, every observation
was measured against the *current* point, so old keyframes (which drift relative
to the window) made RMS grow with age: median 0.56 → 3.9σ from 2 to 12+ views
on fr3. It now stays flat: 0.88 (2 views), 1.24–1.67 (3–8+), p90 2.2–2.8.
The flat level of ~1.6 (expected ~0.8 for a calibrated σ) says the tracker's
σ is optimistic by ~1.7× on TUM (the 0.5 px floor).

**`cull_stale_observations`** (default off) also re-checks observations
outside the window and drops those that no longer reproject (2× threshold).
On its own it changes **nothing** in the trajectories (suite byte-identical):
pose tracking and BA only ever use windowed observations. But it does change
the landmark **covariance**, which `update_quality` computes from all
observations, so stale ones made old landmarks look over-precise. That
matters for Phase 2:

| variant | fr1_desk | fr1_room | fr1_xyz | fr2_desk | fr3_long | mean ATE | mean RPE | RPE° |
|---|---|---|---|---|---|---|---|---|
| default | 2.42 | 26.2 | 3.94 ± 6.15 | 12.3 | **5.19 ± 0.84** | **10.02** | 4.55 | 1.60 |
| anisotropic | 2.30 | 31.2 | 1.88 | 13.6 | 6.31 | 11.07 | 4.75 | 1.53 |
| **anisotropic + stale cull** | **2.29** | 27.4 | **1.89 ± 0.26** | 12.8 | 6.04 ± 1.74 | 10.09 | **4.42** | **1.48** |

Stale culling recovers most of Phase 2's loss: best mean RPE and rotation
drift, ATE equal to the default, fr1_xyz much more accurate and repeatable.
fr3 is still worse (RPE 1.24 → 1.39), so both stay **off** by default for
now. The remaining suspect is the same one at a smaller scale: the
covariance still holds the keyframes fixed. Next: marginal covariance with
camera uncertainty from BA (Schur complement), or calibrate the covariance
scale (actual / predicted error ≈ 1.37 on synthetic; σ ≈ 1.7× optimistic on TUM).

### Consistency term (2026-09-30)

`MapPoint::confidence` = precision × verification × **consistency**. For
landmarks with 3+ views, q = windowed RMS / the map's median windowed RMS
(over active 3+-view landmarks, refreshed at each cull: self-calibrating for
a mis-scaled σ), and consistency = `1 / (1 + (max(0, q − 1) / 0.75)²)`
(`confidence_consistency_scale`). Synthetic default scene: wrong (mover)
landmarks with 3+ views have windowed RMS median 1.04 vs 0.56 for static
ones; their median confidence relative to static points goes 0.91×
(precision) → 0.76× (+ verification) → **0.38×** (+ consistency). Static
points are barely affected (median 0.367 → 0.359).

## Verification: view count in the confidence (2026-09-30)

Points triangulated from exactly two keyframes have a median whitened
reprojection error of 0.56 (vs 1.4–3.9 with more views): two rays always
fit, so a wrong match is invisible. They are 26–44 % of the TUM maps, and 36–60 % of them
had confidence > 0.5 from precision alone. `MapPoint::confidence` is now
**precision × verification**, with verification `1 − (1 − 0.25)·0.5^(n−2)`
for n keyframe observations (n=2: 0.25, 3: 0.63, 4: 0.81;
`confidence_two_view`).

Evidence (synthetic walk, test `confidence_flags_unverified_points`): the 80
independently moving points produce wrong landmarks. They are 5.7 % of
2-view landmarks vs 1.0 % of ≥4-view ones (they die young). Their median
confidence relative to static points drops from 0.91× (precision only) to
0.76× (with verification). That is a real but moderate effect: 40 % of
wrong landmarks survive to 3+ views. Caveat: by position error alone, view
count is *confounded* in forward motion. The longest-seen static points sit
near the focus of expansion and have the least parallax (static gross-error
rate rises from 5 % at 2 views to 20–35 % at 8+ views), which the precision
term covers. Still open: a consistency term (reprojection RMS over ≥3 views)
should catch wrong landmarks that survive. It needs the stale-observation fix
first, because today the RMS grows with views for every point.

## Relation to `DEPTH_INTEGRATION.md`

- **Shared input change.** `TrackObservation` gains `sigma_px` (here) and
  `depth_m` / `depth_sigma_m` (depth doc). Land them as one change with one owner,
  together with the optional tracks-CSV `sigma` / `depth` columns.
- **Scale comes from the network, shape from geometry.** The depth doc's
  Phase 1 / Q5 makes the network the segment's scale gauge. This doc's
  covariances are scale-free (`depth_sigma_ratio`, log depth), so they are
  unaffected by who owns the scale.
- **Self-calibrated depth confidence map** (depth doc, Phase 0b) is trained
  on Level 3 landmark covariances. That requires Level 3 to keep a
  **geometry-only** covariance per landmark, separate from any later
  "with depth prior" covariance (circularity rule).
- A depth prior in BA (depth doc Phase 5) is one more information term in
  `Hpp`. It needs Levels 2–3 here first.

## Phase 0: what landed (2026-09-30)

- `vo_geometry`: `point_information` (3×3, cameras fixed) and
  `pose_information` (6×6). New functions only; the solvers are untouched.
- `TrackObservation::sigma_px` (unused until phase 1; the flow tracker does
  not fill it yet: it needs `variance` in `TrackRecord`, `flow_association.cu`).
- `VisualOdometryConfig::observation_sigma_px` (1 px) and
  `confidence_depth_ratio` (0.05).
- `Landmark` quality refreshed in `cull` (after every keyframe BA, and
  therefore frozen at dormancy/retirement): geometry-only covariance,
  `depth_sigma_ratio` (∞ when depth is unconstrained), max parallax,
  whitened reprojection RMS. Exposed on `MapPoint` (with `confidence` and the
  covariance) and as map-CSV columns `depth_sigma_ratio, max_parallax_deg,
  reprojection_rms, confidence`.
- `OdometryFrameResult::rotation_sigma_degrees`, `translation_sigma_ratio`
  on every tracked frame.
- Trajectory view: colour mode *Plain / Image colour / Confidence*
  (red–yellow–green, grey = unknown), *Min confidence* filter, counter split
  into live / held (dormant, lost segment) / retired.
- Test `reports_uncertainty` (synthetic walk, 0.7 px noise): every retired
  point has an uncertainty; points with confidence > 0.9 have a 3.4 % median
  relative position error vs 7.4 % below 0.5; the actual error over the
  predicted σ has median 1.37, so σ is about right and slightly
  optimistic, as expected from ignoring camera uncertainty.

**Regression check.** `slam-native-vo-tracks` on bench-exported tracks of TUM
fr1_desk and fr3_long_office (600 frames each): trajectory CSVs
byte-identical, map CSV identical in all pre-existing columns.

**Measurements on those runs** (≈ 9.3k and 6.0k map points):

| max parallax | median depth σ / depth |
|---|---|
| < 1° | 0.16–0.18 |
| 1–2° | 0.08–0.09 |
| 2–5° | 0.04 |
| ≥ 5° | 0.009–0.010 |

That is about 1/parallax, as theory predicts. 77–79 % of points have
confidence > 0.5. In the 3D view, stray points far off the structure come out
red. **Finding:** the whitened reprojection RMS has p90 = 4–6σ, because
observations from keyframes that left the BA window are never re-checked
against the moved point. Re-culling them (or excluding them from the RMS) is
a phase 1 candidate.

**Finding (live app):** with re-association on, landmarks whose tracks end
go *dormant*, not retired, so the live view shows e.g. 1214 live / 1966 held /
0 retired after 400 frames of fr3_long_office. Dormant points are drawn (they
are in `active_map()`).

## Open questions

1. Is a scalar σ per observation enough, or do we want an anisotropic 2×2
   (e.g. from flow along edges)? Scalar is proposed for now; the solver changes
   are the same either way.
2. Should low-confidence points be *dropped* at retirement, or kept and
   flagged so the viewer/exports decide? Proposal: keep and flag; filter in
   consumers.
3. Do we want pose covariance in the trajectory CSV (for downstream fusion),
   or only in the UI?
