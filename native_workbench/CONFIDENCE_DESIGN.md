# Confidence: design proposal

Status: **draft for review** (2026-09-30). No code changes yet.

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

## Side note: the view's "active" count is wrong

`active_map()` returns live landmarks **plus dormant and lost-segment ones**
(flagged `MapPoint::dormant`). The trajectory view draws all of them as
"active", but its "Map points … active" counter uses
`OdometryFrameResult::map_points` (= live landmarks only). The counter should
use the drawn set and split live / dormant / retired. Dormant points are a
natural fourth display class (dimmer, or hidden) alongside confidence.

## Open questions

1. Is a scalar σ per observation enough, or do we want an anisotropic 2×2
   (e.g. from flow along edges)? Scalar is proposed for now; the solver changes
   are the same either way.
2. Should low-confidence points be *dropped* at retirement, or kept and
   flagged so the viewer/exports decide? Proposal: keep and flag; filter in
   consumers.
3. Do we want pose covariance in the trajectory CSV (for downstream fusion),
   or only in the UI?
