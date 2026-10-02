# Every-frame spline bundle adjustment: design and milestones

Status 2026-10-02: proposal. Nothing here is implemented. The numeric gates
are first proposals and are meant to be argued with before milestone 0 starts.

Keep every frame's landmark observations, model the camera trajectory as a
spline whose control points are the unknowns, and refine the map with bundle
adjustment over randomly sampled connected clusters. The aim is a less noisy
sparse map with no stale per-frame poses. The front end (SuperPoint, flow
tracker, descriptors, matching, relocalization) is reused unchanged, and the
stochastic solver comes last, after a deterministic spline solver has been
shown correct.

Read [ARCHITECTURE.md](ARCHITECTURE.md) ("Start here") first for the current
pipeline and its contracts, and [../baselines/README.md](../baselines/README.md)
for how versions are compared.

## The idea

Today a landmark is estimated from a handful of keyframe sightings, and
everything a non-keyframe saw is thrown away after that frame's pose is
solved. The proposal keeps those measurements and shrinks the problem a
different way: fewer trajectory unknowns instead of fewer measurements.

| | Today (v5) | Proposed |
| --- | --- | --- |
| Observations kept per landmark | Keyframe sightings only; at most 8 enter a solve | Every frame that observed it |
| Trajectory unknowns | One pose per keyframe in an 8-keyframe window | Spline control points, roughly one per 2 to 4 frames |
| Pose of an ordinary frame | Solved once against the map as it was, stored relative to its keyframe, never updated | The spline evaluated at the frame's timestamp, so it is always current |
| Tracks that become landmarks | Only tracks that span two keyframes | Any track with enough parallax |
| Refinement of old parts of the map | None after they leave the window, except a rigid move at a loop correction | A background solver that keeps revisiting clusters |
| Global consistency | Sim(3) pose graph over keyframes | The same pose graph, applied to control points |

A knot is not a keyframe under another name. Keyframing discards measurements
to make the problem small. Knots keep the measurements and assume instead
that motion is smooth at the knot spacing, which is an assumption that can be
wrong and has to be tested (milestone 3).

Keyframes stay on the map side. Retrieval descriptors, loop edges and
covisibility are still indexed by particular images.

## Why it should help

More points make poses more accurate, but they do nothing for the accuracy of
any one point. Only more observations of that point do. This is why the map
is the thing that should improve.

A simple model shows the size of the effect: a camera translating sideways
over a span B, observing one point with independent pixel noise. The
information on the point's inverse depth is proportional to the spread of the
camera positions, `sum_i (x_i - mean(x))^2`.

| Observations used | Information | Depth error against two end frames |
| --- | --- | --- |
| 2 frames at the ends of the span | B²/2 | 1× |
| 30 frames spread evenly | 30·B²/12 | about 2.2× lower |
| 30 frames, half at each end | 30·B²/4 | about 3.9× lower |

Frames near the ends of a track are worth three times as much as frames in
the middle, which is what the sampling in milestone 8 should exploit.

These numbers are an upper bound. They assume independent noise, and only
part of the error is independent:

| Averages down with more frames | Does not average down |
| --- | --- |
| Sensor noise | Pose error, because neighbouring frames' poses come from the same landmarks |
| Detector jitter, including SuperPoint's 1.875 px grid | Intrinsics and distortion error |
| Compression artifacts, in part | Rolling shutter |

The repo already has one measurement of the gap between theory and practice.
The live-depth filter fuses a measurement every frame against the frame ten
back. On the same pixels, error fell from 3.4–4.0% for one measurement to
2.5–2.7% fused ([LIVE_DEPTH.md](LIVE_DEPTH.md)). That is about 1.4×, with
shared poses named as the limit.

The usual argument for keyframes does not contradict this. Strasdat, Montiel
and Davison ("Visual SLAM: Why Filter?", 2012) scored only the final camera
position, in simulation, with perfect data association. They did not measure
map accuracy.

## How the parts fit

```text
Decode and detect ──► Track and associate ──► Observation store (new)
(NVDEC, SuperPoint,    (flow tracker, radius     every associated observation:
 NVOF flow)             and descriptor gates)    landmark id, time, pixel
                              ▲                          │ observations
                              │ predicted pose           ▼
Place search and loops ──► Trajectory spline (new) ◄── Solver (new)
(landmark descriptors,      split rotation +     control   control points and landmarks;
 P3P RANSAC, Sim(3)         translation; pose    points    newest stretch every frame,
 pose graph)                of any frame = T(t)            sampled clusters in background
        loop corrections                                   │ positions
                                                           ▼
                                                     Landmark map
```

Read it as one loop. Tracks become stored observations, the solver turns them
into control points and landmark positions, and the spline's predicted pose
feeds the next frame's association. Place search re-finds old landmarks, and
the pose graph corrects the spline at loops.

## Reused and new

Everything that produces or matches features stays as it is. The change is
confined to how observations are stored and how the trajectory and map are
solved.

| Component | Status | Where |
| --- | --- | --- |
| NVDEC decode, SuperPoint (TensorRT), NVOF flow | Reused unchanged | `ffmpeg_cuda_decoder.cpp`, `tensorrt_superpoint.cu`, `optical_flow.cu` |
| Track association: flow prediction, radius gate, descriptor cosine gate | Reused unchanged | `flow_association.cu` |
| Landmark descriptors, re-association in the local map, place search, P3P RANSAC relocalization | Reused unchanged | `visual_odometry.cpp`, `place_index.*`, `vo_geometry.*` |
| Feature cache on disk (detections and descriptors) | Reused unchanged | `feature_store.cpp` |
| Two-view initialization and the per-frame pose-only solve | Reused; the pose solve now initializes the newest control point | `visual_odometry.cpp` |
| Triangulation of new landmarks | Reused, relaxed to any two frames with enough parallax | `triangulate_new()` |
| Sim(3) pose graph for loops and merges | Reused; the correction is applied to control points | `pose_graph.*` |
| Observation store: landmark id, timestamp, pixel for every associated observation | New | new file |
| Trajectory spline: evaluation and derivatives with respect to control points | New | new file |
| Bundle adjustment with control points as unknowns | New; replaces `local_bundle_adjustment()` | `vo_geometry.*` |
| Cluster sampler and background solver | New; runs on the existing worker thread | new file |
| Per-frame poses stored relative to a keyframe (`FrameRecord::relative`) | Removed; a frame keeps only its timestamp | `visual_odometry.cpp` |

The solver never reads descriptors, so the observation store costs about 16
bytes per observation (24 with the raw detection, see below): roughly 16 KB
per frame at 1,000 tracks, or under 600 MB for ten minutes at 60 fps.

## The solver loop

The finished system runs two loops: a per-frame loop that must keep up with
the video, and a background loop that refines the rest of the map whenever
the worker is free.

**Per frame**

1. Track and associate as today: the flow tracker carries tracks forward, and
   local landmarks without a live track are projected and matched by
   descriptor.
2. Solve this frame's pose from its landmark observations, starting from the
   spline extrapolated to the frame's timestamp, with P3P RANSAC as the
   fallback.
3. Append every associated observation to the store.
4. Extend the spline. When the frame's time passes the last knot, add a
   control point initialized from the pose in step 2.
5. Triangulate new landmarks from tracks that now have enough parallax.
6. Solve the newest stretch: the last few control points and the landmarks
   they see, with all of their observations.

**Background, on the worker**

1. Pick a cluster: a stretch of consecutive control points, plus any
   stretches tied to it by shared landmarks from a revisit. Favour clusters
   with high residuals or recently changed neighbours.
2. Hold the neighbouring control points fixed. They set the gauge, including
   scale.
3. For each landmark in the cluster, sample observations: one per time bin
   along its track, with extra weight on the ends of the track, and jitter
   within each bin. The sample is fixed for the whole solve.
4. Solve the cluster's control points and landmarks together.
5. Accumulate the result instead of replacing the old one, by a damped step
   or a carried prior per landmark. Without this the map only ever has the
   accuracy of one sample.
6. Publish the new control points and discard any cached poses.

**On a loop or a segment merge**

The Sim(3) pose graph runs as it does now and its correction is applied to
control points. The background loop then cleans up locally, which covers the
global bundle adjustment that ARCHITECTURE.md lists as not done.

## Ground rules for every milestone

- One milestone per branch and pull request. Stop at the gate and report the
  numbers; do not start the next milestone in the same change.
- New behaviour goes behind a `VisualOdometryConfig` field and a bench key.
  With the key off, every baseline trajectory stays byte-identical to
  `baselines/v5` (`tools/baseline.py --skips 0,150,300`).
- Runs stay deterministic. Anything random takes a seed, and anything on the
  worker thread has a synchronous mode for the bench and the tests, as
  `place-sync` does.
- Each milestone adds its CTest cases and, once its gate passes, a baseline
  snapshot (`baselines/v6`, ...) with a row in `baselines/README.md`.
- Respect the contracts in ARCHITECTURE.md: track IDs are never reused,
  tracks are in distorted source pixels and are undistorted on input, only
  supported observations reach the VO.
- Where this document does not settle a design choice, stop and ask.

## Milestones

Correctness comes before sampling: milestones 0 to 7 are deterministic, and
milestone 7 produces the reference answer that the stochastic solver in
milestone 8 is tested against. A random solver has no other way to be shown
correct.

"v5" below means the numbers in `baselines/README.md` (ATE in cm, mean over
the three start frames): fr1_desk 3.2, fr1_room 39.3, fr1_xyz 1.5, fr2_desk
5.0, fr3_long_office 3.9. "ATE no worse" means the mean over the three starts
is at most 5% or 0.2 cm above v5's, whichever is larger.

| # | Build | Gate before moving on |
| --- | --- | --- |
| 0 | **Measure the premise.** The landmark depth metric and the fixed-pose experiment described below. Also run the bench with `kf-min 1 kf-max 1 window 30 local-map 30`. | Median landmark depth error with all observations is at most 0.85× the keyframes-only error on at least 4 of the 5 TUM sequences. If it is not, stop here. |
| 1 | **Observation store.** Keep every associated observation, fused position and raw detection. No change in behaviour. | Trajectories byte-identical to v5 on every baseline run. Memory within 20% of the estimate above. |
| 2 | **Structure-only refinement.** After each keyframe solve, refit the non-keyframe poses in the window, then re-solve every local landmark against all of its observations (3 unknowns per point, Huber loss, poses fixed). | Median landmark depth error at most 0.9× v5's on at least 4 of 5 sequences. ATE no worse. |
| 3 | **Spline library.** Split rotation and translation cumulative cubic B-spline with uniform knots: evaluation, velocity, analytic derivatives with respect to control points. | Derivatives match central finite differences to 1e-6 relative. A trajectory drawn from a spline is recovered to 1e-9 relative. A fit to each TUM sequence's motion-capture trajectory at several knot spacings gives the residual against spacing; the default is the largest spacing whose residual, projected at the sequence's median scene depth, is below 0.3 px RMS. If the motion-capture noise floor prevents that, report the curve and ask. |
| 4 | **Passive spline.** A sidecar that fits one spline per segment to v5's per-frame poses and feeds nothing back, like live depth. It must handle coasting gaps, lost tracking, seeks, loop corrections and merges. | Reprojection RMS with spline poses at most 1.05× that with discrete poses, including after a loop correction. ATE within 0.1 cm of v5. |
| 5 | **Control points as unknowns.** Replace `local_bundle_adjustment()` for the newest stretch, with all observations. | A noise-free trajectory drawn from a spline is reproduced to 1e-4% ATE. The existing synthetic tests in `native_visual_odometry` pass with their current thresholds. TUM ATE no worse than v5; landmark depth error no worse than milestone 2. |
| 6 | **Loops, merges and relocalization on the spline.** Pose-graph corrections move control points; resuming an old place starts a new stretch. | At least as many loops closed per run as v5. ATE no worse than v5 on fr1_desk, fr2_desk and fr3_long_office. |
| 7 | **Deterministic background solver.** Clusters swept in a fixed order with all observations, to convergence. | Total cost never increases between sweeps. On fr1_xyz the result agrees with one global solve to 0.05 cm ATE and 1% median landmark depth error. This is the reference. |
| 8 | **Stochastic sampling.** Prioritized random cluster choice, stratified and jittered observation sampling, accumulation across solves. Seeded, with a synchronous mode. | Reaches milestone 7's ATE within 0.1 cm and its landmark depth error within 2%, at no more than 25% of milestone 7's cost per solve. The same seed gives byte-identical output. Worst frame time on the calling thread no higher than v5's. |
| 9 | **Later: rolling shutter and IMU.** Per-row timestamps with a line delay; gyro and accelerometer residuals on the spline's derivatives. Needs the iOS logger data. | Not scoped yet. |

Milestones 0 to 2 are worth doing even if the spline is never built. They
test whether every-frame observations help at all, and milestone 2 keeps the
gain with a small change to the current code.

### Milestone 0 in detail

Nothing in the VO changes. The work is a metric, a bench option and a table.

**Landmark depth metric** (a new `tools/landmark_depth_vs_gt.py`, following
`tools/depth_vs_gt.py`):

- Input: a bench export with one row per (landmark, keyframe) observation:
  frame index, landmark id, the observed pixel (undistorted and as tracked),
  and the landmark's depth in that keyframe's camera.
- Truth: the TUM depth image nearest in time to that frame, within 20 ms,
  sampled at the tracked pixel. Skip pixels with no depth and pixels whose
  3×3 neighbourhood varies by more than 5% (depth edges).
- Scale: one per frame, the median log ratio over that frame's landmarks, as
  `depth_vs_gt.py` does. Skip frames with fewer than 20 scored landmarks.
- Report per sequence: median `|log z - log z_true - scale|`, the share
  within 5% and 10%, and the number of (landmark, keyframe) pairs scored.

**Fixed-pose experiment** (a bench key, for example `resolve-landmarks K`):

- Record every frame's observation of each landmark during the run, both the
  fused track position and the raw supporting detection (the track record's
  `detection` index).
- At the end of the run, with all frame poses fixed at their final values
  (frame pose = `relative` × its keyframe's pose), re-solve each landmark
  from K of its observations: K = 2 (first and last), 4 and 8 (evenly spread
  in time), and all. Keep the keyframes-only landmark positions as the
  baseline.
- Score each variant with the metric above, on the same (landmark, keyframe)
  pairs, for fused and for raw positions.

**Deliverable:** a table of sequence × variant in the pull request, the
`kf-min 1 kf-max 1 window 30 local-map 30` ATE next to v5's, and a go or
no-go against the gate.

## Risks and open questions

The largest risk is that the gain is small, and milestone 0 exists to find
that out before anything is built.

- **Correlated track noise.** Track positions are Kalman-fused with flow
  (`flow_sigma_px = 0.3`), so errors on consecutive frames are not
  independent. The store keeps the raw supporting detection as well, and
  milestone 0 scores both. Which one the solver uses is decided from that
  table.
- **Knot spacing.** Too sparse smooths away hand shake and biases the
  reprojection. Too dense gives per-frame poses again. Milestone 3 measures
  it. Ovrén and Forssén's spline error weighting is the known mitigation.
- **A spline is not guaranteed to beat discrete poses.** Cioffi et al. found
  continuous time clearly better when sensors are not time-synchronized. For
  one global-shutter camera the case is weaker, and for a rolling-shutter
  phone with an IMU it is strong.
- **Circularity in milestone 2.** Non-keyframe poses were fitted to the
  landmarks, and landmarks are then fitted to those poses. This is valid
  alternation, but the gain will be less than the table above suggests.
- **Gauge.** Every cluster solve needs fixed neighbours to hold scale. The
  first stretch of a segment has none and needs an explicit scale constraint.
  The `kf-min 1` experiment has the same problem: the two fixed keyframes
  become adjacent frames with almost no baseline.
- **Gaps and shocks.** Lost tracking, cuts and new segments each need their
  own spline. A sharp jolt breaks the smoothness assumption inside a stretch.
- **Accumulation in milestone 8.** A carried prior and a later sample can
  contain the same observations, which counts evidence twice. Open: damped
  steps, or priors with bookkeeping of what they already contain?
- **Threading.** Control points are read by the per-frame loop and written by
  the worker. The snapshot-and-swap pattern the place search uses should
  carry over.
- **Rolling-shutter degeneracy (milestone 9).** Video has the same readout
  direction in every frame, which Albl et al. reported can let rolling-shutter
  bundle adjustment collapse toward a flat scene. Not yet verified against
  the paper.
- **Novelty.** A brief search found continuous-time bundle adjustment with
  rolling shutter, and stochastic bundle adjustment, but nothing that
  combines cluster sampling with a spline trajectory.

## Sources

Opened while preparing this document:

- Strasdat, Montiel, Davison, ["Visual SLAM: Why Filter?"](https://www.doc.ic.ac.uk/~ajd/Publications/strasdat_etal_ivc2012.pdf) (2012)
- Ovrén and Forssén, ["Trajectory Representation and Landmark Projection for Continuous-Time Structure from Motion"](https://arxiv.org/abs/1805.02543) (2019)
- Cioffi et al., ["Continuous-Time vs. Discrete-Time Vision-based SLAM: A Comparative Study"](https://arxiv.org/abs/2202.08894) (2022)

Cited from memory, not re-read:

- Furgale, Barfoot, Sibley, "Continuous-Time Batch Estimation Using Temporal Basis Functions" (2012)
- Lovegrove, Patron-Perez, Sibley, "Spline Fusion" (2013)
- Sommer et al., "Efficient Derivative Computation for Cumulative B-Splines on Lie Groups" (2020)
- Ovrén and Forssén, "Spline Error Weighting for Robust Visual-Inertial Fusion" (2018)
- Albl et al., "Degeneracies in Rolling Shutter SfM" (2016)
- Zhou et al., "Stochastic Bundle Adjustment for Efficient and Scalable 3D Reconstruction" (2020)
- Sucar et al., "iMAP" (2021), for replaying random old keyframes
