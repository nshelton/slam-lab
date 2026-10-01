# Live depth image: plan

Status 2026-10-01: build-order steps 1 to 4 are done and measured (bench,
keypoints-only floor, single measurement, propagation and fusion, keypoint
scale and combination; see "Measured so far" and "The filter"), and the
overlay (6) is in the workbench: *Depth* in the Pipeline panel. Not done:
regularization, the voxel feed.

A full-resolution inverse-depth and confidence image that runs beside the
visual odometry, is carried forward every frame and refined by the camera
tracking, and stays in the scale of the triangulated keypoints. It is a
sidecar: it reads the VO's poses and landmarks and the optical flow, and
feeds nothing back. Its consumers are display (an overlay on the video) and
the voxel map.

## Decisions (2026-10-01)

- **Geometric first.** Depth comes from optical flow, the VO pose and the
  landmarks. No depth network; one may be added later as a prior for regions
  that never get parallax, only if those holes matter.
- **Full resolution.** The state is one inverse depth and one variance per
  source pixel (1920×1080 on the 1080p clips).
- **Sidecar.** The VO and the tracker are not changed. Using the depth for
  tracking is a later question.
- **First consumer: an overlay on the video panel**, then the voxel map.

## What full resolution costs

The optical flow is not full resolution and cannot be on this machine: the
RTX 2080 Ti's optical-flow engine reports 4×4 as its only output grid
(`NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES` = 4; probed 2026-10-01). So a flow
vector is shared by 16 pixels, and bilinear upsampling gives a smooth guess
rather than a per-pixel measurement.

The per-pixel measurement therefore needs one more step than the flow: a
short photometric search at full resolution. With the pose known, a pixel's
match in the previous frame lies on its epipolar line; the upsampled flow
says where on that line to look, and the search only refines it by a pixel
or two. That is LSD-SLAM's depth measurement without its long line search.
It needs the previous frame's luma kept on the device (one 2 MB copy per
frame at 1080p; a decoded surface is only valid until the next decode).

## State

Per source pixel, in the current frame, on the device:

- `inverse_depth` (float): 1 / z in the camera frame, in the VO segment's
  units. Inverse depth because its error is close to Gaussian for a
  triangulated point and distant points stay finite.
- `variance` (float): of the inverse depth. Confidence shown to the user is
  derived from it. A pixel with no estimate has infinite variance.
- Bookkeeping: frames since the pixel was first seen, for display and for
  gating what reaches the voxels.

Coordinates are source pixels of the native frame, lens-distorted, like
everything else; rays are undistorted with k1 exactly as the VO does.

## Per frame

After `odometry->process()` in `Session::process_frame` (`src/app.cpp`), on
its own CUDA stream:

1. **Propagate.** Forward-warp the previous image into the current frame
   with the relative pose: back-project each pixel with its depth, transform,
   project, keep the nearest per target pixel (occlusion), and transform the
   inverse depth with it. Variance grows by a process term and by the pose
   uncertainty the VO already reports. Disoccluded pixels start empty.
2. **Measure.** For each pixel: epipolar line from the relative pose; start
   from the upsampled flow; refine along the line by matching a small luma
   patch against the previous frame, sub-pixel by a parabola fit. Inverse
   depth from the disparity along the line; variance from the parallax
   (baseline against depth), the image gradient along the line and the match
   residual. Rejected: gradient too weak along the line, flow far off the
   line (moving object, bad flow), residual too high.
3. **Anchor to the keypoints.** Tracks with a landmark have a depth in the
   VO's scale at a known pixel. Fit one robust scale between the image and
   those depths and apply it, so the image follows bundle adjustment and
   scale drift; then fuse the landmarks in as low-variance measurements.
4. **Fuse.** Per pixel, the product of the propagated estimate and the
   measurement (a scalar Kalman update); a measurement that disagrees by more
   than a few sigma replaces a weak estimate and is ignored by a strong one.
5. **Regularize** (small): one pass that fills isolated holes and removes
   isolated outliers from confident neighbours, weighted by variance and by
   luma similarity so it does not cross edges.

Without a pose (initializing, coasting, lost): step 1 is replaced by a pure
flow warp with a larger process term, steps 2–4 are skipped. A new segment
has a new frame and scale: the image is reset.

## Known hard parts

- **No parallax.** Pure rotation or a static camera gives no measurement; the
  image can only be carried along. This is where a network prior would help.
- **Tiny baselines.** One frame of motion gives too little parallax
  (measured: 12% error against the previous frame, 3.5% ten frames back), so
  the measurement uses an older reference frame. A fixed gap is a stand-in:
  the reference should be chosen by baseline against scene depth, and
  keyframes (bundle-adjusted poses) are the natural candidates.
- **Flow is a 4×4 block estimate.** At depth edges the starting point of the
  search is wrong for one side of the edge; the search range has to cover it.
- **Rolling shutter and moving people** are only handled by the rejection
  tests.
- **Scale.** The anchor step assumes one scale error for the whole image.

## Consumers

- **Overlay** on the video panel: inverse depth as colour, confidence as
  opacity; a readout under the cursor.
- **Voxel map.** At keyframes, pixels above a confidence threshold become
  `VoxelSample`s (point plus this frame's camera centre) through
  `VoxelHash::integrate()`. At full resolution that is up to 2 M points per
  keyframe, so they are subsampled, and the voxel side first needs what its
  measurements already call for: integration without a full rebuild per
  frame, and a ray budget for the ambient pass (630k voxels: 41 ms per pass
  today).

## Measured so far (steps 1 and 2)

`LiveDepth` (`include/slam_native/live_depth.hpp`, `src/live_depth.cu`) has
two modes, both without memory between frames:

- **anchors**: every pixel takes the inverse depth of the landmarks around it
  (inverse distance weighting). No image evidence. This is the floor.
- **measurement**: step 2 above, against a reference frame `reference_gap`
  frames back. The last 32 frames' luma and flow fields are kept on the GPU;
  the flow is chained back through them for the starting point, and the patch
  is warped by the two poses (rotation and scale between the frames).

TUM, first 800 frames, median depth error after one scale per frame:

| Sequence | Keypoints only, all pixels | Measurement, gap 10 | Share of pixels measured |
| --- | --- | --- | --- |
| fr1_desk | 4.8% | 3.5% | 19% |
| fr2_desk | 5.4% | 3.9% | 24% |
| fr3_long_office | 5.0% | 3.5% | 21% |

The reference frame matters most (fr3_long_office, median error / share of
pixels measured): gap 1: 12.4% / 25%; gap 3: 6.0% / 24%; gap 5: 4.9% / 23%;
gap 10: 3.5% / 22%; gap 20: 3.1% / 19%. Matching the previous frame is not
worth doing; ten frames back is the default.

**The measurement alone does not beat the keypoints yet.** It only covers
textured pixels, and those are where the keypoints are. On exactly the pixels
it measures, interpolating the keypoints is better: 2.6% against 3.2%
(fr1_desk), 3.5% against 4.0% (fr2_desk), 2.9% against 4.0% (fr3). The
ground truth is a Kinect, so part of both numbers is its own noise; how much
has not been separated. So the case for the dense image rests on steps 3
and 4 (fusing many measurements and the keypoints), and that is the next
thing to measure.

Other observations:

- Half the pixels have too little gradient along their epipolar line
  (35–60% on TUM, 60% on `disney_04`), and 8–25% are rejected because the
  chained flow lands off the line. `disney_04` (1080p) measures 10%.
- The reported variance is optimistic on real data at long gaps (median
  error / sigma about 2 on fr2 and fr3, 4 on fr1) and pessimistic on the
  synthetic scene (0.15): it has no term for the pose error, and a constant
  floor instead. To fix when the fusion needs it.
- Cost: 1.1 ms per frame at 640×480, 3.0 ms at 1920×1080 (gap 10).
- The synthetic test (`native_live_depth`: a slanted textured plane, exact
  flow) recovers the plane to 0.3% at gap 5 and 0.12% at gap 12, also with
  lens distortion: the geometry is right, the real-data error is noise, pose
  error and ground truth.

Reproduce:

```bash
B=native_workbench/build/voxel-debug; D=data/datasets/tum/rgbd_dataset_freiburg3_long_office_household
$B/slam-native-live-depth-bench $D/rgb.mp4 native_workbench/models/superpoint-1024x576-k2048.engine 800 OUT \
  fx 535.4 fy 539.2 cx 320.1 cy 247.6 mode measurement gap 10
.venv-cuda/bin/python native_workbench/tools/depth_vs_gt.py $D OUT   # then delete OUT (2.4 MB per dumped frame)
```

## The filter (steps 3 and 4)

Two more modes of `LiveDepth`:

- **filter**: the state is carried into each frame, rescaled to the
  landmarks under it, and fused with the frame's measurement.
- **fused**: the filter's image combined, per pixel, with the keypoints-only
  interpolation, which also fills what the filter has nothing for. The
  keypoints are not folded into the state, so they are not counted again
  every frame.

How it differs from the plan above:

- **Propagation is a backward warp.** Each current pixel looks up where the
  flow says it came from, takes the nearest estimate there, moves that point
  with the camera, and shifts the lookup once so the point really lands on
  the pixel. No scatter, no depth buffer, no holes; a point that still lands
  more than 1.5 px away is dropped (disocclusion, or flow against pose).
- **Every measurement gets 6% relative sigma added** for what the matching
  noise does not cover (poses, model), and the fused sigma never goes below
  1%. Disagreeing estimates (2.5 sigma) are not averaged: the more certain
  one wins, and an estimate that keeps being contradicted loses certainty.

TUM, first 800 frames, median depth error after one scale per frame:

| Sequence | Keypoints only | Fused | Keypoints only, textured px | One measurement, same px | Fused, same px |
| --- | --- | --- | --- | --- | --- |
| fr1_desk | 4.6% | 4.7% | 2.5% | 3.4% | 2.5% |
| fr2_desk | 6.1% | 4.7% | 3.5% | 4.0% | 2.7% |
| fr3_long_office | 5.0% | 4.2% | 2.9% | 4.0% | 2.5% |

("Textured px": the fifth of the image a single measurement covers. The
first two columns are over every pixel with ground truth.)

- Fusing over time works: on the same pixels the error falls from 3.4–4.0%
  for one measurement to 2.5–2.7%.
- Against the keypoints alone the gain is modest and uneven: clear on
  fr2_desk and fr3, none on fr1_desk (fast motion; the filter holds an
  estimate for only 50% of pixels there, against about 80% on the others).
- The gain is away from the keypoints (fr3, pixels where the filter has an
  estimate, keypoints only against fused): within 10 px of a keypoint 2.7%
  against 2.7%; 10–30 px: 4.6% against 3.7%; 30–60 px: 8.5% against 5.3%;
  beyond 60 px: 12.2% against 7.1%. Where neither texture nor a keypoint is
  near, nothing helps: those pixels stay at the interpolation's error.
- The pixels the filter is sure of are good: sigma below 2% holds for
  16–21% of pixels, at 2.0–2.1% median error.
- The variance is still optimistic by a factor of 1.5–2 (median error over
  sigma), mostly because consecutive measurements share frames and poses and
  are fused as if independent.
- The parameters hardly matter (fr3, fused, textured pixels): measurement
  sigma 3 / 6 / 10%: 2.63 / 2.48 / 2.46%; gap 5 / 10 / 20: 2.76 / 2.63 /
  2.66%; no scale fit: 2.57%. The remaining error is not the filter's
  tuning. The keypoints' own depths are about 3% off in the median, and the
  filter uses the same poses.
- Cost, measured while another job used the GPU, so upper bounds: 3.5–4.4 ms
  per frame at 640×480, 5.9 ms (filter) and 9.7 ms (fused) at 1920×1080. The
  keypoint interpolation is the expensive part of "fused" at 1080p: every
  pixel visits every keypoint.

## The overlay (step 6)

*Depth* in the Pipeline panel draws the image over the video
(`DepthOverlay`, `src/depth_overlay.cu`: coloured on the GPU into a GL pixel
buffer). Red is near and blue far, between the 5th and 95th percentile of
the landmarks' depths in view; opacity falls with the relative sigma, to
nothing at *Max sigma*. The mode selector switches between keypoints only,
one measurement, the filter and fused; hovering shows the depth and its
sigma under the cursor. `LiveDepth` exists only while the box is ticked, so
the pipeline costs nothing extra otherwise; it starts with the next frame
and restarts on a seek, a lens change, a loop correction or a merge.

The keypoint interpolation is now computed on a 4×4-pixel grid and
upsampled, which is what made "fused" affordable at 1080p. On `disney_04`
(600 frames, Debug build) the last frame took 7.2 ms fused and 3.1 ms for
the filter alone, with 100% and 86% of pixels estimated and 14% measured in
that frame.

What the picture shows on `disney_04`: near and far are right at a glance
(lamp and pillar against the back wall), edges and textured surfaces are
clean, and the ceiling and floor are patchy: blobs of differing depth where
single uncertain estimates were carried along. That is the regularization
step's job.

`SLAM_SNAPSHOT_DEPTH=3 slam-native-snapshot VIDEO ENGINE 0 600 OUT.ppm`
renders it headlessly.

## Measuring it

TUM RGB-D has ground-truth depth per frame (`data/datasets/tum`,
`tools/lsd_depth_vs_gt.py` already compares depth exports against it). A
headless bench runs the real pipeline and reports, per sequence and per
confidence threshold: coverage (share of pixels with an estimate), and
against ground truth after one scale per frame the absolute relative error
and the share within 1.25×. Every step below is measured with it before the
next one starts. The 1080p clips have no ground truth: there the checks are
the overlay, the reprojection of the depth at tracked keypoints against
their landmarks, and timing.

## Build order

Steps 1 to 4 are done. What the measurements suggest for what is left: the
regularization (5) should be aimed at the pixels far from keypoints, the
keypoint interpolation needs a cheaper form before the overlay runs at 1080p
(6), and choosing the reference by baseline rather than a fixed gap would
help fr1_desk-like fast motion.

1. **Bench and container.** `LiveDepth` class with the state, reset and
   export; the bench with ground-truth scoring; previous-luma copy. Scored
   with a trivial estimate (landmark depths splatted) as the floor.
2. **Measurement only.** Step 2 per frame with no memory, to see what one
   frame pair gives: coverage, error and variance calibration (is the error
   inside the reported sigma?).
3. **Propagate and fuse.** Steps 1 and 4. The number to beat is step 2's.
4. **Keypoint anchor.** Step 3, measured as error with and without.
5. **Regularize.** Step 5, measured the same way.
6. **Overlay** in the workbench, timing on the 1080p clips.
7. **Voxels**, after the voxel map integrates incrementally.

## Files

New: `include/slam_native/live_depth.hpp`, `src/live_depth.cu`,
`tests/test_live_depth.cpp` (synthetic: a textured plane and a box under a
known camera path), `tests/bench_live_depth.cpp`, `tools/depth_vs_gt.py` if
the LSD script does not fit as it is.

Shared, small and additive: `src/app.cpp` (one call after the odometry, the
overlay, a toggle), `CMakeLists.txt`. Nothing in `visual_odometry.*`,
`vo_geometry.*` or the tracker.
