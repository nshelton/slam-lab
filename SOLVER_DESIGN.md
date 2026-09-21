# Modular reconstruction design

Status: the first modular implementation is available as `verify-matches`,
`view-geometry`, and `solve`; see [SOLVER.md](SOLVER.md) for usage and current limits,
and [WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md) for the GUI integration contract.
The broader architecture below still includes planned capabilities: global solvers,
track splitting, individual association-conflict records, and periodic/local BA.
The original video-order mapper remains internal for shared machinery and tests; it has
no CLI. The manifested modular workflow is the only supported interface.

Workbench integration priorities and acceptance criteria are tracked separately in
[SOLVER_TODO.md](SOLVER_TODO.md): artifact/observation identity and structured solve
progress first, then immutable publication, live snapshots and supported readers.
These are planned API additions; see the interface document for current vs proposed
behavior.

## Proposed stage boundaries

Features → appearance matches → pairwise geometric verification → verified tracks
and view graph → initialization → camera/point reconstruction → bundle adjustment
and observation filtering.

Pose estimation, triangulation, bundle adjustment and filtering normally repeat;
this is not a single pass that solves cameras once and then permanently fixes them.

| Stage | Input | Persistent output |
| --- | --- | --- |
| Features | Images and extraction configuration | Stable image/feature IDs, pixel coordinates, descriptors, preview dimensions/timestamps |
| Matching | Features and selected image pairs | Immutable raw feature-index pairs, scores and matcher provenance |
| Pair verification | Raw matches, feature coordinates, camera model | F/E/H hypotheses, individual inlier masks, residuals, optional relative poses and quality diagnostics |
| Tracks | Verified feature connections | Candidate observations of the same point, at most one feature per image, retained edge provenance and rejected conflicts |
| View graph | Verified image pairs | Image nodes and pair edges with relative rotation, translation direction and quality/degeneracy flags |
| Initialization | Tracks, view graph, camera model | Initial camera poses and, where required, initial points; seed candidates and rejection reasons |
| Reconstruction | Initialization, all eligible tracks, camera model | Registered cameras, triangulated points, unresolved images/observations and decisions |
| Refinement | Cameras, points and observations | Jointly optimized solution, per-observation errors and explicit removal decisions |

Each output references the IDs and content hashes of its inputs and its own
configuration. Changing the matcher reuses feature extraction. Changing geometric
thresholds reuses matches. Changing the camera model invalidates geometry that
depends on calibration, initialization and downstream reconstruction, not descriptors
or appearance matches. The uncalibrated F fit can remain reusable.

Do not overwrite raw matches with filtered matches. Keep a raw RANSAC mask separate
from cheirality, triangulation and final bundle-adjustment masks. Make intermediate
artifacts readable by Rerun without running inference or solving again.

The matching run's `tracks.npz` is an appearance-only preview. The geometry stage
now rebuilds a separate `tracks.npz` from verified edges; do not substitute the
matching-stage file as established landmarks.
Pair verification and frame-unique track membership still do not guarantee a
correct, static-world point: later multiview checks may split tracks or reject
observations.

## Pairwise geometry

For each pair with sufficient candidate matches:

1. Estimate a fundamental matrix F robustly in image coordinates. This tests the
   epipolar relationship without relying on our assumed focal length.
2. Given the shared fixed intrinsic matrix K, estimate an essential matrix E
   robustly from normalized observations. Recover relative rotation and translation
   direction where observable. With trustworthy calibration E may be the primary
   estimator; F remains useful as a separate diagnostic for our guessed calibration.
3. Estimate a homography H to assess planar or near-pure-rotation explanations.
   Compare model quality and pose/triangulation diagnostics. Do not declare every
   H-consistent pair useless: its correspondences may still be valuable even when
   it is unsuitable for initializing general 3D.
4. Store support count/ratio, spatial coverage, epipolar error, positive-depth
   fraction, triangulation-angle distribution, rotation, direction and model status.

Use undistorted coordinates when a distortion model is provided and keep pixel
thresholds consistent with cached image dimensions. The first version can use
the current zero-distortion, fixed-intrinsic assumption explicitly.

For ideal pinhole image coordinates, E = K_j^T F K_i. In practice estimate E with
its calibrated constraints rather than treating a noisy transformed F as an exact
essential matrix. If F fits well but E does not, inspect the calibration assumption,
degeneracy and motion model; that disagreement is not proof of any one cause.

An essential matrix gives translation **direction**, not its magnitude. Recovered
translations are typically normalized, so comparing their lengths does not rank
baseline, and concatenating them as equal-length camera steps is invalid.
Connected monocular reconstruction has one unresolved overall scale unless an
external metric constraint is supplied.

Proposed pair record fields:

```text
PairGeometry(image_i, image_j)
  source_match_run, source_match_indices, camera_model_hash, estimator_configuration
  F, F_inliers, F_residuals
  E, E_ransac_inliers, E_residuals, cheirality_mask
  H, H_inliers, H_residuals
  R_j_from_i, unit_t_j_from_i                 # optional when observable
  inlier_count, coverage_i, coverage_j, parallax_quantiles
  verified, rotation_usable, translation_usable, seed_eligible, rejection_reasons
```

The view graph contains image-level relationships. Tracks contain feature-level
relationships. Keep both: they answer different questions and support different
solver backends. Preserve disconnected components rather than fabricating poses
that connect them.

For Osaka, moving people and screen-fixed captions remain important failure cases.
A moving object can satisfy its own two-view geometry. Audit spatial support and
consistency across multiple frames, and support explicit exclusion masks. RANSAC
inlier count alone is not a static-background guarantee.

## Seed selection and initialization

For a seeded solver, rank image pairs from the verified view graph. Use overlap
and graph connectivity to propose candidates, then test actual geometry. A candidate
needs sufficient, spatially distributed inliers, usable parallax, positive-depth
triangulations, low reprojection errors and connections to additional views.

High baseline is useful only while enough reliable overlap remains. Use the angle
between corresponding viewing rays after accounting for rotation; raw pixel motion
or timestamp separation is not a baseline measurement. The unit translation from
essential-matrix decomposition provides no absolute-baseline ranking either.

Try several top-ranked candidates. Proposed additional safeguard: triangulate a
candidate pair, use its tracks to register a third view, and check the reconstruction
before committing to that seed. Keep every rejection reason visible. Fix the seed's
first pose and one baseline length to choose the coordinate system and scale.
The first video frame does not need to belong to the seed.

## Interchangeable solver backends

The solver contract should consume geometry, not matcher internals:

```text
ReconstructionProblem
  cameras: camera_id -> model and fixed/optimizable parameters
  images: image_id -> camera_id, image size, timestamp
  observations: (image_id, feature_id) -> pixel_xy
  tracks: track_id -> observations and verification provenance
  view_graph: verified PairGeometry records

solve(problem, configuration) -> Reconstruction
  camera poses and registration status
  3D points and surviving observation-to-point assignments
  per-observation residuals, rejected observations and diagnostics
  coordinate/scale convention and input provenance
```

No descriptors or SuperPoint/LightGlue objects are required downstream of matching.
An adapter can therefore accept cosine, LightGlue, optical-flow or externally
supplied correspondences. External inputs must supply stable observation IDs and
image coordinates with a documented coordinate convention. Supplied tracks remain
subject to geometric validation. Matcher confidence is provenance, not a universally
calibrated probability or interchangeable geometric weight.

Two useful backends share these inputs:

* **Graph-based incremental SfM:** initialize from a strong pair, select the next
  image by available 2D–3D support across the reconstructed scene, register with
  PnP RANSAC, triangulate/extend tracks, and run local and periodic global bundle
  adjustment. Images need not be processed in video order. This can consume the
  entire precomputed correspondence graph, unlike our current limited-reference
  implementation. A COLMAP adapter is a useful established reference.
* **Global SfM:** robustly average relative rotations, then solve camera positions
  and point structure jointly (or use another suitable global positioning method),
  followed by bundle adjustment and filtering. This avoids an incremental seed
  pair but still requires initialization, observable geometry and outlier handling.
  GLOMAP/COLMAP's global mapper is a useful alternative backend. Keep calibration
  quality and approximately collinear walking trajectories in mind when evaluating it.

In either backend, a landmark observed in many frames contributes its actual image
observations to bundle adjustment once each. Its many pairwise matches must not
be counted as independent repeated reprojection measurements.

## Implemented first slice and next work

Independent F/E/H verification, raw mask persistence, Rerun pair inspection,
verified tracks, `ReconstructionProblem`, and the first graph-based incremental
backend are implemented. Synthetic tests exercise outliers, pure rotation, planar
and low-parallax cases, pose/scale conventions, raw masks and imported correspondences.
The 30-frame cosine Osaka experiment is described in [SOLVER.md](SOLVER.md).

The next backend integration work is [SOLVER_TODO.md](SOLVER_TODO.md): stable
artifact/observation mappings and structured solve progress before the GUI relies
on those APIs. The GUI is being developed separately against the file/reader contract.

The next numerical comparison is the same 30-frame selection using the now-complete
LightGlue matches. Inspect the geometry and trajectory, including the earlier failure
around 10 seconds, before assuming a low reprojection error means a correct path.
Then prioritize a mature solver adapter as a reference, calibration sensitivity,
periodic/local BA and track/dynamic-scene filtering. Solver experiments should reuse
the feature and matching artifacts. A custom global optimizer remains future work.

## References

* [COLMAP tutorial](https://colmap.github.io/tutorial): separates matching/verification
  from reconstruction and supports imported features and matches.
* [Structure-from-Motion Revisited, CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf):
  two-view verification, seed selection, registration, triangulation and refinement.
* [COLMAP initialization implementation](https://github.com/colmap/colmap/blob/main/src/colmap/sfm/incremental_mapper_impl.cc):
  candidate selection and checks on support, forward motion and triangulation angle.
* [OpenCV calibration and reconstruction reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html):
  robust F/E estimation, relative pose recovery, cheirality and translation-scale ambiguity.
* [Global Structure-from-Motion Revisited, ECCV 2024](https://arxiv.org/abs/2407.20219)
  and [GLOMAP pipeline](https://github.com/colmap/glomap/blob/main/glomap/controllers/global_mapper.cc):
  view-graph processing, rotation averaging, track construction, global positioning and refinement.
* [COLMAP commands](https://colmap.github.io/cli.html): standalone geometric verifier,
  incremental mapper, global mapper, calibration and rotation-averaging stages.
