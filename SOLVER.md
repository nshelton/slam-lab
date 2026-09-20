# Modular solver v1

The new path consumes saved correspondences from any matcher:

```text
matches -> pairwise F/E/H RANSAC -> verified tracks -> seed + third view
        -> all-track PnP + triangulation -> joint bundle adjustment -> cameras/points
```

Appearance matches are immutable inputs. Verification is cached separately and can
be rerun with new intrinsics or thresholds without feature extraction or matching.
This first backend is graph-based incremental SfM: it chooses images by available
2D–3D support, across the precomputed tracks, instead of using a few recent frames.
The stage boundary permits a later global solver. The original `reconstruct`
command is retained separately.

## Run on Osaka

The first experiment selects 30 images across the first 30 seconds (every tenth
row of the existing 10 fps cache). All 435 pairs among those selected images are
verified. It uses the complete cosine match run:

```bash
.venv/bin/slam-lab verify-matches recordings/osaka-allpairs-cosine \
  --output recordings/osaka-cosine-geometry-step10 --frame-step 10 --max-frames 30

.venv/bin/slam-lab geometry-status recordings/osaka-cosine-geometry-step10

.venv/bin/slam-lab solve recordings/osaka-cosine-geometry-step10 \
  --output recordings/osaka-cosine-solver-v1

.venv/bin/slam-lab view-reconstruction recordings/osaka-cosine-solver-v1
.venv/bin/slam-lab view-geometry recordings/osaka-cosine-geometry-step10 --pair 100 110
```

Those example outputs have already been created. Verification resumes, but solving
requires a new output directory for each experiment. Add `--no-rerun` to `solve`
to skip its automatic `.rrd` export. The workbench can read the normal artifacts.
For all 300 frames, omit `--frame-step`/`--max-frames` and choose a new output.
For LightGlue, point verification at `recordings/osaka-allpairs-lightglue`.

The cosine sample registered 29/30 images (the initial black frame has no features),
produced 1,751 points with 11,046 retained observations, and reached 0.82 px median
and 2.09 px p95 reprojection error. BA converged in 29 evaluations. Seed matching
rows were 210 and 230, with row 220 used for third-view validation. These image-fit
statistics are not ground-truth trajectory accuracy.

## Geometry and initialization

F uses USAC/MAGSAC in pixel coordinates; E uses calibrated five-point RANSAC; H uses
homography RANSAC. The raw masks stay separate from cheirality, triangulation and
final observation filtering. Estimator failures on degenerate data are recorded
and reject that model instead of aborting the whole run.

Verified track edges require both F and E support and minimum pair support. Track
association uses increasing geometric residual and enforces one observation per
image. A planar or low-parallax pair can contribute correspondences while failing
the stricter seed test. Seeds need positive-depth triangulations, reprojection
agreement, at least two degrees of parallax, image coverage and limited homography
dominance. Up to 50 ranked candidates are tested; a third view must pass PnP when
the selection contains more than two images.

After initialization, every registered landmark with an observation in a candidate
image can support that image's PnP. New points are triangulated against a registered
view with a wide estimated baseline, then checked against other registered views.
Final BA jointly refines cameras and points with fixed intrinsics. Final tracks
must normally retain at least three observations with <=3 px error and positive depth.

Intrinsics default to shared fx=fy from 60-degree horizontal FOV, centered cx/cy,
zero distortion. Set `verify-matches --fx ... --fy ... --cx ... --cy ...` in cached
preview pixels when known. The camera model is then fixed throughout solving.
The first seed pose fixes the world frame; its baseline fixes arbitrary scale.
Translation directions from pairwise E are never concatenated as equal-length steps.

## Inspect and iterate

`view-geometry` shows every raw candidate link, with separate tabs for F, E, H,
verified edges and triangulation. Green means accepted by that tab's mask; red
means rejected. Click points for original feature IDs, residuals and parallax.
The matrix contains verified pair counts; -1 denotes missing pairs or self-pairs.
Without `--pair`, it selects the highest scoring seed candidate for inspection.

Geometry is a snapshot of available matches at command start. Repeat verification
after a matching run advances to include newly available pairs and rebuild tracks.
The geometry status distinguishes the available pair count from all possible pairs.
Do not update a geometry run while another process is solving from it.

This prototype still has one-component reconstruction, final-only BA, conservative
greedy track association and no automatic track splitting. Moving people, captions,
wrong focal length and distortion can produce misleading geometric support. Missing
poses are left missing. Global SfM, periodic/local BA and mature solver adapters
remain future backend work; see [SOLVER_DESIGN.md](SOLVER_DESIGN.md).

The tests check synthetic known poses/scale, outlier rejection, pure rotation,
planar/low-parallax rejection, raw mask preservation, immutable source matches,
resume/input invalidation, row-ID mapping, and self-contained Rerun exports.
GUI schemas and APIs are documented in [WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md).
The planned backend integration tasks are in [SOLVER_TODO.md](SOLVER_TODO.md).
For the verified session state and the next LightGlue comparison, see
[SESSION_HANDOFF.md](SESSION_HANDOFF.md) and [WORKFLOW.md](WORKFLOW.md).
