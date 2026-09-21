

points (superpoint)

## matcher
cosine  ( 66.6 pairs/s
with four workers )
 lightglue (41 pairs/sec)

track building (why do we need tracks ?)


----

select pairs / pair verification modules
Good F and good E 



Experiment to run: 
use lightglue matches
use cosine matches, do a solve, and then use the matches to register. then use the 3d structure to see if we can scoop up more matches. How would this work. with registered cameras, we would exhaustively select all unmatched points. they are just rays at this point. 
# Working with SLAM Lab

Run from `/home/nick/slam-lab`. The lab uses monocular RGB and fixed pinhole
intrinsics. SQLite/NPZ/JSON files connect the stages; Rerun is a viewer/export adapter.

## Read first

| Document | Purpose |
| --- | --- |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | Session state, completed results, decisions and next tasks |
| [README.md](README.md) | Installation, input processing, feature caches and overview |
| [MATCHING.md](MATCHING.md) | Cosine/LightGlue matching, CUDA, resume and pair inspection |
| [SOLVER.md](SOLVER.md) | Implemented geometry/track/solver pipeline and Osaka experiment |
| [WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md) | Current GUI interface plus clearly labeled proposed additions |
| [SOLVER_TODO.md](SOLVER_TODO.md) | Prioritized backend tasks and acceptance criteria |
| [SOLVER_DESIGN.md](SOLVER_DESIGN.md) | Architecture and future numerical-solver work |
| [RECONSTRUCTION.md](RECONSTRUCTION.md) | Earlier tracker and its trajectory/RANSAC investigation |
| [DATASETS.md](DATASETS.md) | Downloaded benchmark sources and RGB-only interpretation |
| [reference.md](reference.md) | Research notes; not an implemented API or algorithm guarantee |

## Inspect the completed work

No inference or rematching is needed for these commands:

```bash
.venv/bin/slam-lab match-status recordings/osaka-allpairs-lightglue
.venv/bin/slam-lab match-status recordings/osaka-allpairs-cosine
.venv/bin/slam-lab geometry-status recordings/osaka-cosine-geometry-step10

# Appearance correspondences around 10 seconds; all links shown by default.
.venv/bin/slam-lab view-matches recordings/osaka-allpairs-lightglue --pair 100 101

# Saved RANSAC masks on the every-tenth-frame geometry selection.
.venv/bin/slam-lab view-geometry recordings/osaka-cosine-geometry-step10 --pair 100 110

# Solved camera poses, points, residuals and PnP diagnostics.
.venv/bin/slam-lab view-reconstruction recordings/osaka-cosine-solver-v1
```

For a pre-exported recording, use the viewer installed in the environment:

```bash
.venv/bin/rerun recordings/osaka-allpairs-lightglue/correspondences.rrd
.venv/bin/rerun recordings/osaka-cosine-solver-v1/reconstruction.rrd
```

`slam-lab view-*` locates the environment's viewer even when `.venv/bin` is not on
PATH. For headless work, use `--save <new-file.rrd>` on viewer commands instead of
launching a window. Existing exports are not overwritten.

## Completed LightGlue solver comparison

The comparison reuses completed matches and selects rows 0, 10, ..., 290 across the
first 30 seconds, just like the cosine test. The current strict result is
`recordings/osaka-lightglue-solver-v5`.

```bash
.venv/bin/slam-lab solve-status recordings/osaka-lightglue-solver-v5
.venv/bin/slam-lab view-reconstruction recordings/osaka-lightglue-solver-v5
```

To reproduce it, run `verify-matches` and `solve` with new output directories.
Intrinsics/geometry thresholds belong to `verify-matches`; solver filtering/BA settings
belong to `solve`. Neither stage runs a matcher.

## New input and matching

```bash
.venv/bin/slam-lab process /path/to/video.mp4 --stride 6 --max-frames 300 --device cpu
.venv/bin/slam-lab list

# Replace <cache-id> with the feature-cache directory reported above.
.venv-cuda/bin/slam-lab match-all .slam-cache/<cache-id> \
  --output recordings/new-lightglue-run --matcher lightglue \
  --device cuda --workers 1 --max-frames 300 --no-rerun --background
```

Stride 6 gives about 10 fps only for a roughly 60 fps source. Feature extraction can
resume with a larger limit. Changing extraction settings or relevant package versions
may select a new feature cache; existing matches use their recorded cache directly.

Use `.venv-cuda` for CUDA matching and `.venv` for CPU work and viewers. A sandbox
may hide the GPU even when the host driver works. `--allow-runtime-change` is needed
only when deliberately resuming existing LightGlue pairs under another device or
PyTorch runtime; it preserves the old pairs and tags new runtime provenance.

## Next backend implementation

The supported backend workflow is exactly `match-all -> verify-matches -> solve`.
Completed matching, geometry and reconstruction directories carry mandatory hashed
`manifest.json` envelopes. Geometry validates its matching parent; reconstruction
validates its geometry parent. Missing manifests, changed payloads and mismatched parent
IDs are rejected instead of routed through a compatibility adapter.

New reconstructions contain stable observation/track/point mappings. Solver jobs publish
atomic structured status readable with `solve-status` or `ArtifactReader.status()`.

```bash
.venv/bin/slam-lab solve-status recordings/osaka-lightglue-solver-v5
```

Matching is the root artifact because it embeds selected frame identities, timestamps,
pixel coordinates and feature indices consumed by every downstream stage. Source pixels
remain in the video. Feature-cache publication is deliberately
outside this solver interface. Full solve invocation/options/runtime provenance and the
single manifested `ArtifactReader` are implemented. Continue with live snapshots, draft
pagination and crash reconciliation in [SOLVER_TODO.md](SOLVER_TODO.md).

After code changes, run appropriate tests and the formatting/lint checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

The last solver implementation validation passed 60 tests. No test rerun is needed
merely to read the artifacts or edit prose documentation.
