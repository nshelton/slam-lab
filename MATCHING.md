# Exhaustive appearance matching

LightGlue is connected to the cached SuperPoint keypoints and 256-dimensional
descriptors. It does not run SuperPoint again. This stage compares every unordered
pair of selected images and then groups their features into tentative global tracks.
It is separate from the earlier incremental reconstruction pipeline.

## Completed Osaka results — 2026-09-20

Both 300-frame runs are complete, including appearance-track association and their
`correspondences.rrd` exports. Neither matching job is running at session handoff.

| Run | Pairs | Match edges | Tracks with >=2 observations | Longest track |
| --- | ---: | ---: | ---: | ---: |
| `osaka-allpairs-cosine` | 44,850 | 12,471,433 | 28,778 | 285 |
| `osaka-allpairs-lightglue` | 44,850 | 22,111,491 | 15,641 | 296 |

Track counts describe appearance association, not reconstruction accuracy. LightGlue
retained 2,807 CPU-computed pairs and added 42,043 CUDA-computed pairs. The GPU resume
ran from 20:30:21 to 20:45:24 UTC through track association (about 15 minutes); its
last reported matching rate was 47.8 pairs/s. The CUDA and CPU pairs have separate
runtime provenance. See [SESSION_HANDOFF.md](SESSION_HANDOFF.md) for the result inventory.

## Osaka: start or resume

From `/home/nick/slam-lab`:

```bash
.venv-cuda/bin/slam-lab match-all \
  .slam-cache/7065fae5d5e9bd54ae4e14d2d5f051960dfe2d14266d89d5e6771fa037786216 \
  --output recordings/osaka-allpairs-lightglue \
  --matcher lightglue --device cuda --workers 1 --max-frames 300 \
  --allow-runtime-change \
  --background
```

This uses **all 300 cached frames** from the first 30 seconds and all cached features
(up to 2,048 per frame): **44,850 pairs**. Each pair is computed once; its reverse is
the same correspondence with the indices swapped. No self-matches are computed.
The first frame is black and has no detections; its pairs are recorded as empty.
Near-time pairs run first, followed by progressively larger frame gaps.

`--background` detaches the job so closing the terminal does not stop it. Omit this
option for a foreground progress bar. Rerunning the same command resumes missing
pairs. A process lock prevents two jobs from writing to the same run. Changing the
selected frames, matcher or matching parameters requires a different output directory.
Worker count can change on resume. CUDA requires one worker to bound GPU memory.

The separate `.venv-cuda` environment has PyTorch 2.11.0+cu128 and uses the installed
RTX 2080 Ti (11 GB). A warmed-up sample measured **19–41 ms/pair**; the actual resumed
run sustained roughly **45–48 pairs/s**. For future runs, follow the live ETA below; track association
and Rerun export happen after matching. GPU matching requires the host GPU to be
visible to the process (it is hidden inside Codex's filesystem/process sandbox).

`--allow-runtime-change` explicitly permits switching only the LightGlue device or
PyTorch runtime. It keeps all previously saved matches and records runtime provenance
per pair. The Osaka resume preserved 2,807 CPU pairs; the remaining 42,043 used CUDA. Weights,
matcher settings and feature selection must still agree. CPU/GPU numerics can differ.
The ordinary `.venv` remains unchanged; use `.venv/bin/slam-lab` for status/viewers.

For comparison, an eight-pair FP32 CPU benchmark estimated roughly **6.4 hours** on this
machine with four workers. Treat that as an estimate: pair difficulty varies, and
global track association/export takes additional time. The status command reports
the measured rate and matching ETA as the actual run progresses.

```bash
.venv/bin/slam-lab match-status recordings/osaka-allpairs-lightglue
tail -f recordings/osaka-allpairs-lightglue.log
```

`running` checks the process lock; `phase` is the last saved processing phase.
If the machine restarts or the job fails, saved counts remain valid, but its old
rate/ETA are stale until you resume. Progress metadata updates every 15 seconds;
completed pair counts come directly from SQLite.

## View the correspondences

Pairs become available while the job runs. Selected-frame rows 100 and 101 are at
approximately 10.01 and 10.11 seconds in this Osaka cache:

```bash
.venv/bin/slam-lab view-matches recordings/osaka-allpairs-lightglue --pair 100 101
```

Rerun opens a **Match coverage** tab: green points have an appearance match and red
points are unmatched detections. Regions without points have no cached SuperPoint
detections. These colors describe matching, not RANSAC inliers/outliers.
The **Correspondences** tab shows both images, matched points with shared colors
and IDs, match confidence, descriptor distance, and all connecting lines by default.
Use `--max-links 150` to display only the strongest 150 lines or `--max-links 0`
to hide lines. The tab title reports displayed/total lines. Limiting lines never
filters the coverage overlay or matched points. Gray points in the correspondence
tab are unmatched. Both tabs share a symmetric N × N match-count heatmap.
Heatmap gray means a pair is pending, black means processed with zero matches,
and brightness encodes log match count. The diagonal is pale and excluded.
Rows and columns refer to the selected-frame rows, starting from zero.

The viewer is a snapshot: reopen it to refresh a running job. Without `--pair`, it
shows a recently saved pair while processing, or all global tracks after completion.
Track colors remain stable across the video timeline. Click a point for its global
track ID and observation count; `--min-track-length` only changes the display.

To save a self-contained recording without opening a GUI:

```bash
.venv/bin/slam-lab view-matches recordings/osaka-allpairs-lightglue --pair 100 101 \
  --save recordings/osaka-lightglue-10s.rrd
.venv/bin/rerun recordings/osaka-lightglue-10s.rrd
```

When matching finishes, the command automatically associates tracks and exports
`recordings/osaka-allpairs-lightglue/correspondences.rrd`. Open that file in Rerun to
scrub through the entire selected video with stable track colors. `--no-rerun`
skips this optional export for workbench jobs.

## Exhaustive cosine baseline

```bash
.venv/bin/slam-lab match-all \
  .slam-cache/7065fae5d5e9bd54ae4e14d2d5f051960dfe2d14266d89d5e6771fa037786216 \
  --output recordings/osaka-allpairs-cosine \
  --matcher cosine --cosine-threshold 0.8 --workers 4 --max-frames 300 --background

.venv/bin/slam-lab match-status recordings/osaka-allpairs-cosine
.venv/bin/slam-lab view-matches recordings/osaka-allpairs-cosine --pair 100 101
```

This run computes the complete cosine-similarity matrix for every image pair,
using all cached descriptors. Descriptors are normalized per vector. It retains
unique mutual best matches with cosine similarity >= 0.8; zero-norm descriptors
and exact best-score ties are rejected. There is no ratio test or geometric filter.
Stored confidence is the cosine similarity itself, not a LightGlue confidence.
The full dense matrices are computed and discarded; accepted correspondences and
their scores are saved. `--cosine-threshold` changes the acceptance threshold and
requires a different output directory for previously started runs.

For unit-normalized vectors, squared L2 distance is `2 - 2 * cosine_similarity`,
so cosine and L2 rank neighbors equivalently. This baseline differs from the `nn`
mode in its acceptance rule and stored score: `nn` applies two-sided ratio tests.

A 32-pair CPU benchmark measured 22.6 pairs/s with one worker and 66.6 pairs/s
with four workers (about 11.2 minutes of matching for all 44,850 pairs). Disk writes,
association and export add time. Follow the actual run's measured ETA in `match-status`.

Historically, LightGlue was paused at 2,578 pairs while this baseline ran, then
resumed on CPU to 2,807 pairs before the CUDA migration. Both runs are now complete.
`--resume-after <run>` remains available for future job chains: it starts a saved
run after matching/export succeeds and reuses completed pairs. The child inherits
the launching Python environment. To resume a CUDA run, launch the chain from
`.venv-cuda`; the CPU environment cannot execute CUDA merely because the saved
matcher requests it. Independent explicit commands are clearer for cross-runtime work.

## Outputs and interpretation

| File | Contents |
| --- | --- |
| `matches.sqlite3` | Every pair's feature indices, matcher-specific scores, L2 distances, timing, original image previews and keypoints, and matcher/runtime provenance |
| `tracks.npz` | Per-observation track IDs, frame offsets, track lengths, original frame indices/timestamps, and symmetric pair match counts |
| `summary.json` | Coverage, track counts, rejected association conflicts and matcher settings |
| `correspondences.rrd` | Automatically exported global-track viewer |

In `tracks.npz`, `track_ids[offsets[i]:offsets[i+1]]` gives the track for each
feature in selected-frame row `i`, in the original cached keypoint order.
`track_lengths[track_id]` gives its number of observations. Singletons are retained
in the arrays and hidden by the viewer's default minimum track length of two.

Track association processes edges in descending match-confidence order. It merges
two tracks only if their frame sets do not overlap: a track cannot contain two
different detections from the same image. Consistent cycle edges are counted;
conflicting edges are counted and excluded from association. Raw pair matches
are preserved unchanged in SQLite. Ties follow canonical pair/keypoint order,
independent of worker completion order.

These are **appearance associations**, not geometrically verified landmarks.
Moving people, repeated structures, and fixed video captions can match. Even
frame-consistent tracks may join different physical points. This baseline does
not run RANSAC, estimate intrinsics or poses, or triangulate. Those checks belong
to the next stage and should use the retained pair evidence.

LightGlue uses its pretrained SuperPoint weights, nine layers, standard adaptive
depth/width thresholds (0.95/0.99), and match threshold 0.1. CPU inference uses FP32;
the upstream CUDA attention path casts attention inputs to FP16 and dispatches
through PyTorch's available attention implementation. Do not describe this GPU
run as entirely FP32 or assume a particular FlashAttention kernel. The downloaded
weights' SHA-256 and the pinned implementation revision are recorded in the run.
`--threshold` changes LightGlue's acceptance threshold. `--matcher nn --ratio 0.8`
selects the simpler mutual-nearest-neighbor baseline with bidirectional L2 ratio
tests; use a separate output directory. Scores have different meanings across
the two matchers.

On a new machine install `.[superpoint,matching]` (or add `matching` to the full
installation). Model weights are cached under `.slam-cache/models/checkpoints/`.
Use `--workers 1 --device cuda` for a CUDA installation. Defaults here are CPU,
four workers and one numerical-library thread per worker.
