# SLAM Lab

Process monocular RGB videos or image sequences, cache SuperPoint features and
causal online feature tracks or pairwise matches, verify geometry, and reconstruct
camera poses and sparse points. The Three.js workbench and optional Rerun adapters
read the same SQLite/NPZ/JSON artifacts.

Video → features → online tracks
                 ↘ matches → geometry/verified tracks → cameras and points

Start with [WORKFLOW.md](WORKFLOW.md) for commands and the documentation map.
[SESSION_HANDOFF.md](SESSION_HANDOFF.md) records the verified end-of-session state,
results and next work. Both Osaka all-pairs matching runs are complete; the initial
modular solver has been exercised on 30 selected frames from the cosine run.

The new CUDA-first desktop prototype lives in
[native_workbench/](native_workbench/README.md). It uses NVDEC, TensorRT,
CUDA/OpenGL interop, Dear ImGui, and an asynchronous feature database so decode,
display, SuperPoint, and the forthcoming online landmark tracker share one live
frame path without extracted images.

**All-pairs LightGlue matching:** see [MATCHING.md](MATCHING.md) for the Osaka run,
resuming after interruption, progress checks and viewing correspondences in Rerun.
This builds tentative appearance tracks independently of camera reconstruction.

**Modular camera/point solver:** [SOLVER.md](SOLVER.md) describes pairwise RANSAC,
verified tracks and reconstruction from either matcher. The GUI handoff is
[WORKBENCH_INTERFACE.md](WORKBENCH_INTERFACE.md); Rerun is optional presentation.

## Run

The `.venv` environment is installed with CPU PyTorch. CUDA LightGlue uses the
separate `.venv-cuda` environment on the RTX 2080 Ti;
see [MATCHING.md](MATCHING.md). The CPU environment remains available for geometry,
solving and viewers.

```bash
source .venv/bin/activate

slam-lab process /path/to/video.mp4
slam-lab process /path/to/videos --recursive
slam-lab list

# Build descriptor-only landmarks causally from the cached detections.
slam-lab track-online .slam-cache/<run-id>/cache.sqlite3 \
  --output recordings/my-online-tracks --device cuda
slam-lab track-status recordings/my-online-tracks

# Use the cache directory or cache.sqlite3 path printed by process/list.
slam-lab view .slam-cache/<run-id>
```

`track-online` reuses the exact cached SuperPoint detections and processes them in
timestamp order. It stores frame observations, persistent landmark IDs, spherical
descriptor resultants, mean-resultant concentration, and assignment similarities
in `tracks.sqlite3`. Images remain only in the compressed source video; the
track artifact references them by source path, timestamp, and original frame index. It commits
once per frame and resumes an interrupted output.

The viewer provides a video timeline, keypoint overlays colored by confidence,
feature-count plots, inference timing, and recording metadata. You can inspect
individual points and their confidence values. Both frame number and video time
are available as timelines.

To open the already generated and processed 12-frame demo:

```bash
.venv/bin/rerun recordings/demo.rrd
```

That demo is a synthetic smoke test: 12 frames of moving shapes and a textured
background. For real footage, use the downloaded TUM sequences described below.

## Standard test sequences: monocular RGB

Two [TUM benchmark sequences](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
are downloaded under `data/datasets/tum/`:

| Sequence | RGB frames in this archive | Scene |
| --- | ---: | --- |
| `rgbd_dataset_freiburg1_xyz` | 798 | Handheld camera translating around an office desk |
| `rgbd_dataset_freiburg1_desk` | 613 | Handheld camera sweeping across office desks |

**The lab uses only the single RGB camera.** The original archive names include
`rgbd`, but the reader loads only `rgb.txt` and its RGB PNG files. Depth, stereo,
IMU, and ground truth are not inputs to feature extraction or visualization.

```bash
# Process both sequences, preserving their original RGB pixels and timestamps.
slam-lab process data/datasets/tum --device cpu

# Or process one sequence independently.
slam-lab process data/datasets/tum/rgbd_dataset_freiburg1_xyz --device cpu

# Open the prepared detections in Rerun.
rerun recordings/tum-fr1-xyz-rgb.rrd
rerun recordings/tum-fr1-desk-rgb.rrd
```

The image-sequence reader does not transcode the data to a movie. It parses
timestamps as integer nanoseconds, keeps the absolute capture timestamp in `pts`,
and displays time relative to the first RGB frame. Cache identity includes the RGB
manifest and every referenced RGB image; unrelated files do not affect it.

Archive URLs, byte counts, and download checksums are recorded in
`data/datasets/tum/sources.json`. The original archives are in its `archives/`
subdirectory. Source details and checksums are also in [DATASETS.md](DATASETS.md).

To try a short segment of your own video before processing it all:

```bash
slam-lab process /path/to/video.mp4 --max-frames 100
# Continue with the same extraction settings; completed frames are reused.
slam-lab process /path/to/video.mp4
```

Useful extraction options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--stride` | `1` | Process every Nth decoded frame |
| `--max-side` | `1024` | Downscale the longest image edge to at most this many pixels |
| `--max-keypoints` | `2048` | Maximum SuperPoint features per frame |
| `--threshold` | `0.0005` | SuperPoint detection threshold |
| `--device` | `auto` | Choose CUDA, then MPS, then CPU; explicit choices also supported |
| `--max-frames` | unlimited | Limit selected frames per video; resume by omitting this option |
| `--cache-dir` | `.slam-cache` | Directory for feature caches and model weights |

Folder processing is sequential, shares one model across inputs, and continues
after an individual input fails. Any failed input makes the command exit nonzero.
A directory containing `rgb.txt` is treated as one TUM image sequence. A parent
folder discovers videos and immediate TUM sequence subdirectories; `--recursive`
also searches deeper directories.

## View or export without inference

```bash
# Filter the overlay without changing or recomputing cached features.
slam-lab view .slam-cache/<run-id> --min-score 0.01 --point-radius 2

# Export without starting a desktop viewer; useful on a headless machine.
slam-lab view .slam-cache/<run-id> --save recordings/session.rrd
rerun recordings/session.rrd
```

The browser viewer streams the source video and overlays cached detections by timestamp.
Descriptors stay in the cache for future matching. Existing `.rrd` files are never
overwritten by `--save`.

## Install on another machine

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=auto -e '.[superpoint,matching,reconstruction,dev]'
```

Use `--torch-backend=cpu` to explicitly install CPU PyTorch. `--device cuda` needs
a CUDA-capable PyTorch installation and a working NVIDIA driver. To install only
the cache reader and viewer, use `uv pip install -e .`.

SuperPoint comes from a pinned revision of [LightGlue](https://github.com/cvg/LightGlue).
The first extraction downloads its pretrained weights to
`.slam-cache/models/checkpoints/`; subsequent runs work offline. The weights are
checked against a fixed SHA-256 checksum. This workspace already has the weights.

## Cache and coordinate conventions

Each run is stored as `.slam-cache/<run-id>/cache.sqlite3`. The run ID hashes the
video contents, extraction settings, pinned model code and weights, preprocessing
version, cache schema, and relevant dependency versions. Changing those inputs
creates a separate cache. Identical video contents reuse a cache even if renamed.
Device selection is not part of the key; CPU/GPU results may differ slightly.

Each frame contains:

- Its original frame index, presentation timestamp, and time relative to the first frame.
- Original dimensions and resized detector dimensions.
- `float32` keypoints `(N, 2)`, confidence scores `(N,)`, and descriptors `(N, 256)`.
- Inference duration in milliseconds, excluding decoding, model loading, and cache writes.

The arrays are stored as raw little-endian float32 SQLite blobs, without NPZ compression.
No decoded frame or JPEG is stored. Coordinates are `(x, y)` in detector input pixels,
with the origin at the top left.
To map back to the original coded image, accounting for pixel centers:

```python
original_xy = (keypoints + 0.5) * [original_width / width, original_height / height] - 0.5
```

Presentation timestamps preserve variable frame-rate timing. Files without usable
timestamps are rejected instead of estimating times from average FPS. The first
video stream is used; rotation metadata is not applied, so portrait phone footage
may appear sideways, with features still aligned to its coded pixels.

Frames are committed individually in SQLite transactions. An interrupted run can
resume without recomputing completed features. Resuming still decodes earlier
frames sequentially; a complete cache hit only hashes the source content. Copy an
input fully before processing it, and use one processing command per cache at a time.

The cache keeps full-precision descriptors, compressed losslessly. At 2,048
keypoints, descriptors alone occupy about 2 MiB per frame before compression.
Use `--stride` or `--max-keypoints` for longer videos and smaller experiments.

To read features directly for the next stage:

```python
from pathlib import Path
from slam_lab.cache import FrameCache

with FrameCache(Path('.slam-cache/<run-id>/cache.sqlite3')) as cache:
    for frame in cache.frames():
        print(frame.index, frame.keypoints.shape, frame.descriptors.shape)
```

## Development

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts

# Create another small, textured video without downloading footage.
python scripts/make_demo.py --output data/another-demo.mkv --frames 30
slam-lab process data/another-demo.mkv --device cpu
```

Tests cover variable frame-rate timing, resizing, interrupted and partial resume,
cache invalidation, empty-feature frames, folder failure isolation, TUM RGB pixel
and timestamp preservation, and headless Rerun export. The initial end-to-end check
also ran real SuperPoint on the demo,
checked descriptor normalization and point bounds, and verified the resulting RRD
with `rerun rrd verify`.

## Reconstruct from saved matches

Use the modular path for the current matcher-independent solver:

```bash
.venv/bin/slam-lab verify-matches recordings/osaka-allpairs-lightglue \
  --output recordings/my-lightglue-geometry --frame-step 10 --max-frames 30
.venv/bin/slam-lab solve recordings/my-lightglue-geometry \
  --output recordings/my-lightglue-solution --no-rerun
```

See [SOLVER.md](SOLVER.md) for solver limitations. Matching, geometry and reconstruction
artifacts use mandatory manifests and strict parent validation; reconstructions export
stable selection mappings and structured solver status.
