# SLAM Native Workbench

The native workbench decodes video with NVDEC, runs TensorRT SuperPoint on
uncached frames, and computes NVIDIA optical flow between every consecutive
pair. Cached frames supply the same SuperPoint positions and descriptors.
Images stay on the GPU and the source video remains the only image store.

## Tracking

SuperPoint runs every frame. NVIDIA optical flow predicts where each live
track's point moves, and the track snaps to a nearby SuperPoint detection
with a similar descriptor. The whole tracking path runs on the GPU:
detections, flow, descriptors and track state stay in device memory. Each
frame, the host receives only compact per-track records for display and
statistics. See [ARCHITECTURE.md](ARCHITECTURE.md#tracking-and-keypoint-decisions--2026-09-24)
for the rules, the diagnosis of the earlier tracker, and measurements.

- Existing tracks are associated against all current detections. The gates
  are a radius around the flow prediction and descriptor cosine similarity.
  Assignment is one-to-one, with several propose/accept rounds so a track
  that loses its first choice can take its next one.
- A track with no supporting detection coasts on flow for up to 5 frames.
  It is drawn hollow and keeps its ID if a detection reacquires it.
- Under the cap, matched continuations come first, then coasting tracks, then
  the strongest unmatched detections. The default cap is 1,000 live tracks;
  all raw detections are still cached.
- Missing flow or a temporal/resolution discontinuity breaks identity.

Defaults and CLI controls:

- `--track-radius 6`: association radius in source-image pixels (radius 8
  measured longer tracks; see ARCHITECTURE.md).
- `--max-tracks 1000`: maximum live tracks.
- `--max-coast 5`: flow-only frames for a missed track; `0` makes unmatched
  tracks die immediately.
- `--min-similarity 0.7`: descriptor cosine gate.
- `--assignment-rounds 4`: `1` gives the old single-pass behaviour.
- `--flow-backward PX`: also compute backward flow. Tracks whose
  forward-backward error exceeds PX cannot coast. This costs a second NVOF
  pass and is off by default.

The Pipeline panel's **Association radius** control adjusts the radius live
(0.5–100 source-image pixels). Changes apply to the next processed frame and
survive seeking or restarting within the session. Use Restart to compare a
complete run at a different threshold.

## Camera pose and trajectory

The workbench estimates the camera pose live from the observed (non-coasted)
tracks: monocular visual odometry with keyframes and sliding-window bundle
adjustment. The **Camera trajectory** window shows:
- the path, the current camera frustum and the map points;
- orbit, top, side and front views.

Mouse controls: wheel zooms, left-drag orbits, right-drag pans, and
double-click fits the view. Scale is arbitrary. After tracking is lost a new
segment starts with its own scale. Seeking resets the trajectory.

- `--hfov 60`: assumed horizontal field of view (the default guess).
- `--intrinsics FX FY CX CY`: known intrinsics in source-video pixels.

The solver is independent of the tracker. To get poses from other tracks,
write them as CSV and run the headless tool:

```text
# size 1920 1080
frame_index,timestamp_ns,track_id,x,y
0,0,17,812.5,433.25
...
```

```bash
native_workbench/build/app-debug/slam-native-vo-tracks --tracks tracks.csv \
  --output trajectory.csv [--hfov 60 | --intrinsics FX FY CX CY] [--verbose]
```

The output CSV has one row per posed frame: `frame_index`, `timestamp_ns`,
`segment`, `keyframe`, the camera centre (`cx,cy,cz`) and the
camera-to-world quaternion (`qw,qx,qy,qz`). In C++, build `TrackedFrame`s
and call `VisualOdometry::process()` directly. To export the native tracks,
use `slam-native-tracking-bench VIDEO ENGINE START FRAMES export-tracks
tracks.csv export-trajectory trajectory.csv`. See
[ARCHITECTURE.md](ARCHITECTURE.md#camera-pose-live-monocular-visual-odometry--2026-09-24)
for the algorithm, measurements and limitations.

## ORB-SLAM3 baseline (separate program)

ORB-SLAM3 monocular is available as an external baseline to compare with the
native tracker/VO. It is a separate GPLv3 program; nothing is linked into the
workbench. It is built user-locally (no sudo) into the git-ignored
`third_party/` at the repository root:

- `third_party/ORB_SLAM3`: [ORB_SLAM3_U24](https://github.com/vijaysaini-ra/ORB_SLAM3_U24)
  at commit `c500153cec1825150e7abf07a8291da33499ecd0` (ORB-SLAM3 v1.0 patched
  for Ubuntu 24.04+).
- `third_party/orbslam3-deps`:
  - OpenCV 4.10.0 (minimal modules), Eigen 3.4.0 and Pangolin v0.9.4, built
    from source.
  - Boost serialization, OpenSSL, libepoxy and EGL/GLVND headers, extracted
    from Ubuntu packages with `apt-get download`.
  - The driver, `bin/orbslam3_run`.

```bash
scripts/orbslam3/setup.sh     # ~5 min on 16 cores; rerunnable
scripts/run_orbslam3.sh /home/nick/Downloads/osaka.mp4 120 30 out/orbslam3-osaka-120 [WIDTH=640] [HFOV=60] [FEATURES=1000]
```

To watch it live, set `ORBSLAM3_VIEWER=1`. This opens ORB-SLAM3's Pangolin
viewer (map, keyframes, and the current frame with ORB matches as an inset),
paces playback to real time and holds the final map for 30 s
(`ORBSLAM3_HOLD=SECONDS` to change). The frame inset comes from
`scripts/orbslam3/viewer-frame-in-pangolin.patch`, which `setup.sh` applies,
because the local OpenCV build has no GUI backend.

The run script:
- Streams the clip through ffmpeg as grayscale frames (640 px wide by
  default, source frame indices preserved).
- Writes a pinhole `camera.yaml`: 60° HFOV guess, zero distortion.
- Runs ORB-SLAM3 without its viewer.
- Saves `orbslam3-live.csv` (per-frame tracked poses) and
  `orbslam3-final.csv` (largest map after bundle adjustment and loop closure)
  in the same trajectory CSV format as `slam-native-vo-tracks`, plus
  ORB-SLAM3's own EuRoC-format files.

First Osaka runs (30 s clips at 640×360):

| Clip | Initialized after | Tracked frames | Maps / keyframes | Losses | Time per frame |
| --- | --- | --- | --- | --- | --- |
| 120 s | 131 frames (2.2 s) | 1668 / 1799 | 1 map, 130 keyframes | 0 | ≈6 ms |
| 240 s | — | 1749 / 1799 | 1 map, 99 keyframes | 0 | ≈6 ms |

## Persistence and replay

Schema 3 caches only raw SuperPoint observations: frame identity, timing,
dimensions, coordinates, scores, descriptors, and extraction timings.
Descriptors default to float16; use `--descriptor-storage f32` for lossless
float32 storage. A bounded asynchronous writer batches SQLite transactions.

Tracks, flow fields, associations, correction diagnostics, and
future reconstruction state are in memory only. The cache rejects frames
containing derived tracking state. Reopening a session replays from frame zero,
reusing detections and recomputing flow and tracks with current settings.

Cache identity includes the canonical source path, source size and modification
time, engine path/size/modification time/content fingerprint, extraction version,
input dimensions, keypoint limit, score threshold, and descriptor encoding.
Changing these rejects reuse; choose a fresh cache path. The source identity is
not a full video content hash. Tracking settings are deliberately excluded.

Default caches live at
`recordings/native-<video-name>-superpoint-cache/features.sqlite3`.
Legacy databases are rejected before modification. Selecting an old recent
session automatically proposes the new default path; existing data is retained.

Descriptors can still be large: 2,048 points at 30 FPS uses about 30 MiB/s for
float16 descriptors alone. This cache is reusable feature data, not a recording
of tracking experiments.

## Viewer

Launch without arguments to choose video, TensorRT engine, and cache paths.
Both SuperPoint and optical flow are always enabled.
On Linux, startup defaults OpenGL to NVIDIA render offload before initializing
GLFW, so a plain launch also works when the desktop uses an integrated GPU.
Explicit environment overrides are preserved. A startup check verifies that
OpenGL can share buffers with the selected CUDA device before loading a video.

Run, pause, step, stop, and realtime pacing control sequential processing. Choose another video to
flush the feature cache and return to the launcher.

Filled points are current SuperPoint detections; hollow points are tracks
coasting on flow this frame. Click a point to
inspect its track. Diagnostics show SuperPoint correction distances and track lengths.
Show flow vectors overlays sampled NVIDIA motion vectors. It is the only
feature that copies flow to the host, and it applies from the next frame.

Select **Point view → SuperPoint detections (red)** to inspect all retained
raw detections before the 1,000-track cap. Every detection is red; track trails
and flow arrows are hidden in this view. Tracking continues in the background,
and the view can be switched while paused.

Dragging the scrubber seeks playback and pauses at the selected frame. Tracking,
flow history, trails and selection reset there; that frame seeds new tracks.
Play continues from that position, and Step advances one frame. Restart returns
to the beginning, resets tracking, and starts playback, including after Stop or
EOF. Cached SuperPoints are reused; uncached seek destinations run inference.
Cache row IDs are independent of decoder frame ordinals after seeking; source
timestamps and PTS identify cached observations.

The reasoning behind the current keypoint and association choices is recorded
in [Tracking and keypoint decisions](ARCHITECTURE.md#tracking-and-keypoint-decisions--2026-09-23).

## Model and build

The TensorRT engine uses fixed input and top-k shapes. Inference letterboxes
the image and converts detections back into source pixels.

```bash
.venv-cuda/bin/python native_workbench/tools/export_superpoint_onnx.py \
  native_workbench/models/superpoint-1024x576-k2048.onnx
trtexec --onnx=native_workbench/models/superpoint-1024x576-k2048.onnx \
  --saveEngine=native_workbench/models/superpoint-1024x576-k2048.engine --skipInference
```

Development dependencies: CMake, C++20, CUDA, TensorRT, FFmpeg, SQLite, GLFW,
and OpenGL. CMake fetches pinned Dear ImGui and Eigen 3.4 (header-only). The reviewed Ubuntu installer is
`tools/install_ubuntu2604_system.sh`. Workspace VS Code tasks and CMake presets
support core-debug, app-debug, and app-release.

```bash
cd native_workbench
../.venv-cuda/bin/cmake --preset app-debug
../.venv-cuda/bin/cmake --build --preset app-debug
ctest --test-dir build/app-debug --output-on-failure
build/app-debug/slam-native-workbench
```

For a direct launch from the repository root:

```bash
native_workbench/build/app-debug/slam-native-workbench \
  --video /path/to/video.mp4 \
  --engine native_workbench/models/superpoint-1024x576-k2048.engine \
  --db recordings/native-experiment-superpoint-cache/features.sqlite3
```

Optional GPU integration test (90 frames, then replay with cached detections):

```bash
native_workbench/build/app-debug/slam-native-hybrid-test \
  /path/to/video.mp4 native_workbench/models/superpoint-1024x576-k2048.engine
```

The optical-flow runtime comes with the NVIDIA driver; its vendored API headers
and license are in `third_party/nvof/`. The native cache is independent of the
Python offline matching/solver artifacts, whose behavior is unchanged.

GPU tracker checks (`slam-native-gpu-tracker-test`) and synthetic optical-flow
checks run under CTest in the app build. With a video,
`build/app-debug/slam-native-optical-flow-test VIDEO` also checks streaming
repeatability. For headless tracking metrics on real video:

```bash
native_workbench/build/app-debug/slam-native-tracking-bench \
  /path/to/video.mp4 native_workbench/models/superpoint-1024x576-k2048.engine 120 600
```
