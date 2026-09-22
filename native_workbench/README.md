# SLAM Native Workbench

This is the native, CUDA-first visualizer. Its first pipeline is deliberately
small:

```text
compressed video
  -> FFmpeg demux + NVDEC
  -> CUDA YUV frame
       -> CUDA/OpenGL display texture
       -> CUDA resize/grayscale -> TensorRT SuperPoint
            -> in-memory feature batch -> CUDA cosine search + online landmark tracker
            -> asynchronous SQLite writer
       -> NVIDIA optical-flow engine (alternative mode)
            -> grid-seeded motion tracks -> asynchronous SQLite writer
```

The decoded image never enters host memory and is never written to disk. The
source video remains the sole image source. Frame identity, timing, keypoints,
and track IDs are persisted; SuperPoint mode also stores descriptors.

The implementation currently includes:

- an ImGui/GLFW/OpenGL application with run, pause, step, stop, real-time
  pacing, timing diagnostics, and a keypoint overlay;
- a startup panel with an in-app file browser for the video, engine, and
  database; it shows database disk size, frame and landmark rows, and saved
  observations before opening or resuming a session;
- a video scrubber beneath the image; seeking shows a paused preview and loads
  saved keypoints and landmark IDs when that frame is in the database;
- FFmpeg hardware decoding that requires `AV_PIX_FMT_CUDA` and exposes the
  NV12/P010 device planes directly;
- CUDA/OpenGL interop for the video texture;
- a TensorRT SuperPoint runtime with CUDA luma preprocessing, fixed top-k
  outputs, and source-pixel coordinate recovery;
- a descriptor-only online tracker that searches active spherical means with
  cuBLAS, assigns at most one observation per landmark in each frame, and
  shows persistent colors, short trails, and selected-landmark diagnostics;
- an independent NVIDIA optical-flow mode that feeds NVDEC luma to the driver's
  CUDA optical-flow API, seeds a regular grid, advects persistent track IDs from
  4x4 motion vectors, keeps the longest-lived track per grid cell, and reseeds
  newly exposed cells within a 2,000-track budget; it requires no TensorRT
  engine or SuperPoint descriptors;
- a diagnostic window with live descriptor PCA, a selected track's recent
  descriptor path, total and active track counts, and a track-length histogram;
- a bounded asynchronous SQLite writer using WAL and batched transactions;
- an ONNX exporter for the exact LightGlue SuperPoint weights already used by
  the Python pipeline.

## Storage

Descriptors stay float32 in memory so the online tracker consumes
the inference result directly. SQLite storage defaults to float16. This reduces
descriptor traffic from about 2 MiB to 1 MiB per frame at 2,048 keypoints.
Use `--descriptor-storage f32` when bit-for-bit float32 persistence matters.
Optical-flow sessions store `descriptor_encoding=none` and use a separate
default database under `recordings/native-<video-name>-tracks-flow/`. The
`tracker_algorithm` metadata prevents the two methods from sharing a database.
When reopening the same database with the same model and settings, the app
decodes through already committed frames and resumes inference after the last
stored frame. The initial prototype does not seek to that frame yet; long-run
resume and dataset-scale processing require an indexed keyframe seek.

At 2,048 keypoints and 30 FPS, float16 descriptors alone are 30 MiB/s,
or about 105 GiB/hour. Before multi-hour benchmark ingestion, the store should rotate
at a configured frame or byte count into immutable SQLite segments with a small
manifest database. The writer interface is already isolated so that change
does not affect decode, inference, or tracking.

## Model export

The engine has fixed input and top-k shapes. The app letterboxes each video
frame on the GPU and maps detected coordinates back to source pixels.

```bash
.venv-cuda/bin/python native_workbench/tools/export_superpoint_onnx.py \
  native_workbench/models/superpoint-1024x576-k2048.onnx

trtexec \
  --onnx=native_workbench/models/superpoint-1024x576-k2048.onnx \
  --saveEngine=native_workbench/models/superpoint-1024x576-k2048.engine \
  --skipInference
```

The exporter requires `onnx` and `onnxscript` in addition to the existing CUDA
Python environment. TensorRT plans are tied to the installed TensorRT and GPU
environment; build the engine on the machine that runs the workbench.
The exporter emits one self-contained ONNX file. CUDA luma normalization differs
slightly from the existing Python RGB preprocessing; compare detections on real
video before treating native and Python caches as numerically equivalent.

## Build

Required development packages are CMake, a C++20 compiler, CUDA Toolkit,
TensorRT, FFmpeg (`libavcodec`, `libavformat`, `libavutil`), SQLite, GLFW, and
OpenGL. CMake fetches one pinned Dear ImGui release.

Open the repository root in VS Code. The workspace already recommends the
CMake Tools and C++ extensions and points CMake Tools at this source directory.
Run **Terminal → Run Build Task → Native: build core** (or `Ctrl+Shift+B`) to
compile the feature store with the workspace-local compiler. This is the
working build on the current machine. In the CMake sidebar, select the
`core-debug` configure preset to browse its C++ targets.

After installing the native CUDA Toolkit, TensorRT C++ SDK, and system
development packages, select `app-debug` or `app-release` in CMake Tools, or
run the matching **Native: build app** task. The Debug task is also attached to
the **Native workbench (CUDA)** launch configuration. It opens the in-app
session chooser. The launch configuration additionally needs `gdb`.
**Native: export SuperPoint ONNX** and **Native: build SuperPoint
engine** are also VS Code tasks; the latter requires `trtexec` and a working
GPU.

On this Ubuntu 26.04 x86_64 machine, run the reviewed installer in a VS Code
terminal so `sudo` can request your password, or select the **Native: install
system dependencies** task from **Terminal → Run Task**:

```bash
bash native_workbench/tools/install_ubuntu2604_system.sh
```

It installs the matched CUDA 13.4/TensorRT 11.3 C++ packages and the other
development headers globally. It does not request a driver upgrade; the
installed 595-series driver meets CUDA 13.x's minimum version. The app presets
target the RTX 2080 Ti's `sm_75` architecture.

The same presets work from a terminal:

```bash
cd native_workbench
../.venv-cuda/bin/cmake --preset core-debug
../.venv-cuda/bin/cmake --build --preset core-debug

# Once the native dependencies are installed:
../.venv-cuda/bin/cmake --preset app-release
../.venv-cuda/bin/cmake --build --preset app-release
```

```bash
# Standard CMake invocation without presets is also supported once all native
# dependencies are installed.
cmake -S native_workbench -B native_workbench/build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build native_workbench/build/manual -j
```

Run it from the build directory without arguments:

```bash
cd native_workbench/build/app-debug
./slam-native-workbench
```

Choose a video with **Browse**, review the default engine and database paths,
then select **Start session**. The default database goes under
`recordings/native-<video-name>-tracks/features.sqlite3`; you can type another
path or browse to an existing database. **Choose another video** returns to
the launch panel after the writer flushes. The panel and pipeline sidebar show
the main database plus WAL size, committed frame rows, landmark rows, and saved
feature observations. The **Recent videos** dropdown keeps the last ten
successfully opened videos across launches and restores each video's engine and
database paths. The list is stored at
`~/.config/slam-native-workbench/recent_videos.txt` (or under `XDG_CONFIG_HOME`).
Choose **NVIDIA optical flow** in the **Tracking method** dropdown to run
without an engine. The viewer shows flow track IDs and trails; its diagnostics
window shows active/total tracks and the track-length histogram. **Show flow
vectors** overlays sampled motion arrows from the NVIDIA engine. This first
flow baseline seeds a spatial grid and does not yet reject occlusions or
low-confidence motion, so long tracks may drift.

Drag the time bar below the video to preview another part of the source clip.
This pauses processing. **Run** or **Step** returns to the sequential processing
position and continues building the same tracks. A separate NVDEC decoder
handles preview seeks, so browsing never rewinds the online tracker or changes
frame ordinals in the feature database. Frames that have not been processed yet
show video only; stored frames show their saved feature overlay. During preview,
the cyan tick on the bar marks the processing position.

The **Landmark diagnostics** window projects the current 256-dimensional
descriptors into two PCA coordinates. It fits a fixed basis to the first 2,048
descriptors so the axes do not move during playback. Click a projected point
to select its track; the white path shows that track's recent descriptor
positions. The histogram uses exact bins for lengths 1-16 and doubling ranges
after that, with logarithmic bar heights. Hover a bar for its exact count.
PCA is a diagnostic projection; the tracker still compares descriptors in all
256 dimensions.

For scripted launches, the three paths can still be supplied together:

```bash
native_workbench/build/app-release/slam-native-workbench \
  --video /path/to/video.mp4 \
  --engine native_workbench/models/superpoint-1024x576-k2048.engine \
  --db recordings/native-osaka-tracks/features.sqlite3
```

For optical flow, omit the engine:

```bash
native_workbench/build/app-release/slam-native-workbench \
  --tracker optical-flow \
  --video /path/to/video.mp4 \
  --db recordings/native-osaka-tracks-flow/features.sqlite3
```

The runtime optical-flow library comes with the NVIDIA driver. The vendored
API headers in `third_party/nvof/` are from
[NVIDIA/NVIDIAOpticalFlowSDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK);
their redistribution license is included in the headers.

If your terminal is already in `native_workbench/build/app-debug`, use the
engine relative to that directory:

```bash
./slam-native-workbench \
  --video /home/nick/Downloads/osaka.mp4 \
  --engine ../../models/superpoint-1024x576-k2048.engine \
  --db /home/nick/slam-lab/recordings/native-osaka-tracks/features.sqlite3
```

The tracker defaults to a minimum cosine similarity of 0.82, a 0.02 margin
over the second-best landmark, and 15 inactive frames before a landmark leaves
the search set. Adjust these with `--track-similarity`, `--track-margin`, and
`--track-inactive`. The SQLite file stores landmark IDs and similarities per
frame plus cumulative spherical statistics per landmark. The new schema is
version 2; start with a new database path if you previously ran the version 1
viewer. On resume, cached frames are decoded without real-time pacing, then
the saved active landmark state continues from the last committed frame.

The tracker runs between inference and the storage queue. The display and
database receive the same assignments without re-reading descriptor blobs.
