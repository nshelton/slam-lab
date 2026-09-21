# SLAM Native Workbench

This is the native, CUDA-first visualizer. Its first pipeline is deliberately
small:

```text
compressed video
  -> FFmpeg demux + NVDEC
  -> CUDA YUV frame
       -> CUDA/OpenGL display texture
       -> CUDA resize/grayscale -> TensorRT SuperPoint
            -> in-memory feature batch -> online landmark tracker (next)
            -> asynchronous SQLite writer
```

The decoded image never enters host memory and is never written to disk. The
source video remains the sole image source. Only frame identity, timing,
keypoints, scores, and descriptors are persisted.

The implementation currently includes:

- an ImGui/GLFW/OpenGL application with run, pause, step, stop, real-time
  pacing, timing diagnostics, and a keypoint overlay;
- FFmpeg hardware decoding that requires `AV_PIX_FMT_CUDA` and exposes the
  NV12/P010 device planes directly;
- CUDA/OpenGL interop for the video texture;
- a TensorRT SuperPoint runtime with CUDA luma preprocessing, fixed top-k
  outputs, and source-pixel coordinate recovery;
- a bounded asynchronous SQLite writer using WAL and batched transactions;
- an ONNX exporter for the exact LightGlue SuperPoint weights already used by
  the Python pipeline.

## Storage

Descriptors stay float32 in memory so the forthcoming online tracker consumes
the inference result directly. SQLite storage defaults to float16. This reduces
descriptor traffic from about 2 MiB to 1 MiB per frame at 2,048 keypoints.
Use `--descriptor-storage f32` when bit-for-bit float32 persistence matters.
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
the **Native workbench (CUDA)** launch configuration. It prompts for a source
video and TensorRT engine path. The launch configuration additionally needs
`gdb`. **Native: export SuperPoint ONNX** and **Native: build SuperPoint
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

Run it with:

```bash
native_workbench/build/app-release/slam-native-workbench \
  --video /path/to/video.mp4 \
  --engine native_workbench/models/superpoint-1024x576-k2048.engine \
  --db recordings/native-osaka/features.sqlite3
```

If your terminal is already in `native_workbench/build/app-debug`, use the
engine relative to that directory:

```bash
./slam-native-workbench \
  --video /home/nick/Downloads/osaka.mp4 \
  --engine ../../models/superpoint-1024x576-k2048.engine \
  --db /home/nick/slam-lab/recordings/native-osaka/features.sqlite3
```

## Next integration point

`app.cpp` marks the exact point where online landmark association attaches:
after inference and before ownership moves to the storage queue. The tracker
will therefore receive every descriptor without querying SQLite or waiting for
disk. Its assignments can be added to the same `FrameFeatures` batch and drawn
as stable colors and trails in the current image viewport.
