# Native workbench architecture

## Critical path

There is one decoded-frame owner and no image fan-out through files:

1. FFmpeg demuxes compressed packets and NVDEC produces an NV12 or P010 CUDA
   surface.
2. CUDA/OpenGL interop converts that surface into the persistent display
   texture.
3. A CUDA kernel reads the luma plane, normalizes it, and letterboxes it into
   the TensorRT input allocation.
4. TensorRT runs SuperPoint. Its fixed top-k output avoids dynamic allocation
   in the frame loop.
5. The CPU receives keypoints, scores, and descriptors. cuBLAS searches the
   active landmark means on the GPU, then a one-to-one greedy pass updates the
   spherical resultants on the CPU.
6. A bounded writer queue moves persistence to a separate thread. SQLite uses
   WAL, `synchronous=NORMAL`, and multi-frame transactions.

The alternative optical-flow path takes the decoded luma plane directly into
the NVIDIA Optical Flow CUDA API. The dedicated optical-flow engine emits a
4x4 grid of S10.5 motion vectors, which advects regularly seeded track points.
After each step, the tracker keeps the longest-lived track in each seed cell
and respawns a point in every empty cell, preserving coverage after camera motion.
It does not run TensorRT, compute descriptors, or perform cosine association.
The same SQLite writer stores point coordinates and IDs with
`descriptor_encoding=none`; method metadata keeps the databases distinct.

The initial implementation is synchronous across display and inference so its
timings are easy to trust. The next performance pass should use three frame
slots with CUDA events: decode/display N+1, infer N, and read back/write N-1.

The video scrubber lazily opens a second NVDEC decoder. It seeks backward to a
keyframe and decodes forward to the chosen timestamp for display. The primary
decoder and online tracker remain at their sequential processing position.
Cached feature overlays are looked up by source timestamp and PTS through an
indexed SQLite read-only connection. This keeps scrub previews separate from
the causal track and database write path.

## Persistence contract

The source video is the only image store. A frame row contains:

- source `frame_index`, `timestamp_ns`, and `pts`;
- source dimensions;
- packed little-endian float32 `(x, y)` keypoints and scores;
- packed row-major 256-D descriptors in float16 or float32;
- decode, preprocessing, inference, and readback timings.
- landmark IDs and cosine scores for every feature, tracking timing, and
  new/matched counts; a `landmarks` table stores cumulative resultants.

There is no compatibility promise with the Python workbench cache. The native
store has its own schema version and can change while this prototype is being
measured.

## Why C++ rather than C

Dear ImGui is a C++ library, TensorRT's native API is C++, and RAII is useful
for CUDA resources, FFmpeg references, graphics handles, and a writer thread.
A C layer would add wrappers without making the hot path faster.

## Landmark tracking boundary

The current tracker uploads active landmark means each frame and uses a
dedicated CUDA stream for the cosine matrix and top-two reduction. It emits
assignments and updated diagnostics to the UI/DB:

```text
SuperPoint output
  -> cosine association on GPU
  -> one-to-one assignment for current frame
  -> update mean direction, resultant length, count, last-seen frame
  -> create landmarks for unmatched observations
  -> compact assignment IDs copied to UI and persistence
```

This preserves the descriptor-only experiment. Geometry, LightGlue, and pose
estimation can remain comparison modes outside this first tracker.

The diagnostics window fits a fixed two-component PCA basis to the first
2,048 descriptors and projects current observations onto it. It does not
participate in association. Track-length histogram bins update in memory as
each landmark grows; on resume, they are rebuilt from persisted landmark
observation counts.

## Scale decisions still to measure

- Float16 persistence halves bandwidth; compare association/evaluation results
  against float32 before making it the archival format.
- A single SQLite file is suitable for the first throughput tests. Rotate into
  immutable segments before long benchmark jobs.
- Measure each stage independently with CUDA events and queue occupancy. If the
  writer queue fills, the UI must show storage backpressure rather than hiding
  it inside aggregate FPS.
- Hashing a multi-gigabyte video must run in the background. Opening and first
  frame display cannot wait for a complete source hash.
- Resume currently decodes from the beginning to preserve exact source frame
  ordinals. Add a keyframe index and an exact decode-forward seek before
  multi-hour runs; a timestamp-derived frame number is not sufficient for VFR.
