# Test footage

The lab processes one RGB camera stream. SuperPoint consumes RGB frames and
internally computes grayscale features; no depth, stereo, or IMU is supplied.

## Initial synthetic test

`data/demo.mkv` was generated locally by `scripts/make_demo.py`: 12 frames at
480×320, with a textured background, moving rectangles, a circle, and a frame
label. It validates decoding, extraction, cache resume, and recording export.
It is not a SLAM benchmark. Its recording is `recordings/demo.rrd`.

## TUM sequences, RGB camera only

Source: [TUM benchmark downloads](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
The [file-format documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
describes the 640×480 RGB PNG images. The original distribution includes other
sensor data; the lab reads only `rgb.txt` and the images it references.

| Sequence | RGB frames | Download bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `freiburg1_xyz` | 798 | 448,204,271 | `a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98` |
| `freiburg1_desk` | 613 | 344,011,403 | `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c` |

Official archive links:

- [freiburg1_xyz.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz)
- [freiburg1_desk.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz)

The checksums above were measured from the official downloads on 2026-09-20.
They identify these local copies; they are not separately published vendor checksums.

Both sequences are stored under `data/datasets/tum/rgbd_dataset_<sequence>/`.
Original archive names are retained for provenance. Local archives and downloaded
data are excluded from version control.

Run:

```bash
source .venv/bin/activate
slam-lab process data/datasets/tum --device cpu
slam-lab list
```

### Longer sequences for scale drift

Added 2026-09-30. fr1_xyz and fr1_desk are too short (< 30 s) for scale drift
to accumulate; these are the standard monocular drift benchmarks, all with
motion-capture `groundtruth.txt` in metres.

| Sequence | RGB frames | Duration | Download bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `freiburg3_long_office_household` | 2,585 | 87 s | 1,483,556,251 | `c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c` |
| `freiburg2_desk` | 2,965 | 99 s | 1,893,351,095 | `1a0756d72510a26e26e2a02bf0b6c69c2796619fca531e176fc2e5110173807c` |
| `freiburg1_room` | 1,362 | 49 s | 782,381,450 | `5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da` |

fr3_long_office_household is a large loop around two desks (the usual headline
number); fr2_desk is slow and loops back at the end; fr1_room sweeps a room and
closes a loop, with fast motion.

### Native workbench inputs

The native workbench decodes video, so `native_workbench/tools/tum_to_video.py`
writes three files into each sequence directory (all five are converted):

- `rgb.mp4`: the `rgb.txt` frames in order, H.264 CRF 10, constant 30 fps.
  TUM drops frames, so video time is not capture time.
- `frames.csv`: `frame_index,timestamp`. Join VO poses to `groundtruth.txt`
  through this, by timestamp, never by frame index × 1/30.
- `calibration.json`: published intrinsics (fx fy cx cy) and OpenCV
  distortion. fr3 images are already undistorted; fr1/fr2 have k1 ≈ 0.23–0.26
  in the OpenCV model, which is not the VO's division-model k1.

Monocular scale is arbitrary, so compare trajectories after a Sim(3)
alignment (ATE) and measure drift as relative scale error over sub-windows.

The recordings `recordings/tum-fr1-xyz-rgb.rrd` and
`recordings/tum-fr1-desk-rgb.rrd` show each RGB frame with its SuperPoint keypoints,
confidence values, feature counts, and inference times. No camera trajectory or
3D map is estimated by this stage.
