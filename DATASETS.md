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

## KITTI Odometry: large-scale driving with loops

Added 2026-09-30. Source: [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
(CC BY-NC-SA 3.0). Car-mounted stereo at 10 Hz over urban and rural routes,
with GPS/INS ground-truth poses for 00–10, in metres. Only the left colour
camera (`image_2`, rectified, hfov ≈ 82°) is fetched. Two calibrations
(after cropping to even size):

| Sequences | Size | `--intrinsics` (fx fy cx cy) |
| --- | --- | --- |
| 00, 02 | 1240×376 | `718.856 718.856 607.193 185.216` |
| 05, 06, 07, 08, 09 | 1226×370 | `707.091 707.091 601.887 183.110` |

The official colour archive is a single 69 GB zip of all 22 sequences.
`native_workbench/tools/fetch_kitti_odometry.py` reads the zip's central
directory over HTTP range requests and pulls only the chosen sequences'
`image_2` and `times.txt`, plus `calib.txt` and `poses/NN.txt`. Interrupted runs resume.

| Sequence | Frames | Size | Length | Loops |
| --- | ---: | ---: | ---: | --- |
| `00` | 4,541 | 3.63 GB | 3.7 km | many; the standard loop-closure test |
| `02` | 4,661 | 3.98 GB | 5.1 km | one late loop, long open stretches |
| `05` | 2,761 | 2.21 GB | 2.2 km | several |
| `06` | 1,101 | 0.89 GB | 1.2 km | one loop, traversed twice (same direction) |
| `07` | 1,101 | 0.87 GB | 0.7 km | one loop closing at the end |
| `08` | 4,071 | 3.34 GB | 3.2 km | revisits in the *opposite* direction (hard for appearance-based closure) |
| `09` | 1,591 | 1.30 GB | 1.7 km | one loop at the very end |

```bash
.venv-cuda/bin/python native_workbench/tools/fetch_kitti_odometry.py data/datasets/kitti \
  --sequences 07 06 09 05 00 08 02     # --list prints every sequence's size
.venv-cuda/bin/python native_workbench/tools/kitti_to_video.py data/datasets/kitti 07 06 09 05 00 08 02
```

`kitti_to_video.py` writes, into `data/datasets/kitti/sequences/NN/`:
`rgb.mp4` (10 fps, cropped to even size), `frames.csv`, `calibration.json`
(P2 intrinsics plus the equivalent hfov) and `groundtruth.txt` in TUM format, so
TUM evaluation tools apply unchanged. GT is the grey left camera (cam0); the
colour camera is ~6 cm to its side, which is negligible at this scale. Pass the
printed `--intrinsics` to the workbench: the principal point is off-centre.

Caveats for this pipeline: 10 Hz at driving speed gives large inter-frame
motion (tens of px on the sides; mostly forward motion, so the epipole sits
near the image centre), and the frame rate is three times lower than in the
phone clips. Expect the flow tracker's association radius and coasting
defaults to need tuning.

## ADVIO: handheld iPhone video with loops

Added 2026-10-01. Source: [ADVIO](https://github.com/AaltoVision/ADVIO),
archives on [Zenodo 1476931](https://zenodo.org/records/1476931) (CC BY-NC 4.0).
A pedestrian carrying a phone rig through a mall, a metro station, two office
buildings and outdoor (urban) areas in Helsinki. It is the closest benchmark to the
lab's own phone clips. All 23 sequences (3.1 GB of zips) are in
`data/datasets/advio/advio-NN/`; the ADVIO repo (calibration, scripts) is in
`data/datasets/advio/repo/`.

- **Video:** `iphone/frames.mov`, H.264 1280×720 at 60 fps with a −90° display
  matrix (shown portrait). The workbench reads it directly; no re-encoding.
- **Ground truth:** `ground-truth/pose.csv`, 100 Hz camera poses in metres,
  from the IMU plus manually placed fix points. Accuracy is about a decimetre,
  not motion-capture level: good for loop drift and ATE, too coarse for fine RPE.
  ARKit, ARCore and Tango poses are included as phone-VIO baselines.
- **Calibration:** four batches, published for the *portrait* image.

`native_workbench/tools/advio_prepare.py data/datasets/advio/advio-??` writes,
per sequence: `rgb.mov` (symlink), `frames.csv`, `groundtruth.txt` (TUM format)
and `calibration.json`, with intrinsics converted to native landscape pixels
(portrait = native rotated 90° clockwise, as ADVIO's own rosbag script does:
native fx, fy, cx, cy = fy_p, fx_p, cy_p, 720 − cx_p). It also stores
`vo_division_k1` (≈ +0.02, mild pincushion): the first-order equivalent of the
OpenCV k1 for the VO's division model.

| Sequences | `--intrinsics` (native px) |
| --- | --- |
| 01–12 | `1079.3 1077.2 636.39 357.86` |
| 13–17 | `1084.4 1082.4 643.31 355.32` |
| 18–19 | `1078.5 1076.9 639.31 359.04` |
| 20–23 | `1082.1 1081.1 640.79 360.41` |

Closed loops (ground-truth end within 1.5 m of the start), from the GT tracks:

| Seq | Venue | Duration | Path | Start–end |
| --- | --- | ---: | ---: | ---: |
| 03 | mall | 2.5 min | 153 m | 0.2 m |
| 08 | mall | 1.8 min | 99 m | 0.5 m |
| 09 | mall | 1.6 min | 87 m | 0.1 m |
| 10 | mall | 2.1 min | 125 m | 0.7 m |
| 12 | metro | 1.9 min | 136 m | 1.0 m |
| 13 | office | 2.5 min | 142 m | 0.2 m |
| 14 | office | 1.9 min | 70 m | 1.4 m |
| 18 | office | 3.3 min | 132 m | 0.4 m |
| 19 | office | 2.4 min | 135 m | 0.1 m |
| 20 | outdoor | 5.0 min | 474 m | 0.7 m |
| 21 | outdoor | 5.3 min | 477 m | 0.4 m |

The others are open or partial loops (up to 515 m). Several include stairs,
escalators or elevators, which tests monocular VO under motion that vision
alone cannot observe (in an elevator the camera sees no motion), and crowds
(06, 11, 12, 22, 23).

```bash
./run-workbench.sh --video data/datasets/advio/advio-09/rgb.mov \
  --intrinsics 1079.3 1077.2 636.39 357.86 --k1 0.0222
```
