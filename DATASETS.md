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

The recordings `recordings/tum-fr1-xyz-rgb.rrd` and
`recordings/tum-fr1-desk-rgb.rrd` show each RGB frame with its SuperPoint keypoints,
confidence values, feature counts, and inference times. No camera trajectory or
3D map is estimated by this stage.
