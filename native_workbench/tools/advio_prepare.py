"""Prepare ADVIO sequences (iPhone camera) as workbench inputs.

ADVIO (https://github.com/AaltoVision/ADVIO, CC BY-NC 4.0) ships the iPhone
video as H.264 already (frames.mov: 1280x720 landscape, 60 fps, display
matrix -90 deg, so it shows as portrait). The workbench reads it directly;
this writes, next to iphone/ in each advio-NN directory:

  rgb.mov           symlink to iphone/frames.mov
  frames.csv        frame_index,timestamp (seconds, the shared sensor clock)
  calibration.json  intrinsics in *native* (landscape, unrotated) pixels, as
                    the pipeline uses them; ADVIO publishes them for the
                    portrait image, obtained by rotating 90 deg clockwise
                    (their advio_to_rosbag.py: transpose + horizontal flip).
                    Also the OpenCV radial-tangential coefficients and the
                    first-order equivalent of the VO's division-model k1.
  groundtruth.txt   TUM format (timestamp tx ty tz qx qy qz qw), from
                    ground-truth/pose.csv (100 Hz, quaternion stored w x y z),
                    metres, IMU + manual fix points (decimetre-level, not mocap)

  python advio_prepare.py data/datasets/advio/advio-01 [...]
"""
import json
import math
import os
import sys
from pathlib import Path

# repo/calibration/README.md (portrait: fx fy cx cy k1 k2 p1 p2), by sequence batch.
CALIBRATION = [
    (range(1, 13), (1077.2, 1079.3, 362.14, 636.39, 0.0478, 0.0339, -0.0003, -0.0009)),
    (range(13, 18), (1082.4, 1084.4, 364.68, 643.31, 0.0366, 0.0803, 0.0007, -0.0002)),
    (range(18, 20), (1076.9, 1078.5, 360.96, 639.31, 0.0510, -0.0354, -0.0054, 0.0473)),
    (range(20, 24), (1081.1, 1082.1, 359.59, 640.79, 0.0556, -0.0454, 0.0009, -0.0018)),
]
# repo/README.md, venue table.
VENUES = {**{n: "mall" for n in range(1, 11)}, 11: "metro", 12: "metro",
          **{n: "office" for n in range(13, 20)}, 20: "outdoor", 21: "outdoor",
          22: "outdoor (urban)", 23: "outdoor (urban)"}
WIDTH, HEIGHT = 1280, 720  # native (landscape) frame


def prepare(directory: Path) -> None:
    number = int(directory.name.rsplit("-", 1)[1])
    fx_p, fy_p, cx_p, cy_p, k1, k2, p1, p2 = next(c for r, c in CALIBRATION if number in r)
    # Portrait (u, v) = (H - y, x) in continuous coordinates: native x = v, y = H - u.
    fx, fy, cx, cy = fy_p, fx_p, cy_p, HEIGHT - cx_p
    half_diagonal2 = 0.25 * (WIDTH**2 + HEIGHT**2)
    (directory / "calibration.json").write_text(json.dumps({
        "width": WIDTH, "height": HEIGHT, "fx": fx, "fy": fy, "cx": round(cx, 3), "cy": round(cy, 3),
        "hfov_degrees": round(math.degrees(2 * math.atan(WIDTH / 2 / fx)), 3),
        "display_rotation_cw": 90,
        "portrait": {"width": HEIGHT, "height": WIDTH, "fx": fx_p, "fy": fy_p, "cx": cx_p, "cy": cy_p},
        "distortion": [k1, k2, p1, p2], "distortion_model": "opencv k1 k2 p1 p2 (portrait image)",
        # x_u = x_d / (1 + k r^2 / h^2) vs x_d ~= x_u (1 + k1 (r/f)^2): k ~= k1 h^2 / f^2.
        "vo_division_k1": round(k1 * half_diagonal2 / (fx * fy), 4),
        "venue": VENUES[number],
    }, indent=2) + "\n")

    rows = [line.split(",") for line in (directory / "iphone/frames.csv").read_text().split()]
    with open(directory / "frames.csv", "w") as out:
        out.write("frame_index,timestamp\n")
        for index, (timestamp, _) in enumerate(rows):
            out.write(f"{index},{timestamp}\n")

    poses = [[float(v) for v in line.split(",")]
             for line in (directory / "ground-truth/pose.csv").read_text().split()]
    with open(directory / "groundtruth.txt", "w") as out:
        out.write("# timestamp tx ty tz qx qy qz qw (ADVIO ground truth, camera-to-world)\n")
        for t, x, y, z, qw, qx, qy, qz in poses:
            out.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

    link = directory / "rgb.mov"
    if not link.exists():
        os.symlink("iphone/frames.mov", link)

    length = sum(math.dist(a[1:4], b[1:4]) for a, b in zip(poses, poses[1:]))
    gap = math.dist(poses[0][1:4], poses[-1][1:4])
    print(f"{directory.name}  {VENUES[number]:<16} {len(rows):>6} frames  {poses[-1][0] / 60:4.1f} min  "
          f"{length:6.0f} m  start-end {gap:5.1f} m  --intrinsics {fx} {fy} {cx:.2f} {cy:.2f}")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        prepare(Path(argument))
