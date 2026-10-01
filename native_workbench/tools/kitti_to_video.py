"""Turn a KITTI Odometry sequence (left colour camera) into workbench inputs.

Reads <kitti>/sequences/NN/{image_2,times.txt,calib.txt} and
<kitti>/poses/NN.txt (fetch_kitti_odometry.py), and writes into the sequence
directory:

  rgb.mp4           image_2 in order, H.264 yuv420p CRF 10 at KITTI's 10 fps,
                    cropped to even dimensions (right/bottom pixel; the
                    principal point is unchanged)
  frames.csv        frame_index,timestamp (seconds from times.txt)
  calibration.json  P2 intrinsics (images are rectified: no distortion)
  groundtruth.txt   TUM format (timestamp tx ty tz qx qy qz qw), camera-to-world
                    of the left grey camera (cam0; cam2 is ~6 cm to its side),
                    sequences 00-10 only

  python kitti_to_video.py data/datasets/kitti 07 06 [...]
"""
import json
import math
import subprocess
import sys
from pathlib import Path


def quaternion(r: list[list[float]]) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw) of a rotation matrix (Shepperd's method)."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        return ((r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s, s / 4)
    i = max(range(3), key=lambda k: r[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + r[i][i] - r[j][j] - r[k][k])
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4
    q[j] = (r[j][i] + r[i][j]) / s
    q[k] = (r[k][i] + r[i][k]) / s
    return q[0], q[1], q[2], (r[k][j] - r[j][k]) / s


def image_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]  # PNG IHDR
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def convert(kitti: Path, sequence: str) -> None:
    directory = kitti / "sequences" / sequence
    images = sorted((directory / "image_2").glob("*.png"))
    times = [float(line) for line in (directory / "times.txt").read_text().split()]
    if len(images) != len(times):
        raise RuntimeError(f"{sequence}: {len(images)} images but {len(times)} timestamps (incomplete download?)")
    width, height = image_size(images[0])
    width, height = width // 2 * 2, height // 2 * 2
    p2 = next(line.split()[1:] for line in (directory / "calib.txt").read_text().splitlines()
              if line.startswith("P2:"))
    fx, cx, fy, cy = float(p2[0]), float(p2[2]), float(p2[5]), float(p2[6])
    (directory / "calibration.json").write_text(json.dumps(
        {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
         "distortion": [0.0, 0.0, 0.0, 0.0, 0.0], "distortion_model": "none (rectified)",
         "hfov_degrees": round(math.degrees(2 * math.atan(width / 2 / fx)), 3)}, indent=2) + "\n")
    with open(directory / "frames.csv", "w") as out:
        out.write("frame_index,timestamp\n")
        for index, timestamp in enumerate(times):
            out.write(f"{index},{timestamp:.6f}\n")
    poses = kitti / "poses" / f"{sequence}.txt"
    if poses.exists():
        with open(directory / "groundtruth.txt", "w") as out:
            out.write("# timestamp tx ty tz qx qy qz qw (KITTI poses, camera-to-world)\n")
            for timestamp, line in zip(times, poses.read_text().splitlines()):
                v = [float(x) for x in line.split()]
                r = [v[0:3], v[4:7], v[8:11]]
                qx, qy, qz, qw = quaternion(r)
                out.write(f"{timestamp:.6f} {v[3]:.6f} {v[7]:.6f} {v[11]:.6f} "
                          f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", "10", "-i", "-",
         "-vf", f"crop={width}:{height}:0:0", "-c:v", "libx264", "-preset", "slow", "-crf", "10",
         "-pix_fmt", "yuv420p", str(directory / "rgb.mp4")],
        stdin=subprocess.PIPE)
    for path in images:
        encoder.stdin.write(path.read_bytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {sequence}")
    print(f"{sequence}: {len(images)} frames, {width}x{height}, "
          f"--intrinsics {fx:.3f} {fy:.3f} {cx:.3f} {cy:.3f}")


if __name__ == "__main__":
    for argument in sys.argv[2:]:
        convert(Path(sys.argv[1]), argument)
