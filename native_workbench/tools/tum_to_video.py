"""Turn a TUM RGB-D sequence into inputs the native workbench can read.

The workbench decodes video (NVDEC), not PNG sequences. For each sequence
directory this writes, next to the original data:

  rgb.mp4           the rgb.txt frames in order, H.264 yuv420p, near-lossless
                    (CRF 10), a constant 30 fps (TUM drops frames, so the video
                    clock is not the capture clock: use frames.csv)
  frames.csv        frame_index,timestamp for every video frame, to join the
                    VO trajectory with groundtruth.txt by timestamp
  calibration.json  the published pinhole intrinsics and OpenCV distortion
                    (fr3 images are already undistorted)

  python tum_to_video.py data/datasets/tum/rgbd_dataset_freiburg3_long_office_household [...]
"""
import json
import subprocess
import sys
from pathlib import Path

# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
CALIBRATION = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3,
                  "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7,
                  "distortion": [0.2312, -0.7849, -0.0033, -0.0001, 0.9172]},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
                  "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]},
}


def frames(sequence: Path) -> list[tuple[str, Path]]:
    rows = []
    for line in (sequence / "rgb.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        timestamp, name = line.split()
        rows.append((timestamp, sequence / name))
    return rows


def convert(sequence: Path) -> None:
    rows = frames(sequence)
    camera = next(v for k, v in CALIBRATION.items() if k in sequence.name)
    (sequence / "calibration.json").write_text(json.dumps(
        {"width": 640, "height": 480, **camera, "distortion_model": "opencv k1 k2 p1 p2 k3"}, indent=2) + "\n")
    with open(sequence / "frames.csv", "w") as out:
        out.write("frame_index,timestamp\n")
        for index, (timestamp, _) in enumerate(rows):
            out.write(f"{index},{timestamp}\n")
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
         str(sequence / "rgb.mp4")],
        stdin=subprocess.PIPE)
    for _, path in rows:
        encoder.stdin.write(path.read_bytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {sequence}")
    print(f"{sequence.name}: {len(rows)} frames")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        convert(Path(argument))
