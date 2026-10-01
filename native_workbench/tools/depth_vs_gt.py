#!/usr/bin/env python3
"""Score live-depth dumps against TUM ground-truth depth.

Input: the depth_<frame>.bin files of slam-native-live-depth-bench (per source
pixel an inverse depth and its variance). For each dump the TUM depth image
nearest in time (within 20 ms; registered to RGB) is the truth.

Monocular depth has an unknown scale, so errors are measured after one scale
per frame (median log ratio over the pixels with an estimate). Rows keep the
pixels whose reported relative sigma (sigma / inverse depth) is below a limit:

  coverage   pixels kept / pixels with ground truth
  err        median |log z - log z_true - scale| (~relative error)
  <5%, <10%  share within that relative error;  <25%: the usual delta < 1.25
  err/sigma  median |error| / sigma / 0.6745 (1 = the variance is calibrated)

usage:
  .venv-cuda/bin/python depth_vs_gt.py SEQUENCE_DIR DUMP_DIR [DUMP_DIR ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LIMITS = (0.02, 0.05, 0.1, 0.2, 0.5, np.inf)


def load_truth(sequence: Path):
    with open(sequence / "frames.csv") as f:
        frame_time = {int(r["frame_index"]): float(r["timestamp"]) for r in csv.DictReader(f)}
    times, files = [], []
    for line in (sequence / "depth.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            t, name = line.split()
            times.append(float(t))
            files.append(sequence / name)
    return frame_time, np.array(times), files


def load_dump(path: Path):
    with open(path, "rb") as f:
        width, height = np.fromfile(f, np.int32, 2)
        frame = int(np.fromfile(f, np.uint64, 1)[0])
        rho = np.fromfile(f, np.float32, width * height).reshape(height, width)
        variance = np.fromfile(f, np.float32, width * height).reshape(height, width)
    return frame, rho, variance


def audit(sequence: Path, dumps: Path):
    frame_time, depth_times, depth_files = load_truth(sequence)
    rows = {limit: {"kept": 0, "truth": 0, "err": [], "z": []} for limit in LIMITS}
    scales, frames = [], 0
    for path in sorted(dumps.glob("depth_*.bin"), key=lambda p: int(p.stem.split("_")[1])):
        frame, rho, variance = load_dump(path)
        t = frame_time.get(frame)
        if t is None:
            continue
        k = int(np.argmin(np.abs(depth_times - t)))
        if abs(depth_times[k] - t) > 0.02:
            continue
        truth = np.asarray(Image.open(depth_files[k]), dtype=np.float64) / 5000.0
        if truth.shape != rho.shape:
            continue
        has_truth = truth > 0
        valid = has_truth & np.isfinite(variance) & (rho > 0)
        if valid.sum() < 500:
            continue
        frames += 1
        log_error = -np.log(rho[valid].astype(np.float64)) - np.log(truth[valid])
        sigma = np.sqrt(variance[valid].astype(np.float64)) / rho[valid]  # of log depth
        scale = np.median(log_error)
        scales.append(np.exp(scale))
        error = np.abs(log_error - scale)
        for limit in LIMITS:
            keep = sigma < limit
            rows[limit]["kept"] += int(keep.sum())
            rows[limit]["truth"] += int(has_truth.sum())
            rows[limit]["err"].append(error[keep])
            rows[limit]["z"].append(error[keep] / np.maximum(sigma[keep], 1e-9))
    return frames, rows, np.array(scales)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    sequence = Path(argv[1])
    for arg in argv[2:]:
        frames, rows, scales = audit(sequence, Path(arg))
        print(f"\n{arg}: {frames} frames with ground truth")
        if not frames:
            continue
        print(f"{'sigma <':>8} {'coverage':>9} {'err':>8} {'<5%':>7} {'<10%':>7} {'<25%':>7} {'err/sigma':>9}")
        for limit in LIMITS:
            r = rows[limit]
            e = np.concatenate(r["err"])
            if not len(e):
                print(f"{limit:>8} {'0.0%':>9}")
                continue
            z = np.concatenate(r["z"])
            print(f"{limit:>8} {100 * r['kept'] / r['truth']:>8.1f}% {100 * np.median(e):>7.2f}% "
                  f"{100 * np.mean(e < 0.05):>6.1f}% {100 * np.mean(e < 0.10):>6.1f}% "
                  f"{100 * np.mean(e < np.log(1.25)):>6.1f}% {np.median(z) / 0.6745:>9.2f}")
        print(f"scale (estimate / truth) per frame: median {np.median(scales):.3f}, "
              f"min {scales.min():.3f}, max {scales.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
