#!/usr/bin/env python3
"""Score the dense keyframe clouds' multi-view filter against TUM ground-truth depth.

For every exported cloud point (bench `dense 1 export-clouds DIR`), the TUM
depth image of that keyframe (registered to RGB; nearest timestamp within
20 ms) gives the true depth at its pixel. Per keyframe, the network's depth is
aligned by its median log ratio to the truth (one scale per keyframe, the same
for every variant), and the point's error is |log z_net - log z_true - median|.
A good filter hides the points with large error and keeps the others.

usage:
  .venv-cuda/bin/python dense_vs_gt.py SEQUENCE_DIR NAME=CLOUDS.csv [NAME=CLOUDS.csv ...]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


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


def errors(clouds: Path, sequence: Path):
    frame_time, depth_times, depth_files = load_truth(sequence)
    rows = defaultdict(list)
    with open(clouds) as f:
        for r in csv.DictReader(f):
            rows[int(r["frame_index"])].append((float(r["x"]), float(r["y"]), float(r["depth_m"]), int(r["shown"]),
                                                int(r["agree"]), int(r["disagree"])))
    out = []  # (frame, x, y, error, shown, checked)
    for frame, points in rows.items():
        t = frame_time.get(frame)
        if t is None:
            continue
        k = int(np.argmin(np.abs(depth_times - t)))
        if abs(depth_times[k] - t) > 0.02:
            continue
        truth = np.asarray(Image.open(depth_files[k]), dtype=np.float64) / 5000.0
        h, w = truth.shape
        p = np.array(points)
        u = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        z_true = truth[v, u]
        valid = z_true > 0
        if valid.sum() < 50:
            continue
        log_ratio = np.log(p[valid, 2]) - np.log(z_true[valid])
        e = np.abs(log_ratio - np.median(log_ratio))
        for (x, y, _, shown, agree, disagree), err in zip(p[valid], e):
            out.append((frame, round(x, 2), round(y, 2), err, int(shown), int(agree + disagree > 0)))
    return out


def summary(label, e):
    e = np.asarray(e)
    if len(e) == 0:
        return f"{label}: none"
    return (f"{label}: n={len(e):7d}  |e| p50 {np.median(e):.3f} p90 {np.percentile(e, 90):.3f}  "
            f">0.1: {100 * np.mean(e > 0.1):5.1f}%  >0.2: {100 * np.mean(e > 0.2):5.1f}%")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    sequence = Path(argv[1])
    runs = {}
    for arg in argv[2:]:
        name, path = arg.split("=", 1)
        runs[name] = {(f, x, y): (err, shown, checked) for f, x, y, err, shown, checked in errors(Path(path), sequence)}
        values = list(runs[name].values())
        print(f"== {name}")
        print("  " + summary("shown ", [e for e, s, c in values if s]))
        print("  " + summary("hidden", [e for e, s, c in values if not s]))
        bad = np.array([e > 0.1 for e, s, c in values])
        hidden = np.array([not s for e, s, c in values])
        print(f"  of points with |e| > 0.1: {100 * np.mean(hidden[bad]):.1f}% hidden;  "
              f"of points with |e| <= 0.1: {100 * np.mean(hidden[~bad]):.1f}% hidden")
    if len(runs) == 2:
        (a, ra), (b, rb) = runs.items()
        common = ra.keys() & rb.keys()
        flipped = [ra[k][0] for k in common if not ra[k][1] and rb[k][1]]
        print("  " + summary(f"hidden by {a}, shown by {b}", flipped))
        newly = [ra[k][0] for k in common if ra[k][1] and not rb[k][1]]
        print("  " + summary(f"shown by {a}, hidden by {b}", newly))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
