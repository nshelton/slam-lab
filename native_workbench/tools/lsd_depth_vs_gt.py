#!/usr/bin/env python3
"""Audit the LSD keyframe depth maps against TUM ground-truth depth.

Input: slam-native-lsd-snapshot's LSD_EXPORT_DEPTH CSV (per keyframe pixel:
distorted source pixel, inverse depth and sigma, and the scale-locked network
prior there). For each keyframe the TUM depth image nearest in time (within
20 ms; registered to RGB) gives the true depth at the pixel.

Monocular depth has an unknown scale, so errors are measured after one scale
per keyframe (median log ratio), separately for the estimate and the prior;
the scale itself is reported (drift between keyframes = scale drift).

  err        median |log z - log z_true - scale| (~relative error)
  <5%, <10%  fraction of pixels within that relative error
  err/sigma  median |error| / sigma_log / 0.6745 (1 = calibrated variance)

usage:
  .venv-cuda/bin/python lsd_depth_vs_gt.py SEQUENCE_DIR DEPTH.csv [DEPTH.csv ...]
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


def audit(sequence: Path, path: Path):
    frame_time, depth_times, depth_files = load_truth(sequence)
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(int(r["keyframe"]), int(r["frame_index"]))].append(
                (float(r["x"]), float(r["y"]), float(r["idepth"]), float(r["sigma"]), float(r["prior_idepth"])))
    per_kf = []
    est_err, est_z, prior_err = [], [], []
    for (kf, frame), points in sorted(rows.items()):
        t = frame_time.get(frame)
        if t is None:
            continue
        k = int(np.argmin(np.abs(depth_times - t)))
        if abs(depth_times[k] - t) > 0.02:
            continue
        truth = np.asarray(Image.open(depth_files[k]), dtype=np.float64) / 5000.0
        h, w = truth.shape
        p = np.array(points)
        u = np.round(p[:, 0]).astype(int)
        v = np.round(p[:, 1]).astype(int)
        inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
        p, u, v = p[inside], u[inside], v[inside]
        z_true = truth[v, u]
        ok = z_true > 0
        if ok.sum() < 50:
            continue
        p, z_true = p[ok], z_true[ok]
        log_est = -np.log(p[:, 2]) - np.log(z_true)
        scale = np.median(log_est)
        e = np.abs(log_est - scale)
        sigma_log = p[:, 3] / p[:, 2]
        est_err.append(e)
        est_z.append(e / np.maximum(sigma_log, 1e-9))
        has_prior = p[:, 4] > 0
        pe = np.array([])
        if has_prior.sum() >= 50:
            log_prior = -np.log(p[has_prior, 4]) - np.log(z_true[has_prior])
            pe = np.abs(log_prior - np.median(log_prior))
            prior_err.append(pe)
        per_kf.append((kf, frame, len(p), np.exp(scale), np.median(e), np.median(pe) if len(pe) else np.nan))
    return per_kf, est_err, est_z, prior_err


def line(label, errs, z=None):
    e = np.concatenate(errs) if errs else np.array([])
    if not len(e):
        return f"{label:<26} (no data)"
    text = (f"{label:<26} {len(e):>9} {100 * np.median(e):>7.2f}% {100 * np.mean(e < 0.05):>6.1f}% "
            f"{100 * np.mean(e < 0.10):>6.1f}%")
    if z is not None:
        text += f" {np.median(np.concatenate(z)) / 0.6745:>9.2f}"
    return text


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    sequence = Path(argv[1])
    for arg in argv[2:]:
        per_kf, est_err, est_z, prior_err = audit(sequence, Path(arg))
        print(f"\n{arg}: {len(per_kf)} keyframes with ground truth")
        print(f"{'':<26} {'pixels':>9} {'err':>8} {'<5%':>7} {'<10%':>7} {'err/sigma':>9}")
        print(line("keyframe depth (LSD)", est_err, est_z))
        print(line("network prior, same px", prior_err))
        scales = np.array([s for _, _, _, s, _, _ in per_kf])
        if len(scales) > 1:
            print(f"scale (est / true) per keyframe: first {scales[0]:.3f}, last {scales[-1]:.3f}, "
                  f"min {scales.min():.3f}, max {scales.max():.3f}")
        print(f"{'kf':>4} {'frame':>6} {'pixels':>7} {'scale':>6} {'err':>7} {'prior':>7}")
        for kf, frame, n, s, e, pe in per_kf:
            print(f"{kf:>4} {frame:>6} {n:>7} {s:>6.3f} {100 * e:>6.2f}% {100 * pe:>6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
