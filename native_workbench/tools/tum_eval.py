#!/usr/bin/env python3
"""Score VO trajectories against TUM RGB-D ground truth (ATE and RPE).

The VO is monocular, so each segment is aligned to the ground truth with its
own similarity transform (Umeyama, sim3). Video frames map to capture
timestamps through the sequence's frames.csv (written by tum_to_video.py);
each pose is joined with the nearest ground-truth sample within 20 ms.

Reported per trajectory:
  posed      posed frames / video frames (all segments)
  segments   number of segments with at least 10 matched poses
  ate_cm     RMS position error of the largest segment after sim3, cm
  ate_all_cm the same over all segments (each aligned on its own)
  rpe_cm     RMS translation drift over 1 s windows (largest segment, after
             the segment's sim3 scale), cm
  rpe_deg    RMS rotation drift over 1 s windows, degrees

usage:
  .venv-cuda/bin/python tum_eval.py SEQUENCE_DIR TRAJECTORY.csv [TRAJECTORY.csv ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_ground_truth(sequence: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    times, centres, rotations = [], [], []
    for line in (sequence / "groundtruth.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, tx, ty, tz, qx, qy, qz, qw = map(float, line.split())
        times.append(t)
        centres.append((tx, ty, tz))
        rotations.append(quaternion_to_matrix(qw, qx, qy, qz))  # camera -> world
    return np.array(times), np.array(centres), rotations


def load_frame_times(sequence: Path) -> dict[int, float]:
    with open(sequence / "frames.csv") as f:
        return {int(r["frame_index"]): float(r["timestamp"]) for r in csv.DictReader(f)}


def load_trajectory(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{
        "frame": int(r["frame_index"]),
        "segment": int(r["segment"]),
        "centre": np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]),
        "rotation": quaternion_to_matrix(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])),
    } for r in rows]


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |s R source + t - target| (rows are points)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    U, D, Vt = np.linalg.svd(b.T @ a / len(source))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    variance = (a ** 2).sum() / len(source)
    s = np.trace(np.diag(D) @ S) / variance if variance > 0 else 1.0
    return s, R, mu_t - s * R @ mu_s


def evaluate(sequence: Path, trajectory: Path) -> dict:
    gt_time, gt_centre, gt_rotation = load_ground_truth(sequence)
    frame_time = load_frame_times(sequence)
    poses = load_trajectory(trajectory)
    segments: dict[int, list] = {}
    for p in poses:
        t = frame_time.get(p["frame"])
        if t is None:
            continue
        k = int(np.searchsorted(gt_time, t))
        best = min((j for j in (k - 1, k) if 0 <= j < len(gt_time)), key=lambda j: abs(gt_time[j] - t))
        if abs(gt_time[best] - t) > 0.02:
            continue
        segments.setdefault(p["segment"], []).append((t, p, gt_centre[best], gt_rotation[best]))

    result = {"posed": f"{len(poses)}/{len(frame_time)}", "segments": 0}
    squared_all, count_all = 0.0, 0
    largest = None
    for matches in segments.values():
        if len(matches) < 10:
            continue
        result["segments"] += 1
        est = np.array([m[1]["centre"] for m in matches])
        gt = np.array([m[2] for m in matches])
        s, R, t = umeyama(est, gt)
        errors = np.linalg.norm((s * (R @ est.T)).T + t - gt, axis=1)
        squared_all += (errors ** 2).sum()
        count_all += len(errors)
        if largest is None or len(matches) > len(largest[0]):
            largest = (matches, errors, s, R)
    if largest is None:
        return result | {"ate_cm": float("nan"), "ate_all_cm": float("nan"), "rpe_cm": float("nan"),
                         "rpe_deg": float("nan"), "largest": 0}
    matches, errors, s, R = largest
    result["largest"] = len(matches)
    result["ate_cm"] = 100 * np.sqrt((errors ** 2).mean())
    result["ate_all_cm"] = 100 * np.sqrt(squared_all / count_all)

    # RPE over 1 s: relative motion between poses ~1 s apart, est (scaled into
    # metres) vs ground truth, compared in the first pose's frame.
    times = np.array([m[0] for m in matches])
    drift_t, drift_r = [], []
    for i, m in enumerate(matches):
        j = int(np.searchsorted(times, times[i] + 1.0))
        if j >= len(matches) or abs(times[j] - times[i] - 1.0) > 0.1:
            continue
        Ra, Rb = R @ m[1]["rotation"], R @ matches[j][1]["rotation"]  # camera -> aligned world
        est_rel_t = Ra.T @ (s * R @ (matches[j][1]["centre"] - m[1]["centre"]))
        gt_rel_t = m[3].T @ (matches[j][2] - m[2])
        drift_t.append(np.linalg.norm(est_rel_t - gt_rel_t))
        delta = (Ra.T @ Rb).T @ (m[3].T @ matches[j][3])
        drift_r.append(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
    result["rpe_cm"] = 100 * np.sqrt(np.mean(np.square(drift_t))) if drift_t else float("nan")
    result["rpe_deg"] = np.sqrt(np.mean(np.square(drift_r))) if drift_r else float("nan")
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    sequence = Path(argv[1])
    print(f"{'trajectory':40s} {'posed':>10s} {'seg':>4s} {'largest':>8s} {'ate_cm':>8s} {'ate_all':>8s} "
          f"{'rpe_cm':>8s} {'rpe_deg':>8s}")
    for path in argv[2:]:
        r = evaluate(sequence, Path(path))
        print(f"{Path(path).name[-40:]:40s} {r['posed']:>10s} {r['segments']:4d} {r.get('largest', 0):8d} "
              f"{r['ate_cm']:8.2f} {r['ate_all_cm']:8.2f} {r['rpe_cm']:8.2f} {r['rpe_deg']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
