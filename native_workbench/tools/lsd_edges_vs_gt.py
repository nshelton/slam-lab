#!/usr/bin/env python3
"""Score the LSD keyframe graph's edges against TUM ground truth.

Input: slam-native-lsd-snapshot's LSD_EDGES CSV (accepted edges S_ji with
both keyframes' frame indices). For each edge the ground-truth relative pose
of the two frames gives the true rotation and translation direction (scale is
not observable, so only the direction and |t_est| / baseline are reported).
Edges with > 2 deg rotation or > 20 deg direction error are listed.

usage:
  .venv-cuda/bin/python lsd_edges_vs_gt.py SEQUENCE_DIR EDGES.csv
"""
import sys, csv
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tum_eval as te

seq = Path(sys.argv[1]); edges = list(csv.DictReader(open(sys.argv[2])))
gt_t, gt_p, gt_R = te.load_ground_truth(seq); ft = te.load_frame_times(seq)
def gt(frame):
    t = ft[frame]; k = int(np.argmin(abs(gt_t - t)))
    return (gt_p[k], gt_R[k]) if abs(gt_t[k] - t) < 0.02 else None
def qmat(w, x, y, z): return te.quaternion_to_matrix(w, x, y, z)
rows = []
for e in edges:
    a, b = gt(int(e["frame_i"])), gt(int(e["frame_j"]))
    if a is None or b is None: continue
    (ci, Ri), (cj, Rj) = a, b               # camera -> world
    R_true = Rj.T @ Ri; t_true = Rj.T @ (ci - cj)
    R_est = qmat(*(float(e[k]) for k in ("qw", "qx", "qy", "qz")))
    t_est = np.array([float(e[k]) for k in ("tx", "ty", "tz")])
    dR = R_est @ R_true.T
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    base = np.linalg.norm(t_true)
    ang = np.degrees(np.arccos(np.clip(t_est @ t_true / (np.linalg.norm(t_est) * base + 1e-12), -1, 1))) if base > 0.03 else np.nan
    rows.append((int(e["i"]), int(e["j"]), int(e["fallback"]), int(e["loop"]), float(e["reciprocal"]), rot, ang, base,
                 np.linalg.norm(t_est) / max(base, 1e-9), float(e["s"])))
r = np.array(rows)
print(f"{len(r)} edges with ground truth; rotation error deg percentiles 50/90/max:",
      np.percentile(r[:, 5], [50, 90, 100]).round(2))
ok = ~np.isnan(r[:, 6])
print("translation direction error deg 50/90/max (baseline > 3 cm):", np.percentile(r[ok, 6], [50, 90, 100]).round(1))
print("bad edges (rot > 2 deg or direction > 20 deg):")
for x in r[(r[:, 5] > 2) | (np.nan_to_num(r[:, 6]) > 20)]:
    print(f"  {int(x[0])}->{int(x[1])} fallback {int(x[2])} loop {int(x[3])} recip {x[4]:.0f} rot {x[5]:.2f} dir {x[6]:.1f} base {100*x[7]:.1f} cm |t|/base {x[8]:.2f} s {x[9]:.3f}")
