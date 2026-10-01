"""Drift of a VO trajectory against TUM ground truth.

Aligns (similarity) on the first 25 % of the run, then reports the position
error at points along the run, at the ground truth's closest return to the
start (the loop), the whole-run ATE, and the local scale per quarter
(scale drift).

  .venv-cuda/bin/python trajectory_vs_gt.py TRAJ.csv TUM_SEQUENCE_DIR
"""
import sys
from pathlib import Path

import numpy as np


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / Xc.var(0).sum()
    return s, R, my - s * R @ mx


def main(traj_path, sequence):
    sequence = Path(sequence)
    gt = np.loadtxt(sequence / "groundtruth.txt", comments="#")
    frames = np.loadtxt(sequence / "frames.csv", delimiter=",", skiprows=1)
    stamp = dict(zip(frames[:, 0].astype(int), frames[:, 1]))
    a = np.loadtxt(traj_path, delimiter=",", skiprows=1)
    largest = np.bincount(a[:, 2].astype(int)).argmax()
    a = a[a[:, 2] == largest]
    t = np.array([stamp[int(f)] for f in a[:, 0]])
    idx = np.clip(np.searchsorted(gt[:, 0], t), 0, len(gt) - 1)
    P, C = gt[idx, 1:4], a[:, 4:7]
    n = max(3, len(C) // 4)
    s, R, tr = umeyama(C[:n], P[:n])
    E = np.linalg.norm((s * (R @ C.T)).T + tr - P, axis=1)
    length = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
    print(f"{len(C)} poses (largest segment {largest}), ground-truth path {length:.1f} m; aligned on the first 25 %:")
    print("  error at " + ", ".join(f"{q:.0%} {E[min(len(E) - 1, int(q * len(E)) - 1)]:.3f} m" for q in (0.25, 0.5, 0.75, 0.9, 1.0)))
    tail = int(0.75 * len(P))
    d = np.linalg.norm(P[tail:] - P[0], axis=1)
    k = tail + int(np.argmin(d))
    print(f"  return to the start: frame {int(a[k, 0])} ({d.min():.2f} m from the first pose), error {E[k]:.3f} m")
    s2, R2, t2 = umeyama(C, P)
    ate = np.sqrt(np.mean(np.sum(((s2 * (R2 @ C.T)).T + t2 - P) ** 2, 1)))
    quarters = [umeyama(C[int(q * len(C)):int((q + 0.25) * len(C))], P[int(q * len(C)):int((q + 0.25) * len(C))])[0]
                for q in (0, 0.25, 0.5, 0.75)]
    print(f"  ATE {ate:.3f} m; local scale (m/unit) by quarter " + ", ".join(f"{q:.3f}" for q in quarters))


if __name__ == "__main__":
    main(*sys.argv[1:3])
