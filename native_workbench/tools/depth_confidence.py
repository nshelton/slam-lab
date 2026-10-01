#!/usr/bin/env python3
"""Self-calibrated confidence for network depth (DEPTH_INTEGRATION.md, Phase 0b).

The VO's well-triangulated landmarks are a reference for the network's depth,
up to one scale per segment. At every keyframe each such landmark gives a
residual in log depth

    r = log z_net - log z_vo - s_segment          (z_vo = (T_kf X).z)

whose variance is the network's own error plus the landmark's geometric
error (sigma_geo ~ MapPoint::depth_sigma_ratio). The network variance is
modelled from per-sample cues,

    sigma_net^2 = exp(w . cues),   r ~ N(0, sigma_net^2 + sigma_geo^2),

and w is fitted by maximum likelihood. Cues available from the CSV exports:
edge flag (from the sampler's inflated depth_sigma), log depth, image radius,
distance to the nearest other reference landmark, and temporal inconsistency
(the change of the network's log depth for the same landmark since its
previous keyframe, minus the VO's change). Image gradient / texture cues need
the depth maps themselves (later).

Circularity rule: only landmarks with geometry-only constraint qualify as
reference (max parallax >= 2 deg, >= 3 keyframe observations, finite
depth_sigma_ratio); nothing here feeds back into the VO.

Calibration test: landmarks are split 80/20 by id. On the held-out 20 % the
normalized residuals r / sqrt(sigma_net^2 + sigma_geo^2) should have std ~ 1
and ~68 % / 95 % coverage at 1 / 2 sigma. Gross residuals (|r| > log 2, a
factor-2 depth error) are excluded from the fit and reported separately.

Alignment: by default one log scale per segment. With --per-keyframe the
scale is fitted per keyframe instead (median over that keyframe's
references), so the residual is the network's *shape* error only; the
per-segment scale estimator (DEPTH_INTEGRATION Phase 1) is meant to absorb the
rest (VO scale drift and network scale flicker).

usage:
  .venv-cuda/bin/python depth_confidence.py TRACKS.csv TRAJECTORY.csv MAP.csv [--per-keyframe]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

EDGE_RATIO = 0.2  # sampler: depth_sigma = 0.15 z normally, 3x that at depth edges
GROSS = np.log(2.0)


def quaternion_to_matrix(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load(tracks_path, trajectory_path, map_path):
    keyframes = {}
    with open(trajectory_path) as f:
        for r in csv.DictReader(f):
            if r["keyframe"] == "1":
                R_wc = quaternion_to_matrix(*(float(r[k]) for k in ("qw", "qx", "qy", "qz")))
                keyframes[int(r["frame_index"])] = (int(r["segment"]), R_wc.T,
                                                    np.array([float(r[k]) for k in ("cx", "cy", "cz")]))
    landmarks = {}
    with open(map_path) as f:
        for r in csv.DictReader(f):
            ratio = float(r["depth_sigma_ratio"])
            if (float(r["max_parallax_deg"]) >= 2 and int(r["keyframe_observations"]) >= 3 and np.isfinite(ratio)):
                landmarks[int(r["track_id"])] = (int(r.get("landmark_id", r["track_id"])), int(r["segment"]),
                                                 np.array([float(r[k]) for k in ("x", "y", "z")]), ratio)
    width = height = None
    rows = []
    with open(tracks_path) as f:
        header = None
        for line in f:
            if line.startswith("# size"):
                width, height = map(int, line.split()[2:4])
                continue
            if header is None:
                header = line.strip().split(",")
                index = {name: i for i, name in enumerate(header)}
                continue
            frame = int(line[:line.index(",")])
            if frame not in keyframes:
                continue
            v = line.rstrip("\n").split(",")
            track = int(v[index["track_id"]])
            if track not in landmarks:
                continue
            depth = float(v[index["depth"]]) if "depth" in index and v[index["depth"]] else 0.0
            if depth <= 0:
                continue
            rows.append((frame, track, float(v[index["x"]]), float(v[index["y"]]), depth,
                         float(v[index["depth_sigma"]])))
    return keyframes, landmarks, rows, width, height


def build(keyframes, landmarks, rows, width, height):
    samples = []
    for frame, track, x, y, z_net, z_sigma in rows:
        segment, R_cw, centre = keyframes[frame]
        landmark_id, lm_segment, X, ratio = landmarks[track]
        if lm_segment != segment:
            continue
        z_vo = (R_cw @ (X - centre))[2]
        if z_vo <= 0:
            continue
        samples.append(dict(frame=frame, landmark=landmark_id, segment=segment, x=x, y=y,
                            log_net=np.log(z_net), log_vo=np.log(z_vo), geo=ratio,
                            edge=float(z_sigma / z_net > EDGE_RATIO)))
    # Per-segment log scale (robust), residuals.
    by_segment = defaultdict(list)
    for s in samples:
        by_segment[s["segment"]].append(s["log_net"] - s["log_vo"])
    scale = {k: float(np.median(v)) for k, v in by_segment.items()}
    for s in samples:
        s["r"] = s["log_net"] - s["log_vo"] - scale[s["segment"]]
    # Per-keyframe scale jitter (how much a per-frame fit would absorb).
    by_frame = defaultdict(list)
    for s in samples:
        by_frame[s["frame"]].append(s["r"])
    frame_offset = {k: float(np.median(v)) for k, v in by_frame.items() if len(v) >= 10}
    for s in samples:
        s["r_keyframe"] = s["r"] - frame_offset[s["frame"]] if s["frame"] in frame_offset else np.nan
    # Cues: distance to nearest other reference landmark (px / width), temporal.
    half_diagonal = 0.5 * np.hypot(width, height)
    for frame, group in defaultdict(list, {k: [s for s in samples if s["frame"] == k] for k in by_frame}).items():
        P = np.array([[s["x"], s["y"]] for s in group])
        for i, s in enumerate(group):
            d = np.hypot(*(P - P[i]).T)
            d[i] = np.inf
            s["nearest"] = float(d.min()) / width if len(group) > 1 else 1.0
            s["radius"] = np.hypot(s["x"] - 0.5 * width, s["y"] - 0.5 * height) / half_diagonal
    previous = {}
    for s in sorted(samples, key=lambda s: s["frame"]):
        p = previous.get(s["landmark"])
        s["temporal"] = abs((s["log_net"] - p["log_net"]) - (s["log_vo"] - p["log_vo"])) if p else np.nan
        previous[s["landmark"]] = s
    return samples, scale, frame_offset


CUES = ("edge", "log_depth", "radius", "nearest", "temporal")


def design(samples, mean_log):
    X = np.ones((len(samples), 1 + len(CUES)))
    for i, s in enumerate(samples):
        X[i, 1] = s["edge"]
        X[i, 2] = s["log_net"] - mean_log
        X[i, 3] = s["radius"]
        X[i, 4] = s["nearest"]
        X[i, 5] = s["temporal"] if np.isfinite(s["temporal"]) else 0.0
    return X


def fit(X, r, geo2):
    def nll(w):
        v = np.exp(X @ w) + geo2
        return 0.5 * np.sum(np.log(v) + r * r / v)

    def grad(w):
        e = np.exp(X @ w)
        v = e + geo2
        return 0.5 * X.T @ ((1 / v - r * r / (v * v)) * e)

    w0 = np.zeros(X.shape[1])
    w0[0] = np.log(max(np.var(r) - np.mean(geo2), 1e-6))
    return minimize(nll, w0, jac=grad, method="BFGS").x


def calibration(r, v):
    z = r / np.sqrt(v)
    return dict(std=float(np.std(z)), within1=float(np.mean(np.abs(z) < 1)), within2=float(np.mean(np.abs(z) < 2)),
                nll=float(0.5 * np.mean(np.log(2 * np.pi * v) + z * z)))


def main(argv):
    per_keyframe = "--per-keyframe" in argv
    argv = [a for a in argv if a != "--per-keyframe"]
    if len(argv) != 4:
        print(__doc__)
        return 2
    keyframes, landmarks, rows, width, height = load(*argv[1:4])
    samples, scale, frame_offset = build(keyframes, landmarks, rows, width, height)
    if per_keyframe:
        samples = [s for s in samples if np.isfinite(s["r_keyframe"])]
        for s in samples:
            s["r"] = s["r_keyframe"]
        print("alignment: per keyframe (shape error only)")
    r_all = np.array([s["r"] for s in samples])
    gross = np.abs(r_all) > GROSS
    print(f"reference samples: {len(samples)} at {len(frame_offset)} keyframes, "
          f"{len({s['landmark'] for s in samples})} landmarks; segments {len(scale)}")
    print(f"segment log scale (net metres per VO unit): " +
          ", ".join(f"{k}: {np.exp(v):.3g}" for k, v in sorted(scale.items())))
    jitter = np.array(list(frame_offset.values()))
    print(f"per-keyframe scale offset within segment: std {np.std(jitter):.3f} (log), "
          f"p5/p95 {np.percentile(jitter, 5):+.3f}/{np.percentile(jitter, 95):+.3f}")
    print(f"gross (factor > 2) residuals: {100 * gross.mean():.1f}%  (edge samples: "
          f"{100 * np.mean([s['edge'] for s in samples]):.1f}% of all, "
          f"{100 * np.mean([s['edge'] for s, g in zip(samples, gross) if g]) if gross.any() else 0:.1f}% of gross)")
    kept = [s for s, g in zip(samples, gross) if not g]
    r = np.array([s["r"] for s in kept])
    geo2 = np.array([s["geo"] ** 2 for s in kept])
    print(f"residual |r| p50/p90 (log): {np.percentile(np.abs(r), 50):.3f}/{np.percentile(np.abs(r), 90):.3f}"
          f"  geometric sigma p50: {np.sqrt(np.median(geo2)):.4f}")
    rng = np.random.default_rng(0)
    ids = np.array(sorted({s["landmark"] for s in kept}))
    held = set(rng.choice(ids, size=len(ids) // 5, replace=False).tolist())
    test = np.array([s["landmark"] in held for s in kept])
    mean_log = float(np.mean([s["log_net"] for s in kept]))
    X = design(kept, mean_log)
    # Constant model (the doc's fixed depth_relative_sigma) vs cue model.
    w_const = fit(X[~test, :1], r[~test], geo2[~test])
    w = fit(X[~test], r[~test], geo2[~test])
    v_const = np.exp(X[test, :1] @ w_const) + geo2[test]
    v_cues = np.exp(X[test] @ w) + geo2[test]
    print(f"constant model: sigma_net = {np.exp(0.5 * w_const[0]):.3f} (log depth)")
    print("cue model, sigma_net multiplier per unit cue: " +
          ", ".join(f"{name} x{np.exp(0.5 * c):.2f}" for name, c in zip(CUES, w[1:])) +
          f"  (base {np.exp(0.5 * w[0]):.3f})")
    for name, v in (("constant", v_const), ("cues", v_cues)):
        c = calibration(r[test], v)
        print(f"held-out {name:9s}: n={test.sum()} normalized std {c['std']:.2f}, within 1 sigma "
              f"{100 * c['within1']:.0f}%, within 2 sigma {100 * c['within2']:.0f}%, NLL {c['nll']:.3f}")
    sigma = np.sqrt(np.exp(X @ w))
    print(f"predicted sigma_net p10/p50/p90: " + "/".join(f"{np.percentile(sigma, q):.3f}" for q in (10, 50, 90)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
