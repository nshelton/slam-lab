"""Do neighbouring keyframes' depth maps agree once placed in the map?

Inputs: a bench run with `depth MODEL export-keyframe-depth DIR export-tracks
DIR/tracks.csv export-trajectory DIR/traj.csv export-map DIR/map.csv`.

For keyframe pairs (A, B) 1, 2 and 4 keyframes apart, every 4th pixel of A's
depth map is back-projected (VO lens, A's final pose, a per-pixel log scale
L_A = log metres-per-unit), moved into B, and compared with B's own depth
there: r = (log d_B - L_B) - log z_B (0 = consistent; log units, so 0.1 is
about 10 %). The per-keyframe correction L comes from:

  kf-scale     the keyframe's own estimate (what the dense view draws now)
  lm-scale     median log(d / z_vo) over all landmarks seen at the keyframe
  lm-affine    inverse depth: 1/z_vo = a / d + b, fitted robustly to landmarks
  lm-grid      a 4x3 bilinear grid of log corrections fitted to landmarks
  vo-grid      the VO's own per-keyframe grid (OdometryFrameResult::keyframe_scale_grid)
  vo-settled   the VO's grid refitted when the keyframe left the BA window
  oracle-pair  kf-scale plus the pair's own median residual (upper bound
               of any single-scale fix)

Breakdowns: edge pixels (3x3 max/min > 1.1 in A), image radius (lens/FOV
mismatch shows up as a radial trend of the signed residual), and depth.

  .venv-cuda/bin/python depth_consistency.py DIR (--hfov H [--k1 K] | --K fx fy cx cy [--k1 K])
"""
import argparse
import csv
from pathlib import Path

import numpy as np

STRIDE = 4


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Lens:
    def __init__(self, W, H, args):
        if args.K:
            self.fx, self.fy, self.cx, self.cy = args.K
        else:
            f = 0.5 * W / np.tan(np.radians(args.hfov) / 2)
            self.fx = self.fy = f
            self.cx, self.cy = 0.5 * (W - 1), 0.5 * (H - 1)
        self.k1, self.W, self.H = args.k1, W, H
        self.hd2 = 0.25 * (W * W + H * H)
        self.c = np.array([0.5 * (W - 1), 0.5 * (H - 1)])

    def undistort(self, p):
        if self.k1 == 0:
            return p
        d = p - self.c
        return self.c + d / (1 + self.k1 * (d ** 2).sum(1, keepdims=True) / self.hd2)

    def distort(self, p):
        if self.k1 == 0:
            return p
        d = p - self.c
        ru = np.linalg.norm(d, axis=1, keepdims=True)
        a = self.k1 * ru / self.hd2
        rd = 2 * ru / (1 + np.sqrt(np.maximum(0, 1 - 4 * a * ru)))
        return self.c + d * np.where(ru > 1e-9, rd / np.maximum(ru, 1e-12), 1)

    def rays(self, p_src):
        u = self.undistort(p_src)
        return np.stack([(u[:, 0] - self.cx) / self.fx, (u[:, 1] - self.cy) / self.fy, np.ones(len(u))], 1)

    def project(self, X):
        z = X[:, 2]
        u = np.stack([self.fx * X[:, 0] / z + self.cx, self.fy * X[:, 1] / z + self.cy], 1)
        return self.distort(u), z


class Keyframe:
    def __init__(self, row, directory):
        self.frame = int(row["frame_index"])
        self.segment = int(row["segment"])
        self.w, self.h = int(row["width"]), int(row["height"])
        self.sw, self.sh = int(row["source_width"]), int(row["source_height"])
        self.rot = int(row["rotation"])
        self.sx, self.sy = float(row["scale_x"]), float(row["scale_y"])
        self.ox, self.oy = float(row["offset_x"]), float(row["offset_y"])
        ls = float(row["keyframe_log_scale"])
        ms = float(row["metric_scale"])
        self.kf_log_scale = ls if np.isfinite(ls) else (np.log(ms) if ms > 0 else np.nan)
        self.depth = np.fromfile(directory / f"kf_{self.frame}.f32", np.float32).reshape(self.h, self.w)
        g = [float(row.get(f"g{k}", "nan") or "nan") for k in range(12)]
        self.vo_grid = np.array(g) if np.all(np.isfinite(g)) else None

    def to_output(self, p):  # native source px -> output px (DepthMap::to_output)
        cx, cy = p[:, 0] + 0.5, p[:, 1] + 0.5
        w, h = self.sw, self.sh
        if self.rot == 90:
            ux, uy = h - cy, cx
        elif self.rot == 180:
            ux, uy = w - cx, h - cy
        elif self.rot == 270:
            ux, uy = cy, w - cx
        else:
            ux, uy = cx, cy
        return np.stack([ux * self.sx + self.ox - 0.5, uy * self.sy + self.oy - 0.5], 1)

    def to_source(self, o):  # inverse
        ux, uy = (o[:, 0] + 0.5 - self.ox) / self.sx, (o[:, 1] + 0.5 - self.oy) / self.sy
        w, h = self.sw, self.sh
        if self.rot == 90:
            cx, cy = uy, h - ux
        elif self.rot == 180:
            cx, cy = w - ux, h - uy
        elif self.rot == 270:
            cx, cy = w - uy, ux
        else:
            cx, cy = ux, uy
        return np.stack([cx - 0.5, cy - 0.5], 1)

    def sample(self, o):  # bilinear on the output grid; nan in padding / outside
        u, v = o[:, 0], o[:, 1]
        ok = (u >= 0) & (v >= 0) & (u <= self.w - 1) & (v <= self.h - 1)
        u0 = np.clip(np.floor(u).astype(int), 0, self.w - 2)
        v0 = np.clip(np.floor(v).astype(int), 0, self.h - 2)
        a, b = u - u0, v - v0
        D = self.depth
        q = np.stack([D[v0, u0], D[v0, u0 + 1], D[v0 + 1, u0], D[v0 + 1, u0 + 1]], 1)
        val = (q[:, 0] * (1 - a) + q[:, 1] * a) * (1 - b) + (q[:, 2] * (1 - a) + q[:, 3] * a) * b
        ok &= (q > 0).all(1)
        return np.where(ok, val, np.nan)

    def grid(self):
        vs, us = np.mgrid[STRIDE // 2:self.h:STRIDE, STRIDE // 2:self.w:STRIDE]
        o = np.stack([us.ravel(), vs.ravel()], 1).astype(float)
        d = self.depth[vs.ravel(), us.ravel()]
        # 3x3 edge flag
        pad = np.pad(self.depth, 1, mode="edge")
        stack = np.stack([pad[1 + dv:1 + dv + self.h, 1 + du:1 + du + self.w] for dv in (-1, 0, 1) for du in (-1, 0, 1)])
        lo, hi = stack.min(0), stack.max(0)
        edge = (hi > 1.1 * lo)[vs.ravel(), us.ravel()]
        p = self.to_source(o)
        inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] <= self.sw - 1) & (p[:, 1] <= self.sh - 1) & (d > 0) & (lo[vs.ravel(), us.ravel()] > 0)
        return p[inside], o[inside], d[inside], edge[inside]


def huber_lstsq(A, y, iterations=10, delta=0.05, ridge=None):
    w = np.ones(len(y))
    x = None
    for _ in range(iterations):
        Aw = A * w[:, None]
        M = A.T @ Aw
        if ridge is not None:
            M = M + ridge
        x = np.linalg.solve(M, Aw.T @ y)
        r = np.abs(A @ x - y)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return x


def bilinear_basis(p, W, H, nx=4, ny=3):
    gx = np.clip(p[:, 0] / (W - 1) * (nx - 1), 0, nx - 1 - 1e-9)
    gy = np.clip(p[:, 1] / (H - 1) * (ny - 1), 0, ny - 1 - 1e-9)
    ix, iy = gx.astype(int), gy.astype(int)
    fx, fy = gx - ix, gy - iy
    B = np.zeros((len(p), nx * ny))
    for dx, dy, wgt in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        B[np.arange(len(p)), (iy + dy) * nx + ix + dx] = wgt
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--K", type=float, nargs=4)
    ap.add_argument("--k1", type=float, default=0.0)
    args = ap.parse_args()
    D = args.dir
    rows = list(csv.DictReader(open(D / "keyframes.csv")))
    kfs = [Keyframe(r, D) for r in rows]
    settled = {}
    if (D / "settled.csv").exists():
        for r in csv.DictReader(open(D / "settled.csv")):
            settled[int(r["frame_index"])] = np.array([float(r[f"g{k}"]) for k in range(12)])
    W, H = kfs[0].sw, kfs[0].sh
    lens = Lens(W, H, args)

    traj = np.loadtxt(D / "traj.csv", delimiter=",", skiprows=1)
    pose = {int(r[0]): (quat_to_R(*r[7:11]).T, r[4:7]) for r in traj}  # R world->camera, centre

    # Landmarks (final positions) by latest track id; reference-quality only.
    m = np.genfromtxt(D / "map.csv", delimiter=",", names=True)
    good = (m["keyframe_observations"] >= 3) & (m["max_parallax_deg"] >= 2.0)
    landmark = {int(t): np.array([x, y, z]) for t, x, y, z in zip(m["track_id"][good], m["x"][good], m["y"][good], m["z"][good])}
    # Track depth samples at keyframes (non-edge: sigma/depth <= 0.16).
    samples = {}
    with open(D / "tracks.csv") as f:
        next(f)
        header = next(f).strip().split(",")
        col = {n: i for i, n in enumerate(header)}
        kf_frames = {k.frame for k in kfs}
        for line in f:
            v = line.split(",")
            fr = int(v[0])
            if fr not in kf_frames:
                continue
            d, s = float(v[col["depth"]]), float(v[col["depth_sigma"]])
            t = int(v[2])
            if d > 0 and s <= 0.16 * d and t in landmark:
                samples.setdefault(fr, []).append((float(v[3]), float(v[4]), d, t))

    # Per-keyframe corrections L(p) = log metres per unit at source pixel p.
    corrections = {}
    fit_stats = {"lm-scale": [], "lm-affine": [], "lm-grid": []}
    for k in kfs:
        if k.frame not in pose:
            continue
        R, c = pose[k.frame]
        meth = {"kf-scale": (lambda p, ls=k.kf_log_scale: np.full(len(p), ls))}
        if k.vo_grid is not None:
            meth["vo-grid"] = (lambda p, g=k.vo_grid: bilinear_basis(p, W, H) @ g)
        if k.frame in settled or k.vo_grid is not None:  # settled when available, else provisional
            g = settled.get(k.frame, k.vo_grid)
            meth["vo-settled"] = (lambda p, g=g: bilinear_basis(p, W, H) @ g)
        s = np.array(samples.get(k.frame, []))
        if len(s) >= 30:
            X = np.array([landmark[int(t)] for t in s[:, 3]])
            z = ((X - c) @ R.T)[:, 2]
            ok = z > 1e-9
            s, z = s[ok], z[ok]
            r = np.log(s[:, 2]) - np.log(z)  # log metres per unit, per landmark
            ls = np.median(r)
            meth["lm-scale"] = (lambda p, ls=ls: np.full(len(p), ls))
            fit_stats["lm-scale"].append(np.median(np.abs(r - ls)))
            # inverse depth affine: 1/z = a/d + b  ->  L(p) = log d - log z
            A = np.stack([1 / s[:, 2], np.ones(len(s))], 1)
            a, b = huber_lstsq(A, 1 / z, delta=0.05 * np.median(1 / z))
            fit_stats["lm-affine"].append(np.median(np.abs(np.log(np.maximum(A @ [a, b], 1e-9)) - np.log(1 / z))))
            meth["lm-affine"] = ("affine", a, b)
            Bm = bilinear_basis(s[:, :2], W, H)
            ridge = 2.0 * np.eye(Bm.shape[1])  # pulls cells without landmarks toward the mean
            g = huber_lstsq(Bm, r - ls, ridge=ridge, delta=0.05) + ls
            fit_stats["lm-grid"].append(np.median(np.abs(Bm @ g - r)))
            meth["lm-grid"] = (lambda p, g=g: bilinear_basis(p, W, H) @ g)
        corrections[k.frame] = meth

    def corrected_log_units(meth, p, d):
        if isinstance(meth, tuple):
            _, a, b = meth
            inv = a / d + b
            return np.where(inv > 1e-9, -np.log(np.maximum(inv, 1e-9)), np.nan)
        return np.log(d) - meth(p)

    methods = ["kf-scale", "lm-scale", "lm-affine", "lm-grid", "vo-grid", "vo-settled", "oracle-pair"]
    results = {(g, mth): [] for g in (1, 2, 4) for mth in methods}
    detail = {mth: {"edge": [], "radius": [], "depth": [], "r": []} for mth in methods}
    for i, A in enumerate(kfs):
        if A.frame not in corrections:
            continue
        pA, oA, dA, edgeA = A.grid()
        rays = lens.rays(pA)
        RA, cA = pose[A.frame]
        for gap in (1, 2, 4):
            if i + gap >= len(kfs):
                continue
            B = kfs[i + gap]
            if B.segment != A.segment or B.frame not in corrections:
                continue
            RB, cB = pose[B.frame]
            for mth in methods:
                base = "kf-scale" if mth == "oracle-pair" else mth
                if base not in corrections[A.frame] or base not in corrections[B.frame]:
                    continue
                zA = np.exp(corrected_log_units(corrections[A.frame][base], pA, dA))
                Xw = (zA[:, None] * rays) @ RA + cA
                XB = (Xw - cB) @ RB.T
                pB, zB = lens.project(XB)
                front = zB > 1e-9
                inside = front & (pB[:, 0] >= 0) & (pB[:, 1] >= 0) & (pB[:, 0] <= W - 1) & (pB[:, 1] <= H - 1)
                dB = np.full(len(pA), np.nan)
                dB[inside] = B.sample(B.to_output(pB[inside]))
                ok = np.isfinite(dB) & inside
                if ok.sum() < 200:
                    continue
                logB = corrected_log_units(corrections[B.frame][base], pB[ok], dB[ok])
                r = logB - np.log(zB[ok])
                if mth == "oracle-pair":
                    r = r - np.nanmedian(r)
                r = r[np.isfinite(r)]
                results[(gap, mth)].append(r)
                if gap == 1:
                    rr = np.linalg.norm(pA[ok] - lens.c, axis=1) / np.sqrt(lens.hd2)
                    detail[mth]["edge"].append(edgeA[ok][np.isfinite(logB - np.log(zB[ok]))])
                    detail[mth]["radius"].append(rr[np.isfinite(logB - np.log(zB[ok]))])
                    detail[mth]["depth"].append(dA[ok][np.isfinite(logB - np.log(zB[ok]))])
                    detail[mth]["r"].append(r)

    print(f"{D.name}: {len(kfs)} keyframes; landmark fit residual (median |log|): " +
          ", ".join(f"{k} {np.median(v):.3f}" for k, v in fit_stats.items() if v))
    print("\nPair disagreement |r| (log depth): median / p90 / share > 0.10, by keyframe gap")
    print(f"{'method':>12} " + " ".join(f"{'gap ' + str(g):>24}" for g in (1, 2, 4)))
    for mth in methods:
        cells = []
        for g in (1, 2, 4):
            v = np.concatenate(results[(g, mth)]) if results[(g, mth)] else np.array([])
            cells.append(f"{np.median(np.abs(v)):.3f} / {np.quantile(np.abs(v), .9):.3f} / {np.mean(np.abs(v) > .1):.2f}"
                         if len(v) else "-")
        print(f"{mth:>12} " + " ".join(f"{c:>24}" for c in cells))

    print("\nGap 1 breakdown, median |r| [signed median]:")
    for mth in ("kf-scale", "lm-grid"):
        d = detail[mth]
        if not d["r"]:
            continue
        r = np.concatenate(d["r"]); e = np.concatenate(d["edge"]); rad = np.concatenate(d["radius"])
        dep = np.concatenate(d["depth"])
        parts = [f"non-edge {np.median(np.abs(r[~e])):.3f}", f"edge {np.median(np.abs(r[e])):.3f} ({e.mean():.0%} of px)"]
        for lo, hi in [(0, .33), (.33, .66), (.66, 1.01)]:
            sel = (rad >= lo) & (rad < hi)
            parts.append(f"radius {lo:.2f}-{hi:.2f} {np.median(np.abs(r[sel])):.3f} [{np.median(r[sel]):+.3f}]")
        q = np.quantile(dep, [1 / 3, 2 / 3])
        for name, sel in [("near", dep < q[0]), ("mid", (dep >= q[0]) & (dep < q[1])), ("far", dep >= q[1])]:
            parts.append(f"{name} {np.median(np.abs(r[sel])):.3f} [{np.median(r[sel]):+.3f}]")
        print(f"  {mth}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
