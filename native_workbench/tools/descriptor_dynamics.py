"""Is SuperPoint descriptor drift predictable (directional) or a random walk?

Same inputs as descriptor_stability.py. Three checks on the tracks:
  1. Mean squared displacement MSD(tau) = E|x(t+tau) - x(t)|^2 over long
     tracks. MSD ~ tau^2 means ballistic (directional) drift that a
     constant-velocity model can extrapolate; MSD ~ tau means a random walk.
     The intercept is the per-observation noise (2 sigma^2).
  2. Direction persistence: cosine between the past drift (over the history)
     and the future drift (history end -> query).
  3. Prediction: constant-position and constant-velocity Kalman filters on the
     descriptor (isotropic, so one scalar 2x2 filter per track drives all 256
     dims), tuned on one half of the video and evaluated on the other, vs the
     latest descriptor. Score: cosine to the true future descriptor and
     recall@1 of the true detection (whole frame / within 16 px).

  .venv-cuda/bin/python descriptor_dynamics.py tracks.csv features.sqlite3 OUT_PREFIX
"""
import sys

import matplotlib
import numpy as np
import torch

from descriptor_stability import load_features, load_tracks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY = 30
GAPS = [1, 5, 15, 30, 60]


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def kalman(times, D, q, r, velocity, t_query):
    """Isotropic Kalman filter over descriptor observations D (n x 256) at `times`.

    velocity=False: random-walk level model (process noise q per frame).
    velocity=True: constant velocity, white-acceleration process noise q.
    All dimensions share one scalar covariance, so gains are scalars.
    Returns the prediction at t_query (unit-normalised)."""
    x = D[0].clone()
    v = torch.zeros_like(x)
    P = np.array([[r, 0.0], [0.0, 1e-3]]) if velocity else np.array([[r]])
    t_prev = times[0]
    for t, z in zip(times[1:], D[1:]):
        dt = float(t - t_prev)
        t_prev = t
        if velocity:
            F = np.array([[1, dt], [0, 1]])
            Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            P = F @ P @ F.T + Q
            x = x + dt * v
            S = P[0, 0] + r
            K = P[:, 0] / S
            innov = z - x
            x = x + K[0] * innov
            v = v + K[1] * innov
            P = P - np.outer(K, P[0, :])
        else:
            P = P + q * dt
            K = P[0, 0] / (P[0, 0] + r)
            x = x + K * (z - x)
            P = P * (1 - K)
    dt = float(t_query - t_prev)
    return unit(x + dt * v) if velocity else unit(x)


def main(tracks_path, features_path, out):
    xy, desc = load_features(features_path)
    frame, track, pos = load_tracks(tracks_path)
    det = np.full(len(frame), -1, np.int64)
    dist = np.full(len(frame), np.inf, np.float32)
    for f in np.unique(frame):
        rows = np.nonzero(frame == f)[0]
        d, j = torch.cdist(torch.from_numpy(pos[rows]).cuda(), xy[f]).min(dim=1)
        det[rows], dist[rows] = j.cpu().numpy(), d.cpu().numpy()
    ok = dist < 0.01
    frame, track, det = frame[ok], track[ok], det[ok]
    starts = np.r_[0, np.nonzero(np.diff(track))[0] + 1]
    ends = np.r_[starts[1:], len(track)]
    split = int(np.median(frame))
    tracks = []
    for a, b in zip(starts, ends):
        if frame[b - 1] - frame[a] < HISTORY:
            continue
        tracks.append((frame[a:b], det[a:b], torch.stack([desc[f][j] for f, j in zip(frame[a:b], det[a:b])])))
    print(f"{len(tracks)} tracks with >= {HISTORY} frames of life")

    # 1. MSD vs lag over tracks with >= 60 observations.
    lags = np.arange(1, 91)
    msd_sum, msd_n = np.zeros(len(lags)), np.zeros(len(lags))
    for f, _, D in tracks:
        if len(D) < 60:
            continue
        G = 2 - 2 * (D @ D.T)  # squared chord distance between all pairs
        lag = torch.from_numpy(f[None, :] - f[:, None]).cuda()
        for k, L in enumerate(lags):
            m = lag == int(L)
            if m.any():
                msd_sum[k] += float(G[m].sum()); msd_n[k] += int(m.sum())
    msd = msd_sum / np.maximum(msd_n, 1)
    # Fit MSD = noise + a * tau^alpha on lags 1..60.
    best = None
    for noise in np.linspace(0, msd[0], 41)[:-1]:
        sel = (lags <= 60) & (msd_n > 0)
        yv = np.log(msd[sel] - noise); xv = np.log(lags[sel])
        A = np.vstack([xv, np.ones_like(xv)]).T
        coef, res, *_ = np.linalg.lstsq(A, yv, rcond=None)
        err = float(((A @ coef - yv) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, noise, coef[0], np.exp(coef[1]))
    _, noise, alpha, scale = best
    print(f"\nMSD fit (lags 1-60): noise floor {noise:.4f}, exponent alpha = {alpha:.2f} "
          f"(1 = random walk, 2 = ballistic); MSD at lag 1/10/30/60: "
          + " / ".join(f"{msd[L - 1]:.3f}" for L in (1, 10, 30, 60)))

    # 2 + 3. Direction persistence and prediction, per half.
    rng = np.random.default_rng(0)
    persist = {g: [] for g in GAPS}
    persist_null = {g: [] for g in GAPS}
    queries = {0: [], 1: []}  # half -> (times, D_hist, t_query, frame, det)
    for f, j, D in tracks:
        half = int(f[0] >= split)
        hist = f < f[0] + HISTORY
        th, Dh = f[hist], D[hist]
        if len(Dh) < 10:
            continue
        past = Dh[-5:].mean(0) - Dh[:5].mean(0)
        for gi, gap in enumerate(GAPS):
            hi = GAPS[gi + 1] if gi + 1 < len(GAPS) else 10 ** 9
            cand = np.nonzero((f - th[-1] >= gap) & (f - th[-1] < hi))[0]
            if not len(cand):
                continue
            q = cand[0]
            future = D[q] - Dh[-5:].mean(0)
            persist[gap].append(float(unit(past) @ unit(future)))
            other = tracks[rng.integers(len(tracks))][2]
            persist_null[gap].append(float(unit(past) @ unit(other[-1] - other[0])))
            queries[half].append((gap, th, Dh, int(f[q]), int(j[q]), D[q]))
    print("\nDirection persistence: cosine(past drift over history, future drift), median [p25, p75]; "
          "null = past vs another track's drift")
    for g in GAPS:
        v, n0 = np.array(persist[g]), np.array(persist_null[g])
        print(f"  gap {g:>2}+: {np.median(v):+.3f} [{np.quantile(v, .25):+.3f}, {np.quantile(v, .75):+.3f}]"
              f"   null {np.median(n0):+.3f}   n={len(v)}")

    def evaluate(qs, predictor):
        cos, r_all, r16 = {g: [] for g in GAPS}, {g: [] for g in GAPS}, {g: [] for g in GAPS}
        for gap, th, Dh, fq, jq, true in qs:
            p = predictor(th, Dh, fq)
            cos[gap].append(float(p @ true))
            s = desc[fq] @ p
            beats = s > s[jq]
            near = torch.linalg.norm(xy[fq] - xy[fq][jq], dim=1) <= 16
            r_all[gap].append(int(beats.sum()) == 0)
            r16[gap].append(int((beats & near).sum()) == 0)
        return {g: (np.mean(cos[g]), np.mean(r_all[g]), np.mean(r16[g])) for g in GAPS}

    def tune(half, velocity):
        # Tune on a subsample of the other half, objective: mean cosine at gaps >= 15.
        qs = [q for q in queries[half] if q[0] >= 15]
        qs = [qs[i] for i in rng.permutation(len(qs))[:600]]
        best = None
        r_grid = [1e-3, 3e-3, 1e-2, 3e-2]
        q_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3] if velocity else [1e-4, 1e-3, 1e-2, 1e-1]
        for r in r_grid:
            for q in q_grid:
                c = np.mean([float(kalman(th, Dh, q, r, velocity, fq) @ t) for _, th, Dh, fq, _, t in qs])
                if best is None or c > best[0]:
                    best = (c, q, r)
        return best[1], best[2]

    rows = {}
    for test_half in (0, 1):
        train_half = 1 - test_half
        qcp, rcp = tune(train_half, False)
        qcv, rcv = tune(train_half, True)
        print(f"test half {test_half}: tuned constant-position q={qcp:g} r={rcp:g}; constant-velocity q={qcv:g} r={rcv:g}")
        preds = {
            "latest": lambda th, Dh, fq: Dh[-1],
            "mean last 5": lambda th, Dh, fq: unit(Dh[-5:].mean(0)),
            "KF const-position": lambda th, Dh, fq, q=qcp, r=rcp: kalman(th, Dh, q, r, False, fq),
            "KF const-velocity": lambda th, Dh, fq, q=qcv, r=rcv: kalman(th, Dh, q, r, True, fq),
        }
        for name, p in preds.items():
            res = evaluate(queries[test_half], p)
            for g in GAPS:
                n = sum(1 for q in queries[test_half] if q[0] == g)
                rows.setdefault(name, {}).setdefault(g, []).append((n, res[g]))
    print("\nPrediction (held-out half, pooled): cosine to true future descriptor | recall@1 whole frame | within 16 px")
    print(f"{'predictor':>18} " + " ".join(f"{'gap ' + str(g) + '+':>22}" for g in GAPS))
    for name, per_gap in rows.items():
        cells = []
        for g in GAPS:
            n = np.array([a for a, _ in per_gap[g]], float)
            v = np.array([b for _, b in per_gap[g]])
            m = (v * n[:, None]).sum(0) / n.sum()
            cells.append(f"{m[0]:.3f} | {m[1]:.2f} | {m[2]:.2f}")
        print(f"{name:>18} " + " ".join(f"{c:>22}" for c in cells))

    # Plots: MSD, persistence, and a few descriptor trajectories in each track's own top-2 PCs.
    long = sorted(tracks, key=lambda t: -len(t[2]))[:6]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(2, 4, 1)
    ax.loglog(lags, msd, "k.", label="measured")
    ax.loglog(lags, noise + scale * lags ** alpha, "r-", label=f"noise + a·τ^{alpha:.2f}")
    ax.loglog(lags, noise + scale * lags, "b--", lw=0.8, label="random walk (τ¹)")
    ax.loglog(lags, noise + scale * lags ** 2, "g--", lw=0.8, label="ballistic (τ²)")
    ax.set(xlabel="lag τ (frames)", ylabel="MSD (2 − 2cos)", title="Descriptor mean squared displacement",
           ylim=(msd.min() * 0.5, msd.max() * 2)); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 4, 5)
    for g in GAPS:
        ax.hist(persist[g], bins=40, range=(-1, 1), histtype="step", density=True, label=f"gap {g}+")
    ax.hist(sum(persist_null.values(), []), bins=40, range=(-1, 1), density=True, alpha=0.25, color="k", label="null")
    ax.set(xlabel="cos(past drift, future drift)", title="Drift direction persistence"); ax.legend(fontsize=7)
    for k, (f, _, D) in enumerate(long):
        ax = fig.add_subplot(2, 4, [2, 3, 4, 6, 7, 8][k])
        C = D - D.mean(0)
        _, s, Vh = torch.linalg.svd(C, full_matrices=False)
        P = (C @ Vh[:2].T).cpu().numpy()
        share = float((s[:2] ** 2).sum() / (s ** 2).sum())
        ax.plot(P[:, 0], P[:, 1], "-", color="0.8", lw=0.6)
        sc = ax.scatter(P[:, 0], P[:, 1], c=f, s=8, cmap="viridis")
        ax.set_title(f"track: {len(D)} obs, frames {f[0]}-{f[-1]}, top-2 PCs = {share:.0%} var", fontsize=9)
        plt.colorbar(sc, ax=ax, label="frame")
    fig.tight_layout()
    fig.savefig(out + "_descriptor_dynamics.png", dpi=105)
    print("wrote", out + "_descriptor_dynamics.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
