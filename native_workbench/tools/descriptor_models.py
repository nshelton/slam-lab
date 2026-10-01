"""Which per-landmark descriptor model best re-finds a landmark later?

Same inputs as descriptor_stability.py (bench with flow-sigma 0, export-tracks
and export-features). For every track with at least HISTORY frames of life, a
model is built from the observations in its first HISTORY frames; it then has
to pick the track's true detection among the detections of a later frame
(whole frame, and within r px of the true position = a pose-predicted window).

Models (score of a candidate descriptor x):
  first / latest     cosine to the first / last history descriptor
  mean               cosine to the normalised mean
  max-all            max cosine over every history descriptor (store everything)
  max-kf4            max cosine over 4 evenly spaced history descriptors
  diag-gauss         -sum((x - mu)^2 / var), var shrunk towards the global
                     within-track variance (a per-dimension variance model)
  ppca-k             probabilistic PCA per landmark: k principal directions
                     of its history + isotropic residual (Mahalanobis)
  whiten-mean/latest global within-track whitening (learned on the other half
                     of the video), then cosine to mean / latest
  pca64-mean         global PCA to 64 dims, then cosine to mean (compression)

Tracks are split by start frame: whitening and PCA are fit on one half and
every model is evaluated on the other half.

  .venv-cuda/bin/python descriptor_models.py tracks.csv features.sqlite3 [HISTORY]
"""
import sys

import numpy as np
import torch

from descriptor_stability import load_features, load_tracks

GAPS = [1, 5, 15, 30, 60]
RADII = [16, 32, 64]


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main(tracks_path, features_path, history=30):
    history = int(history)
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

    # Per-track descriptor stacks (only tracks with enough life to matter).
    tracks = []
    for a, b in zip(starts, ends):
        if frame[b - 1] - frame[a] < history:
            continue
        D = torch.stack([desc[f][j] for f, j in zip(frame[a:b], det[a:b])])
        tracks.append((frame[a:b], det[a:b], D))
    train = [t for t in tracks if t[0][0] < split]
    test = [t for t in tracks if t[0][0] >= split]

    def fit_global(ts):
        # Within-track covariance (each track centred on its own mean) and total covariance.
        within = torch.cat([D - D.mean(0) for _, _, D in ts])
        allD = torch.cat([D for _, _, D in ts])
        Sw = within.T @ within / len(within)
        mu = allD.mean(0)
        St = (allD - mu).T @ (allD - mu) / len(allD)
        e, V = torch.linalg.eigh(Sw + 1e-3 * torch.eye(256, device=Sw.device) * Sw.trace() / 256)
        W = V @ torch.diag(e.rsqrt()) @ V.T
        et, Vt = torch.linalg.eigh(St)
        P64 = Vt[:, -64:]
        return Sw.diag(), W, mu, P64, et.flip(0)

    # Evaluate on each half with the global transforms fit on the other half.
    results = {}
    names = None
    for eval_set, fit_set in [(test, train), (train, test)]:
        var_w, W, mu_all, P64, eig_total = fit_global(fit_set)
        for frames_t, dets_t, D in eval_set:
            hist = frames_t < frames_t[0] + history
            H = D[hist]
            n = len(H)
            last_f = frames_t[hist][-1]
            mu = H.mean(0)
            models = {}
            models["first"] = lambda X, h=H: X @ h[0]
            models["latest"] = lambda X, h=H: X @ h[-1]
            models["mean"] = lambda X, m=unit(mu): X @ m
            models["max-all"] = lambda X, h=H: (X @ h.T).max(1).values
            kf = H[torch.linspace(0, n - 1, min(4, n)).round().long()]
            models["max-kf4"] = lambda X, k=kf: (X @ k.T).max(1).values
            lam = 10.0  # prior strength in pseudo-observations
            var = (((H - mu) ** 2).sum(0) + lam * var_w) / (n + lam)
            models["diag-gauss"] = lambda X, m=mu, v=var: -(((X - m) ** 2) / v).sum(1)
            C = H - mu
            _, s, Vh = torch.linalg.svd(C, full_matrices=False)
            ev = s ** 2 / max(n - 1, 1)
            for k in (2, 5):
                if n <= k + 1:
                    continue
                # Residual variance from the rest of the spectrum, floored by the global within-track level.
                sigma2 = max(float(ev[k:].sum() / (256 - k)), float(var_w.mean()) * 0.5)
                B, lk = Vh[:k].T, ev[:k] + sigma2
                def ppca(X, m=mu, B=B, lk=lk, s2=sigma2):
                    d = X - m
                    p = d @ B
                    return -((p ** 2 / lk).sum(1) + ((d ** 2).sum(1) - (p ** 2).sum(1)) / s2)
                models[f"ppca-{k}"] = ppca
            HW = unit(H @ W)
            models["whiten-mean"] = lambda X, m=unit(HW.mean(0)), W=W: unit(X @ W) @ m
            models["whiten-latest"] = lambda X, l=HW[-1], W=W: unit(X @ W) @ l
            models["pca64-mean"] = lambda X, m=unit((H - mu_all) @ P64).mean(0), P=P64, c=mu_all: unit((X - c) @ P) @ unit(m)
            if names is None and "ppca-5" in models:
                names = list(models)
            # One query per gap bin: the first observation at least `gap` frames after the history.
            for gi, gap in enumerate(GAPS):
                hi = GAPS[gi + 1] if gi + 1 < len(GAPS) else 10 ** 9
                cand = np.nonzero((frames_t - last_f >= gap) & (frames_t - last_f < hi))[0]
                if not len(cand):
                    continue
                q = cand[0]
                f, j = int(frames_t[q]), int(dets_t[q])
                X = desc[f]
                near = torch.linalg.norm(xy[f] - xy[f][j], dim=1)
                for name, score in models.items():
                    s_all = score(X)
                    beats = s_all > s_all[j]
                    r = results.setdefault((gap, name), [])
                    r.append([int(beats.sum()) == 0] + [int((beats & (near <= R)).sum()) == 0 for R in RADII])
        print(f"eval half: {len(eval_set)} tracks; global within-track variance share of total: "
              f"{float(var_w.sum() / eig_total.sum()):.2f}; total variance in top 64 PCs: "
              f"{float(eig_total[:64].sum() / eig_total.sum()):.2f}")

    print(f"\nhistory = first {history} frames of each track; query = observation `gap` frames after the history.")
    print("Recall@1 of the true detection: whole frame (~2k detections) | within 16 / 32 / 64 px\n")
    print(f"{'model':>14} " + " ".join(f"{'gap ' + str(g) + '+':>24}" for g in GAPS))
    for name in names:
        cells = []
        for g in GAPS:
            v = np.array(results.get((g, name), []), float)
            cells.append(f"{v[:, 0].mean():.2f} | " + " ".join(f"{x:.2f}" for x in v[:, 1:].mean(0)) if len(v) else "-")
        print(f"{name:>14} " + " ".join(f"{c:>24}" for c in cells))
    print(f"{'n queries':>14} " + " ".join(f"{len(results.get((g, 'mean'), [])):>24}" for g in GAPS))


if __name__ == "__main__":
    main(*sys.argv[1:4])
