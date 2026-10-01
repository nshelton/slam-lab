"""Are SuperPoint descriptors stable enough over time to re-associate landmarks?

Uses the tracker's own tracks as ground-truth identity. Inputs come from
slam-native-tracking-bench with fusion off (flow-sigma 0), so every supported
track position is exactly a raw detection and joins to its descriptor:

  slam-native-tracking-bench VIDEO ENGINE 0 N flow-sigma 0 \
      export-tracks tracks.csv export-features features.sqlite3
  .venv-cuda/bin/python descriptor_stability.py tracks.csv features.sqlite3 OUT_PREFIX

Experiments (anchor = the track's first observed descriptor):
  1. drift: cosine(anchor, descriptor at age dt), and vs pixel displacement.
  2. discriminability: rank of the true detection among the target frame's
     detections, globally (a pure descriptor lookup) and within radius R of
     the true position (a pose-predicted search window).
  3. database scale: true cosine vs the best impostor among anchors of other
     tracks alive at the anchor frame (certainly distinct points), and the
     rank among every anchor in the database.
"""
import sqlite3
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGE_BINS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 200]
DISP_BINS = [0, 10, 25, 50, 100, 200, 400, 2000]
RADII = [16, 32, 64, 128, 256]


def load_features(path):
    db = sqlite3.connect(path)
    xy, desc = {}, {}
    for index, count, pts, d, enc in db.execute(
            "select frame_index, keypoint_count, keypoints_xy_f32, descriptors, descriptor_encoding from frames"):
        dtype = np.float16 if enc == "f16" else np.float32
        xy[index] = torch.from_numpy(np.frombuffer(pts, np.float32).reshape(count, 2).copy()).cuda()
        desc[index] = torch.from_numpy(np.frombuffer(d, dtype).reshape(count, 256).astype(np.float32)).cuda()
    return xy, desc


def load_tracks(path):
    a = np.loadtxt(path, delimiter=",", comments="#", skiprows=2, usecols=(0, 2, 3, 4))
    order = np.lexsort((a[:, 0], a[:, 1]))  # by track, then frame
    return a[order, 0].astype(np.int64), a[order, 1].astype(np.int64), a[order, 2:4].astype(np.float32)


def quantile(v, q):
    return float(np.quantile(v, q)) if len(v) else float("nan")


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
    print(f"joined {ok.mean():.4f} of {len(frame)} observations within 0.01 px")
    frame, track, pos, det = frame[ok], track[ok], pos[ok], det[ok]

    starts = np.r_[0, np.nonzero(np.diff(track))[0] + 1]
    ends = np.r_[starts[1:], len(track)]
    tid = track[starts]
    first_f, last_f = frame[starts], frame[ends - 1]
    life = last_f - first_f + 1
    print(f"tracks {len(tid)}; lifespan median {np.median(life):.0f} frames, >=30: {(life >= 30).sum()}, "
          f">=90: {(life >= 90).sum()}, >=200: {(life >= 200).sum()}")

    # One (anchor, observation) sample per track per age bin; anchor = first observation.
    rng = np.random.default_rng(0)
    S = []  # track index, anchor row, target row
    for t, (a, b) in enumerate(zip(starts, ends)):
        ages = frame[a:b] - frame[a]
        for lo, hi in zip(AGE_BINS, AGE_BINS[1:] + [10 ** 9]):
            cand = np.nonzero((ages >= lo) & (ages < hi))[0]
            if len(cand):
                S.append((t, a, a + cand[rng.integers(len(cand))]))
    S = np.array(S)
    r0, r1 = S[:, 1], S[:, 2]
    age = frame[r1] - frame[r0]
    disp = np.linalg.norm(pos[r1] - pos[r0], axis=1)

    true_cos = np.empty(len(S)); impostor = np.empty(len(S)); rank = np.empty(len(S), np.int64)
    rank_r = {r: np.empty(len(S), np.int64) for r in RADII}
    for i, (a, b) in enumerate(zip(r0, r1)):
        sims = desc[frame[b]] @ desc[frame[a]][det[a]]
        t = sims[det[b]].clone()
        sims[det[b]] = -2
        beats = sims >= t
        true_cos[i], impostor[i], rank[i] = t.item(), sims.max().item(), int(beats.sum())
        near = torch.linalg.norm(xy[frame[b]] - xy[frame[b]][det[b]], dim=1)
        for r in RADII:
            rank_r[r][i] = int((beats & (near <= r)).sum())

    def row(mask):
        return (f"{mask.sum():6d} {np.median(true_cos[mask]):7.3f} {quantile(true_cos[mask], .1):7.3f} "
                f"{np.median(impostor[mask]):8.3f} {np.mean(rank[mask] == 0):7.3f} "
                + " ".join(f"{np.mean(rank_r[r][mask] == 0):6.3f}" for r in RADII))

    header = "     n cos_med cos_p10 imp_med R@1_all " + " ".join(f"{'r' + str(r):>6}" for r in RADII)
    print("\nAnchor (first obs) vs observation at age dt frames. imp_med = best impostor in that frame; "
          "R@1 = anchor's nearest descriptor is the true detection (whole frame / within r px):")
    print("age     " + header)
    table = []
    for lo, hi in zip(AGE_BINS, AGE_BINS[1:] + [10 ** 9]):
        m = (age >= lo) & (age < hi)
        if m.any():
            print(f"{lo:>4}+   " + row(m))
            table.append((lo, np.mean(rank[m] == 0), [np.mean(rank_r[r][m] == 0) for r in RADII]))
    print("\nBy image displacement from the anchor (px):")
    print("disp    " + header)
    for lo, hi in zip(DISP_BINS, DISP_BINS[1:]):
        m = (disp >= lo) & (disp < hi)
        if m.any():
            print(f"{lo:>4}+   " + row(m))

    # Anchor choice on long tracks: query = last obs; anchors from the first half.
    firsts, latests, means = [], [], []
    for a, b in zip(starts, ends):
        if b - a < 40:
            continue
        D = torch.stack([desc[f][j] for f, j in zip(frame[a:b], det[a:b])])
        h = len(D) // 2
        m = D[:h].mean(0)
        q = D[-1]
        firsts.append((D[0] @ q).item()); latests.append((D[h - 1] @ q).item()); means.append((m / m.norm() @ q).item())
    print(f"\nAnchor choice, {len(firsts)} tracks >= 40 obs, query = last obs, anchor from first half "
          f"(median / p10): first {np.median(firsts):.3f}/{quantile(firsts, .1):.3f}  "
          f"latest {np.median(latests):.3f}/{quantile(latests, .1):.3f}  "
          f"mean {np.median(means):.3f}/{quantile(means, .1):.3f}")

    # Database scale. Anchors = first descriptor of every track.
    anchors = torch.stack([desc[f][j] for f, j in zip(first_f, det[starts])])
    first_t = torch.from_numpy(first_f).cuda(); last_t = torch.from_numpy(last_f).cuda()
    q_mask = age >= 30
    res = []
    for t, a, b in S[q_mask]:
        sims = anchors @ desc[frame[b]][det[b]]
        true = sims[t].item()
        sims[t] = -2
        alive = (first_t <= frame[a]) & (last_t >= frame[a])
        in_db = first_t < frame[b]
        res.append((true, sims[alive].max().item(), int(alive.sum()) - 1,
                    int(((sims >= true) & in_db).sum()), int(in_db.sum())))
    res = np.array(res)
    print(f"\nDatabase scale ({len(res)} queries at age >= 30): median {np.median(res[:, 2]):.0f} co-alive distinct "
          f"impostors, median DB size {np.median(res[:, 4]):.0f} anchors")
    print(f"  true beats every co-alive impostor: {np.mean(res[:, 0] > res[:, 1]):.3f}"
          f" (margin 0.05: {np.mean(res[:, 0] > res[:, 1] + 0.05):.3f})")
    print(f"  rank in whole DB: R@1 {np.mean(res[:, 3] == 0):.3f}  R@5 {np.mean(res[:, 3] < 5):.3f}  "
          f"R@50 {np.mean(res[:, 3] < 50):.3f}   (DB may hold re-births of the same point under other IDs)")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for lo, hi, c in [(1, 2, "C0"), (5, 8, "C1"), (20, 30, "C2"), (60, 90, "C3"), (120, 10 ** 9, "C4")]:
        m = (age >= lo) & (age < hi)
        ax[0].hist(true_cos[m], bins=70, range=(0.3, 1), histtype="step", density=True, color=c,
                   label=f"true, age {lo}{'+' if hi > 10 ** 8 else f'-{hi - 1}'}")
    ax[0].hist(impostor, bins=70, range=(0.3, 1), density=True, alpha=0.3, color="k",
               label="best impostor in target frame")
    ax[0].set(xlabel="cosine to anchor", title="Drift vs in-frame impostors"); ax[0].legend(fontsize=8)
    ages_x = [str(t[0]) for t in table]
    ax[1].plot(ages_x, [t[1] for t in table], "k-o", label="whole frame (~2k detections)")
    for k, r in enumerate(RADII):
        ax[1].plot(ages_x, [t[2][k] for t in table], "-o", label=f"within {r} px")
    ax[1].set(xlabel="age (frames, bin start)", ylabel="recall@1", title="Anchor retrieves the true detection",
              ylim=(0, 1.02)); ax[1].legend(fontsize=8)
    ax[2].hist(res[:, 0], bins=60, range=(0.3, 1), density=True, alpha=0.5, label="true (age >= 30)")
    ax[2].hist(res[:, 1], bins=60, range=(0.3, 1), density=True, alpha=0.5,
               label=f"best of ~{np.median(res[:, 2]):.0f} co-alive anchors")
    ax[2].set(xlabel="cosine", title="Database: true vs best distinct impostor"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out + "_descriptor_stability.png", dpi=110)
    print("wrote", out + "_descriptor_stability.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
