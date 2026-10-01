"""How far does a constant-velocity pose prediction drift across an occlusion,
and does projection + the latest descriptor re-find the old landmarks?

Inputs (same run): bench tracks CSV (flow-sigma 0) + feature cache, and the VO
trajectory + map from slam-native-vo-tracks on those tracks.

A. Simulated occlusion on well-tracked stretches: at frame t the camera is
   "blinded" until t+g. The pose at t+g is predicted from poses <= t
   (constant position, or constant velocity averaged over k frames) and
   compared with the VO pose at t+g, in degrees and in reprojection pixels of
   the landmarks known at t. Then every such landmark searches frame t+g's
   detections within r px of its predicted projection and takes the best
   descriptor (its latest descriptor at or before t). Ground truth: the
   landmark's own track observed at t+g.
B. Real column events, where the VO lost tracking and restarted in a new
   segment: carry the old segment's last pose forward with the motion model,
   project all of that segment's landmarks into frames after the event, match
   by radius + descriptor, then PnP-RANSAC. Many inliers vs a shuffled-
   descriptor null = relocalized into the old map.

  .venv-cuda/bin/python pose_propagation.py tracks.csv features.sqlite3 traj.csv map.csv HFOV K1
"""
import sys

import cv2
import numpy as np
import torch

from descriptor_stability import load_features, load_tracks

GAPS = [5, 10, 15, 30, 45]
RADII = [16, 32, 64, 128]
MIN_COS = 0.7


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(R):
    return cv2.Rodrigues(R)[0].ravel()


def rotmat(v):
    return cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]


def load_traj(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    poses = {}
    for row in a:
        f, seg = int(row[0]), int(row[2])
        Rcw = quat_to_R(*row[7:11])  # camera -> world
        poses[f] = (seg, Rcw.T, row[4:7].copy())  # (segment, R world->camera, centre)
    return poses


def predict(poses, t, g, k):
    """Pose at t+g from poses <= t. k=0: constant position; else CV over k frames."""
    seg, R, c = poses[t]
    if k == 0:
        return R, c
    s0, R0, c0 = poses[t - k]
    w = rotvec(R @ R0.T) / k  # per-frame rotation increment (world->camera, left)
    return rotmat(w * g) @ R, c + (c - c0) / k * g


def project(K, R, c, X):
    Xc = (X - c) @ R.T
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.maximum(z, 1e-9)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def undistort(xy, w, h, k1):
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    hd2 = 0.25 * (w * w + h * h)
    d = xy - [cx, cy]
    return [cx, cy] + d / (1 + k1 * (d ** 2).sum(1, keepdims=True) / hd2)


def angle_deg(Ra, Rb):
    return np.degrees(np.linalg.norm(rotvec(Ra @ Rb.T)))


def main(tracks_path, features_path, traj_path, map_path, hfov, k1):
    hfov, k1 = float(hfov), float(k1)
    W, H = 1920, 1080
    f = 0.5 * W / np.tan(np.radians(hfov) / 2)
    K = np.array([[f, 0, 0.5 * (W - 1)], [0, f, 0.5 * (H - 1)], [0, 0, 1]])
    xy, desc = load_features(features_path)
    und = {fr: torch.from_numpy(undistort(v.cpu().numpy().astype(np.float64), W, H, k1)).float().cuda()
           for fr, v in xy.items()}
    frame, track, pos = load_tracks(tracks_path)
    det = np.full(len(frame), -1, np.int64)
    dist = np.full(len(frame), np.inf, np.float32)
    for fr in np.unique(frame):
        rows = np.nonzero(frame == fr)[0]
        d, j = torch.cdist(torch.from_numpy(pos[rows]).cuda(), xy[fr]).min(dim=1)
        det[rows], dist[rows] = j.cpu().numpy(), d.cpu().numpy()
    ok = dist < 0.01
    frame, track, det = frame[ok], track[ok], det[ok]
    obs = {}  # track -> (frames, dets), frames sorted
    starts = np.r_[0, np.nonzero(np.diff(track))[0] + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(track)]):
        obs[int(track[a])] = (frame[a:b], det[a:b])

    poses = load_traj(traj_path)
    m = np.loadtxt(map_path, delimiter=",", skiprows=1)
    lm_id, lm_seg, lm_X = m[:, 0].astype(np.int64), m[:, 1].astype(int), m[:, 2:5]
    lm_first, lm_last = m[:, 10].astype(int), m[:, 11].astype(int)
    keep = np.array([i in obs for i in lm_id])
    lm_id, lm_seg, lm_X, lm_first, lm_last = lm_id[keep], lm_seg[keep], lm_X[keep], lm_first[keep], lm_last[keep]
    segs = sorted(set(s for s, _, _ in poses.values()))
    seg_frames = {s: sorted(fr for fr, p in poses.items() if p[0] == s) for s in segs}
    print("segments (first-last posed frame, landmarks): " + ", ".join(
        f"{s}: {v[0]}-{v[-1]} ({(lm_seg == s).sum()})" for s, v in seg_frames.items()))

    def latest_desc(i, t):
        fr, dt = obs[int(lm_id[i])]
        n = np.searchsorted(fr, t, side="right") - 1
        return (desc[int(fr[n])][int(dt[n])], int(fr[n])) if n >= 0 else (None, None)

    def observed_at(i, t):
        fr, dt = obs[int(lm_id[i])]
        n = np.searchsorted(fr, t)
        return int(dt[n]) if n < len(fr) and fr[n] == t else -1

    def associate(idx, uv, z, t_target, anchors, radius):
        """Best descriptor within radius of each projection -> (landmark, det, cos) accepted."""
        P = und[t_target]
        D = desc[t_target]
        out = []
        for n, i in enumerate(idx):
            if z[n] <= 0 or not (0 <= uv[n, 0] < W and 0 <= uv[n, 1] < H):
                continue
            near = torch.nonzero(torch.linalg.norm(P - torch.tensor(uv[n], dtype=torch.float32, device="cuda"), dim=1)
                                 <= radius).ravel()
            if not len(near):
                continue
            s = D[near] @ anchors[n]
            b = int(s.argmax())
            if float(s[b]) >= MIN_COS:
                out.append((n, int(near[b]), float(s[b])))
        return out

    # ---------------- A: simulated occlusion ----------------
    print(f"\nA. Simulated occlusion (camera blinded from t to t+g), k = frames of velocity history; "
          f"hfov {hfov}, k1 {k1}")
    motion_models = [("const-position", 0), ("const-vel k=1", 1), ("const-vel k=5", 5), ("const-vel k=10", 10)]
    stats = {}
    assoc = {}
    for s, frs in seg_frames.items():
        frs_set = set(frs)
        for t in frs[10::5]:
            for g in GAPS:
                if t + g not in frs_set or t - 10 not in frs_set:
                    continue
                known = np.nonzero((lm_seg == s) & (lm_first <= t) & (lm_last >= t - 10))[0]
                if len(known) < 20:
                    continue
                _, Rt, ct = poses[t + g]
                uv_true, z_true = project(K, Rt, ct, lm_X[known])
                vis = (z_true > 0) & (uv_true[:, 0] >= 0) & (uv_true[:, 0] < W) & (uv_true[:, 1] >= 0) & (uv_true[:, 1] < H)
                for name, k in motion_models:
                    Rp, cp = predict(poses, t, g, k)
                    uv, z = project(K, Rp, cp, lm_X[known])
                    px = np.linalg.norm(uv - uv_true, axis=1)[vis & (z > 0)]
                    if len(px):
                        stats.setdefault((name, g), []).append((angle_deg(Rp, Rt), np.median(px)))
                    if name != "const-vel k=5" and name != "const-position":
                        continue
                    anchors, ok_i = [], []
                    for n, i in enumerate(known):
                        a, _ = latest_desc(i, t)
                        anchors.append(a if a is not None else torch.zeros(256, device="cuda"))
                    anchors = torch.stack(anchors)
                    gt = np.array([observed_at(i, t + g) for i in known])
                    for r in RADII:
                        acc = associate(known, uv, z, t + g, anchors, r)
                        has_gt = int((gt >= 0).sum())
                        correct = sum(1 for n, j, _ in acc if gt[n] == j)
                        with_gt = sum(1 for n, j, _ in acc if gt[n] >= 0)
                        e = assoc.setdefault((name, g, r), np.zeros(4))
                        e += [has_gt, correct, with_gt, len(acc)]
                    # Oracle: the VO's own pose at t+g.
                    if name == "const-position":
                        for r in RADII:
                            acc = associate(known, uv_true, z_true, t + g, anchors, r)
                            correct = sum(1 for n, j, _ in acc if gt[n] == j)
                            with_gt = sum(1 for n, j, _ in acc if gt[n] >= 0)
                            e = assoc.setdefault(("oracle VO pose", g, r), np.zeros(4))
                            e += [int((gt >= 0).sum()), correct, with_gt, len(acc)]
    print("Pose prediction error at t+g vs VO pose: median rotation (deg) | median landmark reprojection shift (px) "
          "[p90 over samples]")
    print(f"{'model':>16} " + " ".join(f"{'g=' + str(g) + ' (' + format(g / 30, '.2f') + 's)':>24}" for g in GAPS))
    for name, _ in motion_models:
        cells = []
        for g in GAPS:
            v = np.array(stats.get((name, g), []))
            cells.append(f"{np.median(v[:, 0]):5.2f} | {np.median(v[:, 1]):6.1f} [{np.quantile(v[:, 1], .9):6.1f}]"
                         if len(v) else "-")
        print(f"{name:>16} " + " ".join(f"{c:>24}" for c in cells))
    print(f"{'samples':>16} " + " ".join(f"{len(stats.get(('const-position', g), [])):>24}" for g in GAPS))

    print(f"\nAssociation: projection gate r px + best latest-descriptor (cos >= {MIN_COS}). "
          "recall = correct / landmarks whose track is observed at t+g; precision = correct / accepted with GT; "
          "accepted/GT counts all accepted incl. landmarks without GT")
    for name in ["oracle VO pose", "const-vel k=5", "const-position"]:
        print(f"  {name}")
        for r in RADII:
            cells = []
            for g in GAPS:
                e = assoc.get((name, g, r))
                cells.append(f"R {e[1] / max(e[0], 1):.2f} P {e[1] / max(e[2], 1):.2f} x{e[3] / max(e[0], 1):.1f}"
                             if e is not None else "-")
            print(f"    r={r:>3}  " + " ".join(f"{c:>20}" for c in cells))

    # ---------------- B: real column events ----------------
    print("\nB. Real events: old segment's last pose carried forward (const-vel k=5), all old landmarks projected, "
          f"radius gate + descriptor, then PnP-RANSAC (4 px). null = descriptors shuffled among landmarks.")
    rng = np.random.default_rng(0)
    for s in segs[:-1]:
        frs = seg_frames[s]
        if len(frs) < 30:
            continue
        a = frs[-1] - 3  # a few frames before the VO declared loss
        while a not in poses or a - 5 not in poses:
            a -= 1
        idx = np.nonzero(lm_seg == s)[0]
        anchors = []
        for i in idx:
            d_, _ = latest_desc(i, a)
            anchors.append(d_ if d_ is not None else torch.zeros(256, device="cuda"))
        anchors = torch.stack(anchors)
        shuffled = anchors[torch.from_numpy(rng.permutation(len(idx))).cuda()]
        print(f"  segment {s}: last pose used frame {a} ({len(idx)} landmarks)")
        for g in [5, 10, 15, 20, 30, 45, 60, 90]:
            b = a + g
            if b not in xy:
                continue
            Rp, cp = predict(poses, a, g, 5)
            uv, z = project(K, Rp, cp, lm_X[idx])
            line = f"    +{g:>2} (frame {b}, now seg {poses[b][0] if b in poses else '-'}):"
            for r in [64, 256]:
                for label, A in [("", anchors), ("null ", shuffled)]:
                    acc = associate(idx, uv, z, b, A, r)
                    inl, shift = 0, float("nan")
                    if len(acc) >= 6:
                        obj = lm_X[idx][[n for n, _, _ in acc]].astype(np.float64)
                        img = und[b][[j for _, j, _ in acc]].cpu().numpy().astype(np.float64)
                        rvec0 = cv2.Rodrigues(Rp)[0]
                        tvec0 = (-Rp @ cp).reshape(3, 1)
                        okp, rv, tv, inliers = cv2.solvePnPRansac(obj, img, K, None, rvec0.copy(), tvec0.copy(), True,
                                                                  2000, 4.0, 0.999)
                        if okp and inliers is not None:
                            inl = len(inliers)
                            R_est = cv2.Rodrigues(rv)[0]
                            c_est = (-R_est.T @ tv).ravel()
                            u_est, _ = project(K, R_est, c_est, lm_X[idx])
                            vis = z > 0
                            shift = float(np.median(np.linalg.norm(u_est - uv, axis=1)[vis]))
                    if label == "":
                        line += f"  r{r}: {len(acc):4d} cand, {inl:4d} PnP inl, pred off {shift:6.1f}px"
                    else:
                        line += f" (null {inl})"
            print(line)


if __name__ == "__main__":
    main(*sys.argv[1:7])
