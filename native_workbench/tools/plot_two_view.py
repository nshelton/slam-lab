"""Plot a slam-native-two-view --points CSV.

usage: plot_two_view.py POINTS.csv VIDEO OUT.png

Panels: frame A with track displacement (green = essential-matrix inlier,
red = outlier); inlier fraction versus distance from the image centre (lens
distortion shows up as outliers concentrated at large radius); top-down view
of the triangulated points and both cameras (x right, z forward, camera A at
the origin, baseline = 1); parallax versus depth.
"""
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def frame_image(video, index, width, height):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-vf", f"select=eq(n\\,{index})", "-vsync", "0",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def main():
    points_csv, video, out = sys.argv[1:4]
    header = open(points_csv).readline().split()
    frame_a, frame_b = int(header[2]), int(header[3])
    width, height = int(header[5]), int(header[6])
    fx, fy, cx, cy = map(float, header[8:12])
    cam_b = np.array(list(map(float, header[13:16])))
    d = np.genfromtxt(points_csv, delimiter=",", names=True, skip_header=1)
    inl = d["e_inlier"] == 1
    good = d["good"] == 1

    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    a = ax[0, 0]
    a.imshow(frame_image(video, frame_a, width, height))
    for mask, color in ((~inl, "red"), (inl, "lime")):
        a.quiver(d["xa"][mask], d["ya"][mask], d["xb"][mask] - d["xa"][mask], d["yb"][mask] - d["ya"][mask],
                 color=color, angles="xy", scale_units="xy", scale=1, width=0.0015)
    a.set_title(f"frame {frame_a} -> {frame_b}: {inl.sum()}/{len(d)} essential inliers (red = outlier)")
    a.axis("off")

    a = ax[0, 1]
    radius = np.hypot(d["xa"] - cx, d["ya"] - cy) / (0.5 * np.hypot(width, height))
    bins = np.linspace(0, 1, 11)
    which = np.digitize(radius, bins) - 1
    frac = [inl[which == k].mean() if (which == k).any() else np.nan for k in range(10)]
    count = [(which == k).sum() for k in range(10)]
    a.bar(bins[:-1] + 0.05, frac, width=0.09)
    for k in range(10):
        a.text(bins[k] + 0.05, 0.02, str(count[k]), ha="center", fontsize=8)
    a.set_xlabel("distance from image centre (1 = corner)")
    a.set_ylabel("essential-matrix inlier fraction")
    a.set_ylim(0, 1)
    a.set_title("inliers vs radius (bar labels = track count)")

    a = ax[1, 0]
    X, Z = d["X"][good], d["Z"][good]
    sc = a.scatter(X, Z, s=4, c=d["parallax_deg"][good], cmap="viridis", vmin=0, vmax=3)
    fig.colorbar(sc, ax=a, label="parallax (deg)")
    a.plot([0, cam_b[0]], [0, cam_b[2]], "r-", lw=2)
    a.plot(0, 0, "r^", ms=10, label="camera A")
    a.plot(cam_b[0], cam_b[2], "m^", ms=10, label="camera B")
    half = np.arctan(0.5 * width / fx)
    zmax = np.percentile(Z, 95) if len(Z) else 1
    a.plot([0, -np.tan(half) * zmax], [0, zmax], "k--", lw=0.8)
    a.plot([0, np.tan(half) * zmax], [0, zmax], "k--", lw=0.8)
    a.set_xlabel("x (right)")
    a.set_ylabel("z (forward)")
    a.set_aspect("equal")
    a.set_ylim(-1, np.percentile(Z, 98) if len(Z) else 1)
    a.legend()
    a.set_title("top view, baseline = 1 (dashed = camera A field of view)")

    a = ax[1, 1]
    a.scatter(d["Z"][good], d["parallax_deg"][good], s=4)
    a.set_xlabel("depth z (baseline = 1)")
    a.set_ylabel("parallax (deg)")
    a.set_xlim(0, np.percentile(Z, 98) if len(Z) else 1)
    a.set_title("parallax vs depth")

    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main()
