#!/usr/bin/env python3
"""Score landmark depths against TUM ground-truth depth (SPLINE_BA.md, milestone 0).

Input: the CSV of `slam-native-tracking-bench ... export-landmarks PATH`
(optionally with `resolve-landmarks 1`): one row per (landmark, keyframe)
sighting with the observed pixel as tracked (x, y) and one or more depth
columns (`depth_keyframes`, the VO's own landmark; `depth_fused_K` /
`depth_raw_K`, the landmark re-solved from K of its every-frame observations
with the poses fixed). Truth: the TUM depth image nearest in time to that frame
(within 20 ms; registered to RGB), sampled at the tracked pixel. Pixels with no
depth and pixels whose 3x3 neighbourhood varies by more than 5% (depth edges)
are skipped.

Monocular depth has an unknown scale, so errors are measured after one scale
per frame (the median log ratio over that frame's landmarks, per depth column).
Frames with fewer than 20 scored landmarks are skipped. All depth columns are
scored on the same (landmark, keyframe) pairs: those where every column has a
positive finite depth (a variant with too few observations writes nan).

Reported per depth column:
  err        median |log z - log z_true - scale| (~relative error)
  <5%, <10%  share of pairs within that relative error
  pairs      (landmark, keyframe) pairs scored, and the frames they come from
  ratio      err / err of depth_keyframes (the gate in SPLINE_BA.md: <= 0.85)

usage:
  .venv-cuda/bin/python landmark_depth_vs_gt.py SEQUENCE_DIR LANDMARKS.csv [LANDMARKS.csv ...]
  (each CSV is scored on its own; --markdown prints a table row per CSV)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_TIME_GAP = 0.02      # s, depth image to video frame
EDGE_VARIATION = 0.05    # (max - min) / centre over the 3x3 neighbourhood
MIN_LANDMARKS = 20       # per frame, for the scale
DEPTH_SCALE = 5000.0     # TUM: 16-bit PNG, 5000 units per metre


def load_truth(sequence: Path):
    with open(sequence / "frames.csv") as f:
        frame_time = {int(r["frame_index"]): float(r["timestamp"]) for r in csv.DictReader(f)}
    times, files = [], []
    for line in (sequence / "depth.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            t, name = line.split()
            times.append(float(t))
            files.append(sequence / name)
    return frame_time, np.array(times), files


def load_rows(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c.startswith("depth_")]
        rows = list(reader)
    return rows, columns


def sample_depth(truth: np.ndarray, x: float, y: float) -> float:
    """Depth at the nearest pixel, or nan at the border, without depth, or on an edge."""
    col, row = int(round(x)), int(round(y))
    if row < 1 or col < 1 or row >= truth.shape[0] - 1 or col >= truth.shape[1] - 1:
        return np.nan
    patch = truth[row - 1:row + 2, col - 1:col + 2]
    centre = truth[row, col]
    if centre <= 0 or (patch <= 0).any():
        return np.nan
    if (patch.max() - patch.min()) > EDGE_VARIATION * centre:
        return np.nan
    return float(centre)


def score(sequence: Path, path: Path) -> dict:
    frame_time, depth_times, depth_files = load_truth(sequence)
    rows, columns = load_rows(path)
    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame_index"]), []).append(r)
    # Per column, the log ratios of every scored pair, and the frames used.
    errors = {c: [] for c in columns}
    frames_used, pairs_with_truth = 0, 0
    for frame, group in sorted(by_frame.items()):
        t = frame_time.get(frame)
        if t is None or not len(depth_times):
            continue
        k = int(np.argmin(np.abs(depth_times - t)))
        if abs(depth_times[k] - t) > MAX_TIME_GAP:
            continue
        truth = np.asarray(Image.open(depth_files[k]), dtype=np.float64) / DEPTH_SCALE
        estimates, truths = [], []
        for r in group:
            z_true = sample_depth(truth, float(r["x"]), float(r["y"]))
            if not np.isfinite(z_true):
                continue
            pairs_with_truth += 1
            z = np.array([float(r[c]) for c in columns])
            if not (np.isfinite(z).all() and (z > 0).all()):  # the same pairs for every column
                continue
            estimates.append(z)
            truths.append(z_true)
        if len(truths) < MIN_LANDMARKS:
            continue
        frames_used += 1
        log_ratio = np.log(np.array(estimates)) - np.log(np.array(truths))[:, None]  # pairs x columns
        scale = np.median(log_ratio, axis=0)
        for j, c in enumerate(columns):
            errors[c].append(np.abs(log_ratio[:, j] - scale[j]))
    result = {"frames": frames_used, "pairs_with_truth": pairs_with_truth, "rows": len(rows), "columns": {}}
    for c in columns:
        e = np.concatenate(errors[c]) if errors[c] else np.array([])
        result["columns"][c] = {
            "pairs": int(e.size),
            "err": float(np.median(e)) if e.size else float("nan"),
            "within5": float(np.mean(e < 0.05)) if e.size else float("nan"),
            "within10": float(np.mean(e < 0.10)) if e.size else float("nan"),
        }
    return result


def print_report(name: str, result: dict) -> None:
    print(f"\n{name}: {result['rows']} (landmark, keyframe) rows, {result['pairs_with_truth']} with ground-truth depth, "
          f"{result['frames']} keyframes scored")
    base = result["columns"].get("depth_keyframes", {}).get("err", float("nan"))
    print(f"{'depth column':>20} {'err':>8} {'<5%':>7} {'<10%':>7} {'pairs':>8} {'ratio':>7}")
    for c, r in result["columns"].items():
        ratio = r["err"] / base if base and np.isfinite(base) and base > 0 else float("nan")
        print(f"{c:>20} {100 * r['err']:>7.2f}% {100 * r['within5']:>6.1f}% {100 * r['within10']:>6.1f}% "
              f"{r['pairs']:>8d} {ratio:>7.3f}")


def markdown_row(name: str, result: dict) -> str:
    cells = [name, str(result["frames"])]
    base = result["columns"].get("depth_keyframes", {}).get("err", float("nan"))
    for c, r in result["columns"].items():
        ratio = r["err"] / base if base and np.isfinite(base) and base > 0 else float("nan")
        cells.append(f"{100 * r['err']:.2f}% ({ratio:.2f}x, n={r['pairs']})" if c != "depth_keyframes"
                     else f"{100 * r['err']:.2f}% (n={r['pairs']})")
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("landmarks", type=Path, nargs="+")
    parser.add_argument("--markdown", action="store_true", help="also print one Markdown table row per CSV")
    args = parser.parse_args(argv[1:])
    rows = []
    for path in args.landmarks:
        result = score(args.sequence, path)
        print_report(str(path), result)
        if args.markdown:
            rows.append((path, result))
    if rows:
        columns = list(rows[0][1]["columns"])
        print("\n| run | keyframes | " + " | ".join(columns) + " |")
        print("|" + "---|" * (len(columns) + 2))
        for path, result in rows:
            print(markdown_row(path.stem, result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
