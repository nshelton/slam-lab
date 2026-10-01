#!/usr/bin/env python3
"""Run the VO on every TUM RGB-D sequence and score it against ground truth.

Monocular VO is chaotic: a tiny numerical change can flip a keyframe decision
and change the result of a whole run. So every sequence is run from several
start offsets (the tracks are trimmed; the tracker output is the same), and
results are reported as mean and spread over those trials. Compare variants
by the mean, and only trust differences larger than the spread.

  .venv-cuda/bin/python native_workbench/tools/tum_suite.py NAME [--offsets 0,150,300]
      [--set KEY=VALUE ...]   (passed through to slam-native-vo-tracks)

Tracks are exported once per sequence with the native tracker (bench, full
length, colours and per-track sigma) into out/tum/<sequence>/tracks.csv and
reused; delete them after tracker changes. Intrinsics come from the
sequence's calibration.json. Lens distortion is not modelled: fr3 images are
already undistorted, fr1/fr2 are not, which affects every variant equally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tum_eval import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "native_workbench/build/app-debug"
ENGINE = ROOT / "native_workbench/models/superpoint-1024x576-k2048.engine"
METRICS = ("ate_cm", "ate_all_cm", "rpe_cm", "rpe_deg")


def tracks_for(sequence: Path, out: Path, offset: int) -> Path:
    full = out / "tracks.csv"
    if not full.exists() or full.stat().st_size == 0:
        with open(out / "bench.log", "w") as log:
            subprocess.run([BUILD / "slam-native-tracking-bench", sequence / "rgb.mp4", ENGINE, "0", "1000000",
                            "export-tracks", full], stdout=log, stderr=subprocess.STDOUT, check=True)
    if offset == 0:
        return full
    trimmed = out / f"tracks-from-{offset}.csv"
    if not trimmed.exists():
        with open(full) as src, open(trimmed, "w") as dst:
            for line in src:
                if line[0].isdigit() and int(line.split(",", 1)[0]) < offset:
                    continue
                dst.write(line)
    return trimmed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--offsets", default="0,150,300")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--build", default=str(BUILD), help="directory with slam-native-vo-tracks (e.g. build/core-debug)")
    args = parser.parse_args()
    offsets = [int(o) for o in args.offsets.split(",")]
    extra = [a for kv in args.set for a in ("--set", kv)]
    # Private copy: a concurrent rebuild must not change (or break) the binary mid-run.
    scratch = Path(tempfile.mkdtemp(prefix="tum-suite-"))
    vo_tracks = Path(shutil.copy2(Path(args.build) / "slam-native-vo-tracks", scratch / "slam-native-vo-tracks"))

    print(f"{args.name}  (mean ± spread over start offsets {args.offsets})")
    print(f"{'sequence':26s} {'ate_cm':>14s} {'ate_all_cm':>14s} {'rpe_cm':>14s} {'rpe_deg':>14s} {'segs':>5s}")
    means = {m: [] for m in METRICS}
    for sequence in sorted((ROOT / "data/datasets/tum").glob("rgbd_dataset_*")):
        out = ROOT / "out/tum" / sequence.name
        out.mkdir(parents=True, exist_ok=True)
        calibration = json.loads((sequence / "calibration.json").read_text())
        trials = []
        for offset in offsets:
            trajectory = out / f"{args.name}-from-{offset}.traj.csv"
            subprocess.run([vo_tracks, "--tracks", tracks_for(sequence, out, offset),
                            "--output", trajectory, "--intrinsics",
                            *(str(calibration[k]) for k in ("fx", "fy", "cx", "cy")), *extra],
                           stdout=subprocess.DEVNULL, check=True)
            trials.append(evaluate(sequence, trajectory))
        row = f"{sequence.name.removeprefix('rgbd_dataset_')[:26]:26s}"
        for m in METRICS:
            values = np.array([t[m] for t in trials], dtype=float)
            means[m].append(np.nanmean(values))
            row += f" {np.nanmean(values):7.2f} ±{np.nanmax(values) - np.nanmin(values):5.2f}"
        row += f" {np.mean([t['segments'] for t in trials]):5.1f}"
        print(row)
    print(f"{'mean over sequences':26s}" + "".join(f" {np.mean(means[m]):7.2f}       " for m in METRICS))
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
