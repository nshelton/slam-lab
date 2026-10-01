#!/usr/bin/env python3
"""Snapshot what the full pipeline produces, for comparing versions.

Runs slam-native-tracking-bench (decode, SuperPoint, flow tracking, VO) over
every TUM sequence and the dreamworks clip, from frame 0. A run is
deterministic, but monocular VO is chaotic: one keyframe decision can change
a whole run. So each sequence can be run from several start frames (--skips;
the bench decodes and drops them) and compared by the mean over those runs;
only differences larger than the spread are real. Writes

  OUT/<dataset>/<sequence>/trajectory[-from-N].csv   frame,timestamp,segment,pose
  OUT/<dataset>/<sequence>/run[-from-N].log          the bench's stdout/stderr
  OUT/summary.csv                                    tum_eval metrics per run
  OUT/manifest.json                                  commit, binary, command lines

  .venv-cuda/bin/python native_workbench/tools/baseline.py OUT --build BUILD_DIR [--only SUBSTRING] [--skips 0,150,300]
      [--set KEY VALUE ...]   (extra bench options, e.g. --set window 10)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tum_eval import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "native_workbench/models/superpoint-1024x576-k2048.engine"
DREAMWORKS = Path.home() / "Downloads/dreamworks.MOV"
METRICS = ("posed", "segments", "ate_cm", "ate_all_cm", "rpe_cm", "rpe_deg")


def sequences() -> list[tuple[str, str, Path, Path | None]]:
    """(dataset, name, video, sequence dir with ground truth or None)."""
    found = []
    for d in sorted((ROOT / "data/datasets/tum").glob("rgbd_dataset_*")):
        found.append(("tum", d.name.removeprefix("rgbd_dataset_"), d / "rgb.mp4", d))
    if DREAMWORKS.exists():
        found.append(("video", "dreamworks", DREAMWORKS, None))
    return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--build", type=Path, required=True, help="directory with slam-native-tracking-bench")
    parser.add_argument("--only", default="", help="run sequences whose name contains this")
    parser.add_argument("--skips", default="0", help="start frames, e.g. 0,150,300")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    args = parser.parse_args()
    bench = args.build / "slam-native-tracking-bench"
    extra = [x for kv in args.set for x in kv]
    args.out.mkdir(parents=True, exist_ok=True)
    previous = {}  # runs of an interrupted snapshot, kept when resuming
    if (args.out / "manifest.json").exists():
        previous = {(r["dataset"], r["sequence"], r.get("skip", 0)): r for r in json.loads((args.out / "manifest.json").read_text())["runs"]}
    source = args.build.resolve().parents[1]
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": subprocess.run(["git", "-C", source, "status", "--porcelain", "--", "src", "include"],
                                capture_output=True, text=True).stdout.strip() != "",
        "bench_sha256": sha256(bench),
        "engine": ENGINE.name,
        "engine_sha256": sha256(ENGINE),
        "runs": [],
    }
    rows = []
    skips = [int(k) for k in args.skips.split(",")]
    for dataset, name, video, sequence in sequences():
        if args.only not in name:
            continue
        out = args.out / dataset / name
        out.mkdir(parents=True, exist_ok=True)
        intrinsics = []
        if sequence is not None:
            calibration = json.loads((sequence / "calibration.json").read_text())
            intrinsics = [x for k in ("fx", "fy", "cx", "cy") for x in (k, str(calibration[k]))]
        trials = []
        for skip in skips:
            suffix = f"-from-{skip}" if skip else ""
            trajectory = out / f"trajectory{suffix}.csv"
            command = [str(bench), str(video), str(ENGINE), "0", "1000000", *intrinsics, *extra,
                       *(["skip", str(skip)] if skip else []), "export-trajectory", str(trajectory)]
            # The bench writes the trajectory only when it finishes: an existing one is a complete run.
            if trajectory.exists() and (dataset, name, skip) in previous:
                run = previous[(dataset, name, skip)]
            else:
                started = time.monotonic()
                with open(out / f"run{suffix}.log", "w") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                run = {"dataset": dataset, "sequence": name, "skip": skip, "command": command,
                       "seconds": round(time.monotonic() - started, 1), "exit": code}
            manifest["runs"].append(run)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            row = {"dataset": dataset, "sequence": name, "skip": skip, "exit": run.get("exit", 0)}
            if sequence is not None and row["exit"] == 0:
                result = evaluate(sequence, trajectory)
                row |= {m: (round(result[m], 3) if isinstance(result[m], float) else result[m]) for m in METRICS}
            rows.append(row)
            trials.append(row)
        ate = np.array([t["ate_cm"] for t in trials if "ate_cm" in t], dtype=float)
        line = f"{dataset:6s} {name:34s} exit={','.join(str(t['exit']) for t in trials)}"
        if ate.size:
            line += (f"  ate_cm {np.nanmean(ate):7.2f} ±{np.nanmax(ate) - np.nanmin(ate):6.2f}"
                     f"  segments {np.mean([t['segments'] for t in trials if 'segments' in t]):.1f}")
        print(line, flush=True)
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "sequence", "skip", "exit", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
