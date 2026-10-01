#!/usr/bin/env python3
"""Fetch selected KITTI Odometry sequences without downloading whole archives.

The official archives bundle all 22 sequences (colour: 69 GB). ZIP keeps a
central directory, so this reads it over HTTP range requests and pulls only
the members it needs: the left colour camera (image_2) and times.txt of the
chosen sequences, plus calib.txt and the ground-truth poses (00-10).

  python fetch_kitti_odometry.py data/datasets/kitti --sequences 00 05 07
  python fetch_kitti_odometry.py data/datasets/kitti --list   # sizes only

Layout (as in the archives): <out>/sequences/NN/{image_2/*.png,times.txt,calib.txt},
<out>/poses/NN.txt. Interrupted runs resume (complete files are skipped).
KITTI is CC BY-NC-SA 3.0 (https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
COLOR = BASE + "data_odometry_color.zip"
CALIB = BASE + "data_odometry_calib.zip"
POSES = BASE + "data_odometry_poses.zip"
# Sequences with ground truth whose trajectories revisit places.
LOOP_SEQUENCES = ["00", "02", "05", "06", "07", "08", "09"]


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file via HTTP Range requests."""

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        for attempt in range(6):
            try:
                request = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={self.position}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2**attempt)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=4 << 20))


def relative(name: str) -> str:
    return name.split("dataset/", 1)[1]


def extract(url: str, names: list[str], out: Path, workers: int) -> None:
    """Streams the wanted members: archives store each sequence's frames
    contiguously, so a few long range requests replace one request per file
    (latency, not bandwidth, limits per-file requests: ~1.5 vs ~15 MB/s each)."""
    archive = open_remote_zip(url)
    wanted = set(names)
    infos = sorted((i for i in archive.infolist() if i.filename in wanted), key=lambda i: i.header_offset)
    pending = [i for i in infos if not ((out / relative(i.filename)).exists() and
                                       (out / relative(i.filename)).stat().st_size == i.file_size)]
    total = sum(i.compress_size for i in pending)
    print(f"{url.rsplit('/', 1)[1]}: {len(infos)} files, {len(pending)} to fetch, {total / 1e9:.2f} GB",
          flush=True)
    if not pending:
        return
    # Split into runs of members that are adjacent in the archive (gap < 1 MB),
    # then into about `workers` chunks of similar size.
    target = max(1, total // workers)
    chunks: list[list[zipfile.ZipInfo]] = [[]]
    for info in pending:
        chunk = chunks[-1]
        if chunk:
            previous = chunk[-1]
            gap = info.header_offset - (previous.header_offset + previous.compress_size)
            if gap > (1 << 20) or sum(i.compress_size for i in chunk) >= target:
                chunks.append([])
        chunks[-1].append(info)
    lock = threading.Lock()
    progress = {"bytes": 0, "files": 0}
    started = time.time()

    def stream(chunk: list[zipfile.ZipInfo]) -> None:
        last = chunk[-1]
        # Local headers are 30 bytes + name + extra; allow 64 KB of extra.
        first, final = chunk[0].header_offset, last.header_offset + 30 + 65536 + last.compress_size
        request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{final}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            position = first
            for info in chunk:
                skip = info.header_offset - position
                if skip:
                    response.read(skip)
                header = response.read(30)
                if header[:4] != b"PK\x03\x04":
                    raise RuntimeError(f"bad local header for {info.filename}")
                name_length = int.from_bytes(header[26:28], "little")
                extra_length = int.from_bytes(header[28:30], "little")
                response.read(name_length + extra_length)
                data = response.read(info.compress_size)
                if len(data) != info.compress_size:
                    raise RuntimeError(f"short read for {info.filename}")
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    data = zlib.decompress(data, -15)
                elif info.compress_type != zipfile.ZIP_STORED:
                    raise RuntimeError(f"unsupported compression for {info.filename}")
                position = info.header_offset + 30 + name_length + extra_length + info.compress_size
                path = out / relative(info.filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".part")
                partial.write_bytes(data)
                partial.rename(path)
                with lock:
                    progress["bytes"] += info.compress_size
                    progress["files"] += 1
                    if progress["files"] % 500 == 0:
                        rate = progress["bytes"] / (time.time() - started) / 1e6
                        print(f"  {progress['files']}/{len(pending)} files ({rate:.1f} MB/s)", flush=True)

    def with_retries(chunk: list[zipfile.ZipInfo]) -> None:
        for attempt in range(5):
            try:
                todo = [i for i in chunk if not (out / relative(i.filename)).exists()]
                if todo:
                    stream(todo)
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(with_retries, chunk) for chunk in chunks]):
            future.result()
    rate = progress["bytes"] / max(1e-6, time.time() - started) / 1e6
    print(f"  done: {progress['files']} files ({rate:.1f} MB/s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--sequences", nargs="+", default=LOOP_SEQUENCES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list", action="store_true", help="print per-sequence sizes and exit")
    args = parser.parse_args()

    color = open_remote_zip(COLOR)
    by_sequence: dict[str, list[zipfile.ZipInfo]] = {}
    for info in color.infolist():
        parts = info.filename.split("/")
        if len(parts) >= 4 and parts[1] == "sequences" and (
            parts[3] == "image_2" and not info.is_dir() or parts[3] == "times.txt"
        ):
            by_sequence.setdefault(parts[2], []).append(info)
    if args.list:
        for sequence, infos in sorted(by_sequence.items()):
            frames = sum(1 for info in infos if info.filename.endswith(".png"))
            size = sum(info.file_size for info in infos)
            print(f"{sequence}: {frames} frames, {size / 1e9:.2f} GB")
        return
    missing = [s for s in args.sequences if s not in by_sequence]
    if missing:
        sys.exit(f"unknown sequences: {missing}")

    args.out.mkdir(parents=True, exist_ok=True)
    calib = [f"dataset/sequences/{s}/calib.txt" for s in args.sequences]
    extract(CALIB, calib, args.out, 4)
    poses = [f"dataset/poses/{s}.txt" for s in args.sequences if int(s) <= 10]
    extract(POSES, poses, args.out, 4)
    for sequence in args.sequences:
        names = [info.filename for info in by_sequence[sequence]]
        extract(COLOR, names, args.out, args.workers)


if __name__ == "__main__":
    main()
