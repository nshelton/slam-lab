"""Resumable exhaustive image matching followed by global feature-track association."""

import fcntl
import hashlib
import json
import sqlite3
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from io import BytesIO
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from slam_lab.artifacts import opaque_id, validate_manifest, write_manifest
from slam_lab.cache import FrameCache, resolve_cache
from slam_lab.config import fingerprint
from slam_lab.matching import CosineMatcher, DescriptorMatcher, LightGlueMatcher, PairMatches

PAIR_DTYPE = np.dtype([("first", "<i4"), ("second", "<i4"), ("score", "<f4"), ("distance", "<f4")])


def algorithm_identity(identity):
    """Runtime changes may alter numerics, but never permit changing the matcher itself."""
    result = {**identity, "matcher": dict(identity["matcher"])}
    if result["matcher"].get("name") == "lightglue-superpoint":
        for key in ("device", "torch", "cuda", "device_name"):
            result["matcher"].pop(key, None)
    return result


def pack_matches(matches):
    array = np.empty(len(matches.indices), dtype=PAIR_DTYPE)
    array["first"], array["second"] = matches.indices.T
    array["score"], array["distance"] = matches.scores, matches.distances
    return zlib.compress(array.tobytes(), level=1)


def unpack_matches(blob):
    array = np.frombuffer(zlib.decompress(blob), dtype=PAIR_DTYPE)
    return PairMatches(
        np.column_stack([array["first"], array["second"]]),
        array["score"].copy(),
        array["distance"].copy(),
    )


class MatchStore(AbstractContextManager):
    def __init__(self, path, *, create=False):
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "ro"
        self.db = sqlite3.connect(
            f"{self.path.resolve().as_uri()}?mode={mode}", uri=True, timeout=30
        )
        self.runtime_id = None
        if create:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS frames (
                    row_id INTEGER PRIMARY KEY, frame_index INTEGER NOT NULL,
                    timestamp_ns INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    keypoints BLOB NOT NULL, keypoint_count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS pairs (
                    first INTEGER NOT NULL, second INTEGER NOT NULL, count INTEGER NOT NULL,
                    matches BLOB NOT NULL, elapsed_ms REAL NOT NULL,
                    PRIMARY KEY(first, second), CHECK(first < second));
                CREATE TABLE IF NOT EXISTS match_runtimes (
                    runtime_id TEXT PRIMARY KEY, provenance TEXT NOT NULL);
            """)
            if "runtime_id" not in {row[1] for row in self.db.execute("PRAGMA table_info(pairs)")}:
                self.db.execute("ALTER TABLE pairs ADD COLUMN runtime_id TEXT")

    def __exit__(self, *args):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, **values):
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
            )

    def initialize(self, frames, identity, source, *, allow_runtime_change=False):
        saved = self.get("identity")
        if saved is not None:
            if self.get("artifact_id") is None:
                raise ValueError("Matching run has no artifact identity; choose a new --output")
            compatible = allow_runtime_change and algorithm_identity(saved) == algorithm_identity(
                identity
            )
            if saved != identity and not compatible:
                raise ValueError(
                    "This output contains a different selection or matcher. Choose a new --output. "
                    "For only a LightGlue device/PyTorch change, use --allow-runtime-change."
                )
            self.register_runtime(identity["matcher"])
            return
        if self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]:
            raise ValueError("Unrecognized incomplete match store")
        with self.db:
            self.db.executemany(
                "INSERT INTO frames VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        row,
                        f.index,
                        f.timestamp_ns,
                        f.width,
                        f.height,
                        np.asarray(f.keypoints, dtype="<f4").tobytes(),
                        len(f.keypoints),
                    )
                    for row, f in enumerate(frames)
                ],
            )
            values = {
                "schema_version": 1,
                "artifact_id": opaque_id("matches"),
                "identity": identity,
                "source": source,
                "frames": len(frames),
                "total_pairs": len(frames) * (len(frames) - 1) // 2,
                "phase": "matching",
                "created_at": datetime.now(UTC).isoformat(),
            }
            self.db.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
            )
        self.register_runtime(identity["matcher"])

    def register_runtime(self, provenance):
        original = self.get("identity")["matcher"]
        original_id = fingerprint(original)
        self.runtime_id = fingerprint(provenance)
        with self.db:
            for runtime_id, value in ((original_id, original), (self.runtime_id, provenance)):
                self.db.execute(
                    "INSERT OR IGNORE INTO match_runtimes VALUES (?,?)",
                    (runtime_id, json.dumps(value, sort_keys=True)),
                )
            # Legacy pairs were computed with the original run's recorded runtime.
            self.db.execute(
                "UPDATE pairs SET runtime_id=? WHERE runtime_id IS NULL", (original_id,)
            )
        self.set(active_matcher=provenance)

    def runtime_summary(self):
        if not self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_runtimes'"
        ).fetchone():
            return []
        return [
            {"runtime_id": row[0], "matcher": json.loads(row[1]), "pairs": row[2]}
            for row in self.db.execute(
                "SELECT r.runtime_id,r.provenance,COUNT(p.first) FROM match_runtimes r "
                "LEFT JOIN pairs p ON p.runtime_id=r.runtime_id GROUP BY r.runtime_id "
                "ORDER BY r.runtime_id"
            )
        ]

    def put_pair(self, first, second, matches, elapsed_ms):
        with self.db:
            self.db.execute(
                "INSERT INTO pairs(first,second,count,matches,elapsed_ms,runtime_id) "
                "VALUES (?,?,?,?,?,?)",
                (
                    first,
                    second,
                    len(matches.indices),
                    pack_matches(matches),
                    elapsed_ms,
                    self.runtime_id,
                ),
            )

    def pair(self, first, second):
        reverse = first > second
        row = self.db.execute(
            "SELECT matches FROM pairs WHERE first=? AND second=?",
            (min(first, second), max(first, second)),
        ).fetchone()
        if row is None:
            raise ValueError("That pair has not been processed yet")
        matches = unpack_matches(row[0])
        if reverse:
            matches.indices = matches.indices[:, ::-1].copy()
        return matches

    def frame(self, row):
        record = self.db.execute(
            "SELECT frame_index,timestamp_ns,width,height,keypoints "
            "FROM frames WHERE row_id=?",
            (row,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Unknown selected-frame row: {row}")
        return {
            "index": record[0],
            "timestamp_ns": record[1],
            "width": record[2],
            "height": record[3],
            "keypoints": np.frombuffer(record[4], dtype="<f4").reshape(-1, 2),
        }


def selected_identity(frames, matcher, max_frames, frame_step):
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(
            np.asarray(
                [frame.index, frame.timestamp_ns, frame.width, frame.height], dtype="<i8"
            ).tobytes()
        )
        digest.update(np.asarray(frame.keypoints, dtype="<f4").tobytes())
        digest.update(np.asarray(frame.descriptors, dtype="<f4").tobytes())
    return {
        "schema": 1,
        "selection_hash": digest.hexdigest(),
        "max_frames": max_frames,
        "frame_step": frame_step,
        "matcher": matcher.provenance,
        "pair_policy": "exhaustive-upper-triangle-v1",
        "track_policy": "confidence-ordered-union-unique-frame-v1",
    }


def associate_tracks(frame_counts, edges, scores):
    """Greedy global association; a track may contain at most one feature from each frame."""
    offsets = np.r_[0, np.cumsum(frame_counts)]
    total = int(offsets[-1])
    parent, sizes = list(range(total)), [1] * total
    frame_masks = [1 << row for row, count in enumerate(frame_counts) for _ in range(count)]

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    conflicts = cycles = merged = 0
    # Stable ties follow canonical pair/keypoint order, independent of worker completion order.
    for edge_index in np.argsort(-scores, kind="stable"):
        left, right = edges[edge_index]
        left, right = find(int(left)), find(int(right))
        if left == right:
            cycles += 1
            continue
        if frame_masks[left] & frame_masks[right]:
            conflicts += 1
            continue
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]
        frame_masks[left] |= frame_masks[right]
        frame_masks[right] = 0
        merged += 1
    roots = np.fromiter((find(node) for node in range(total)), dtype=np.int64, count=total)
    _, track_ids = np.unique(roots, return_inverse=True)
    lengths = np.bincount(track_ids)
    return (
        offsets,
        track_ids.astype(np.int32),
        lengths,
        {
            "merged_edges": merged,
            "consistent_cycle_edges": cycles,
            "conflicting_edges": conflicts,
            "observations": total,
            "tracks_including_singletons": len(lengths),
            "tracks_two_or_more": int((lengths >= 2).sum()),
            "tracks_three_or_more": int((lengths >= 3).sum()),
            "singleton_observations": int((lengths == 1).sum()),
            "longest_track": int(lengths.max()) if len(lengths) else 0,
        },
    )


def finalize_matches(store, output):
    completed = store.db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
    if completed != store.get("total_pairs"):
        raise ValueError("Finish all pairs before building global tracks")
    frame_counts = [
        row[0] for row in store.db.execute("SELECT keypoint_count FROM frames ORDER BY row_id")
    ]
    offsets = np.r_[0, np.cumsum(frame_counts)]
    edge_count = store.db.execute("SELECT COALESCE(SUM(count),0) FROM pairs").fetchone()[0]
    edges, scores = np.empty((edge_count, 2), np.int32), np.empty(edge_count, np.float32)
    counts = np.zeros((len(frame_counts), len(frame_counts)), np.int32)
    cursor = 0
    for first, second, count, blob in store.db.execute(
        "SELECT first,second,count,matches FROM pairs ORDER BY first,second"
    ):
        matches = unpack_matches(blob)
        edges[cursor : cursor + count] = matches.indices + offsets[[first, second]]
        scores[cursor : cursor + count] = matches.scores
        counts[first, second] = counts[second, first] = count
        cursor += count
    offsets, track_ids, lengths, summary = associate_tracks(frame_counts, edges, scores)
    rows = store.db.execute(
        "SELECT frame_index,timestamp_ns FROM frames ORDER BY row_id"
    ).fetchall()
    archive = BytesIO()
    np.savez_compressed(
        archive,
        offsets=offsets,
        track_ids=track_ids,
        track_lengths=lengths,
        pair_match_counts=counts,
        frame_indices=np.array([row[0] for row in rows]),
        timestamps_ns=np.array([row[1] for row in rows], dtype=np.int64),
    )
    temporary = output / "tracks.npz.tmp"
    temporary.write_bytes(archive.getvalue())
    temporary.replace(output / "tracks.npz")
    summary.update(
        frames=len(frame_counts),
        unique_pairs=len(frame_counts) * (len(frame_counts) - 1) // 2,
        candidate_edges=edge_count,
        matcher=store.get("active_matcher", store.get("identity")["matcher"]),
        runtimes=store.runtime_summary(),
        geometry_verified=False,
        input_modality="monocular RGB",
        description="Appearance correspondence tracks; no camera poses or 3D points",
    )
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    store.set(phase="complete", summary=summary, completed_at=datetime.now(UTC).isoformat())
    store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    write_manifest(
        output,
        artifact_id=store.get("artifact_id"),
        artifact_type="matches",
        parent_artifact_id=None,
        producer_job_id=None,
        payloads=["matches.sqlite3", "summary.json", "tracks.npz"],
        capabilities=["video_referenced_frames", "matching_observation_namespace"],
        origin={
            "kind": "feature_cache_snapshot",
            "feature_cache": store.get("feature_cache"),
            "selection_identity": store.get("identity_hash"),
        },
    )
    return summary


def _match_all(
    cache_path,
    output,
    *,
    matcher_name="lightglue",
    ratio=0.8,
    threshold=0.1,
    cosine_threshold=0.8,
    max_frames=300,
    frame_step=1,
    workers=4,
    device="cpu",
    model_dir=None,
    progress=True,
    matcher=None,
    resume_after=None,
    allow_runtime_change=False,
):
    from threadpoolctl import threadpool_limits

    if max_frames < 2 or frame_step < 1 or workers < 1:
        raise ValueError("Need at least 2 max-frames, positive frame-step and workers")
    cache_path, output = (
        resolve_cache(Path(cache_path).expanduser()).resolve(),
        Path(output).expanduser().resolve(),
    )
    if output.exists() and not (output / "matches.sqlite3").exists():
        raise FileExistsError(f"Unrecognized existing output directory: {output}")
    with FrameCache(cache_path) as cache:
        frames = list(islice(cache.frames(), 0, max_frames * frame_step, frame_step))
        source = cache.get("source_path")
    if len(frames) < 2:
        raise ValueError("Need at least two cached frames")
    if matcher is None:
        if matcher_name == "lightglue":
            matcher = LightGlueMatcher(
                Path(model_dir or cache_path.parent.parent / "models"),
                device=device,
                threshold=threshold,
                threads=1,
            )
        elif matcher_name == "nn":
            matcher = DescriptorMatcher(ratio)
        elif matcher_name == "cosine":
            matcher = CosineMatcher(cosine_threshold)
        else:
            raise ValueError(f"Unknown matcher: {matcher_name}")
    if getattr(matcher, "device", device) == "cuda" and workers != 1:
        raise ValueError("Use --workers 1 for CUDA to bound GPU memory")
    identity = selected_identity(frames, matcher, max_frames, frame_step)
    with MatchStore(output / "matches.sqlite3", create=True) as store:
        store.initialize(frames, identity, source, allow_runtime_change=allow_runtime_change)
        if store.get("phase") == "complete":
            manifest = validate_manifest(output, artifact_type="matches")
            if manifest["artifact_id"] != store.get("artifact_id"):
                raise ValueError("Matching manifest does not belong to this match store")
            return store.get("summary")
        store.set(
            phase="matching",
            feature_cache=str(cache_path),
            identity_hash=fingerprint(identity),
            last_started_at=datetime.now(UTC).isoformat(),
            workers=workers,
            queued_after_run=None,
            resume_after_run=str(Path(resume_after).expanduser().resolve())
            if resume_after
            else None,
        )
        done = set(store.db.execute("SELECT first,second FROM pairs"))
        total = store.get("total_pairs")
        # Near pairs first give useful previews quickly; every pair is still scheduled exactly once.
        pending_pairs = iter(
            (first, first + gap)
            for gap in range(1, len(frames))
            for first in range(len(frames) - gap)
            if (first, first + gap) not in done
        )
        started, completed_here, last_report = perf_counter(), 0, 0.0

        def process(pair):
            start = perf_counter()
            result = matcher(frames[pair[0]], frames[pair[1]])
            result.validate(len(frames[pair[0]].keypoints), len(frames[pair[1]].keypoints))
            return pair, result, 1000 * (perf_counter() - start)

        with (
            threadpool_limits(limits=1),
            ThreadPoolExecutor(max_workers=workers) as pool,
            tqdm(
                total=total, initial=len(done), desc="All-pairs matching", disable=not progress
            ) as bar,
        ):
            active = set()
            for pair in islice(pending_pairs, workers * 2):
                active.add(pool.submit(process, pair))
            while active:
                ready, active = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    pair, result, elapsed = future.result()
                    store.put_pair(*pair, result, elapsed)
                    completed_here += 1
                    bar.update(1)
                    pair = next(pending_pairs, None)
                    if pair is not None:
                        active.add(pool.submit(process, pair))
                elapsed = perf_counter() - started
                if elapsed - last_report >= 15 or not active:
                    rate = completed_here / max(elapsed, 1e-9)
                    count = len(done) + completed_here
                    store.set(
                        completed_pairs=count,
                        pairs_per_second=rate,
                        estimated_remaining_seconds=(total - count) / rate,
                        updated_at=datetime.now(UTC).isoformat(),
                    )
                    if not progress:
                        print(
                            f"{count}/{total} pairs; {rate:.2f} pairs/s; "
                            f"ETA {(total - count) / rate / 3600:.2f} h",
                            flush=True,
                        )
                    last_report = elapsed
        store.set(phase="associating_tracks")
        summary = finalize_matches(store, output)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def match_all(cache_path, output, *, export=False, **options):
    """Hold an OS lock for the entire run, including automatic recording export."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Matching is already running: {output}") from error
        try:
            summary = _match_all(cache_path, output, **options)
        except KeyboardInterrupt:
            if (output / "matches.sqlite3").exists():
                with MatchStore(output / "matches.sqlite3", create=True) as store:
                    store.set(
                        phase="paused",
                        updated_at=datetime.now(UTC).isoformat(),
                        pairs_per_second=None,
                        estimated_remaining_seconds=None,
                    )
            raise
        if export:
            from slam_lab.match_view import view_matches

            recording = output / "correspondences.rrd"
            if not recording.exists():
                temporary = output / "correspondences.rrd.tmp"
                # A failed export can be regenerated from the completed match store.
                if temporary.exists():
                    temporary.unlink()
                view_matches(output, output=temporary)
                temporary.replace(recording)
            print(f"Recording: {recording}", flush=True)
        return summary


def match_status(path):
    path = Path(path).expanduser().resolve()
    running = False
    lock_path = path.with_suffix(".lock")
    if lock_path.exists():
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                running = True
    if not (path / "matches.sqlite3").exists():
        return {"phase": "starting" if running else "not_started", "running": running}
    with MatchStore(path / "matches.sqlite3") as store:
        count, edges = store.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(count),0) FROM pairs"
        ).fetchone()
        return {
            "running": running,
            "phase": store.get("phase"),
            "frames": store.get("frames"),
            "completed_pairs": count,
            "total_pairs": store.get("total_pairs"),
            "candidate_edges": edges,
            "pairs_per_second": store.get("pairs_per_second"),
            "estimated_remaining_seconds": store.get("estimated_remaining_seconds"),
            "updated_at": store.get("updated_at"),
            "matcher": store.get("active_matcher", store.get("identity")["matcher"]),
            "runtimes": store.runtime_summary(),
            "summary": store.get("summary"),
            "resume_after_run": store.get("resume_after_run"),
            "queued_after_run": store.get("queued_after_run"),
        }
