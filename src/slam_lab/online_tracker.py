"""Causal descriptor-only landmark tracking over cached SuperPoint frames."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from slam_lab.artifacts import atomic_write_json, opaque_id, utc_now, write_manifest
from slam_lab.cache import FrameCache, resolve_cache

TRACK_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class OnlineTrackerConfig:
    min_similarity: float = 0.82
    min_margin: float = 0.02
    max_inactive_frames: int = 15
    landmark_chunk_size: int = 4096
    device: str = "cpu"

    def __post_init__(self):
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("min-similarity must be between -1 and 1")
        if not 0 <= self.min_margin <= 2:
            raise ValueError("min-margin must be between 0 and 2")
        if self.max_inactive_frames < 1:
            raise ValueError("max-inactive-frames must be at least 1")
        if self.landmark_chunk_size < 1:
            raise ValueError("landmark-chunk-size must be at least 1")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")

    def identity(self) -> dict:
        return {
            "schema_version": TRACK_SCHEMA_VERSION,
            "algorithm": "online-spherical-mean-greedy-v1",
            "settings": asdict(self),
        }


class OnlineTrackStore(AbstractContextManager):
    def __init__(self, root: Path, *, create: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "tracks.sqlite3"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "ro"
        self.db = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=30)
        self.db.row_factory = sqlite3.Row
        if create:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frames (
                    row_id INTEGER PRIMARY KEY,
                    frame_index INTEGER NOT NULL UNIQUE,
                    timestamp_ns INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    keypoint_count INTEGER NOT NULL,
                    new_count INTEGER NOT NULL,
                    matched_count INTEGER NOT NULL,
                    mean_similarity REAL
                );
                CREATE TABLE IF NOT EXISTS landmarks (
                    landmark_id INTEGER PRIMARY KEY,
                    observation_count INTEGER NOT NULL,
                    resultant BLOB NOT NULL,
                    concentration REAL NOT NULL,
                    first_row INTEGER NOT NULL,
                    last_row INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    row_id INTEGER NOT NULL,
                    feature_id INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    score REAL NOT NULL,
                    landmark_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    similarity REAL,
                    second_similarity REAL,
                    PRIMARY KEY (row_id, feature_id),
                    FOREIGN KEY (row_id) REFERENCES frames(row_id),
                    FOREIGN KEY (landmark_id) REFERENCES landmarks(landmark_id)
                );
                CREATE INDEX IF NOT EXISTS observations_landmark
                    ON observations(landmark_id, row_id);
            """)

    def __exit__(self, *args):
        self.db.close()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, **values) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?,?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
        )


def _normalized(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 0
    result = np.zeros_like(values)
    result[valid] = values[valid] / norms[valid, None]
    return result, valid


def _best_two_numpy(
    descriptors: np.ndarray, means: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(descriptors)
    best_index = np.full(count, -1, np.int64)
    best = np.full(count, -np.inf, np.float32)
    second = np.full(count, -np.inf, np.float32)
    rows = np.arange(count)
    for start in range(0, len(means), chunk_size):
        similarities = descriptors @ means[start : start + chunk_size].T
        width = similarities.shape[1]
        if width == 1:
            local_best_index = np.zeros(count, np.int64)
            local_best = similarities[:, 0]
            local_second = np.full(count, -np.inf, np.float32)
        else:
            top = np.argpartition(similarities, -2, axis=1)[:, -2:]
            top_values = np.take_along_axis(similarities, top, axis=1)
            order = np.argsort(top_values, axis=1)
            local_second = top_values[rows, order[:, 0]]
            local_best = top_values[rows, order[:, 1]]
            local_best_index = top[rows, order[:, 1]]
        replace = local_best > best
        second = np.where(replace, np.maximum(best, local_second), np.maximum(second, local_best))
        best = np.where(replace, local_best, best)
        best_index = np.where(replace, start + local_best_index, best_index)
    return best_index, best, second


def _best_two_torch(
    descriptors: np.ndarray, means: np.ndarray, chunk_size: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CUDA tracking requires PyTorch") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device == "cpu":
        return _best_two_numpy(descriptors, means, chunk_size)
    query = torch.from_numpy(descriptors).to(device)
    count = len(descriptors)
    best_index = torch.full((count,), -1, dtype=torch.int64, device=device)
    best = torch.full((count,), -torch.inf, device=device)
    second = torch.full((count,), -torch.inf, device=device)
    with torch.inference_mode():
        for start in range(0, len(means), chunk_size):
            reference = torch.from_numpy(means[start : start + chunk_size]).to(device)
            similarities = query @ reference.T
            width = similarities.shape[1]
            if width == 1:
                local_best = similarities[:, 0]
                local_second = torch.full_like(local_best, -torch.inf)
                local_index = torch.zeros(count, dtype=torch.int64, device=device)
            else:
                values, indices = torch.topk(similarities, 2, dim=1)
                local_best, local_second, local_index = values[:, 0], values[:, 1], indices[:, 0]
            replace = local_best > best
            second = torch.where(
                replace,
                torch.maximum(best, local_second),
                torch.maximum(second, local_best),
            )
            best = torch.where(replace, local_best, best)
            best_index = torch.where(replace, start + local_index, best_index)
    return (
        best_index.cpu().numpy(),
        best.cpu().numpy().astype(np.float32),
        second.cpu().numpy().astype(np.float32),
    )


def _best_two(
    descriptors: np.ndarray, means: np.ndarray, config: OnlineTrackerConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(descriptors) or not len(means):
        count = len(descriptors)
        return (
            np.full(count, -1, np.int64),
            np.full(count, -np.inf, np.float32),
            np.full(count, -np.inf, np.float32),
        )
    if config.device in {"cuda", "auto"}:
        return _best_two_torch(descriptors, means, config.landmark_chunk_size, config.device)
    return _best_two_numpy(descriptors, means, config.landmark_chunk_size)


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def online_track_status(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    path = root / "tracks.sqlite3"
    if not path.is_file():
        raise ValueError(f"Online track artifact not found: {root}")
    with OnlineTrackStore(root) as store:
        phase = store.get("phase", "unknown")
        processed = store.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        observations = store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        landmarks = store.db.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0]
        matched = store.db.execute(
            "SELECT COUNT(*) FROM observations WHERE state='matched'"
        ).fetchone()[0]
        return {
            "phase": phase,
            "running": phase == "running",
            "processed_frames": processed,
            "total_frames": store.get("total_frames"),
            "observations": observations,
            "landmarks": landmarks,
            "matched_observations": matched,
            "new_observations": observations - matched,
            "artifact_id": store.get("artifact_id"),
            "source_cache": store.get("source_cache"),
            "source_cache_id": store.get("source_cache_id"),
            "source_path": store.get("source_path"),
            "config": store.get("config", {}),
            "updated_at": store.get("updated_at"),
        }


def track_online(
    cache_path: Path,
    output: Path,
    *,
    config: OnlineTrackerConfig | None = None,
    max_frames: int | None = None,
    progress: bool = True,
    producer_job_id: str | None = None,
) -> Path:
    """Associate cached descriptors causally and commit one frame at a time."""
    config = config or OnlineTrackerConfig()
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be at least 1")
    cache_path = resolve_cache(cache_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".writer.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Online tracker already running: {output}") from error
        with FrameCache(cache_path) as cache, OnlineTrackStore(output, create=True) as store:
            cache_id = cache.get("run_id")
            identity = {
                "source_cache_id": cache_id,
                "tracker": config.identity(),
            }
            existing = store.get("identity")
            if existing is not None and existing != identity:
                raise ValueError("Existing online track artifact has different inputs or settings")
            artifact_id = store.get("artifact_id") or opaque_id("online-tracks")
            available = cache.count()
            total = min(available, max_frames) if max_frames is not None else available
            processed = store.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            if processed > total:
                raise ValueError(
                    "Existing output contains more frames than the requested selection"
                )
            if processed == total and (output / "manifest.json").is_file():
                return output
            if (output / "manifest.json").exists():
                raise ValueError("Completed online track artifacts are immutable")
            store.set(
                artifact_id=artifact_id,
                identity=identity,
                source_cache=str(cache_path),
                source_cache_id=cache_id,
                source_path=cache.get("source_path"),
                config=config.identity(),
                phase="running",
                total_frames=total,
                pid=os.getpid(),
                updated_at=utc_now(),
            )
            store.db.commit()

            next_landmark = store.db.execute(
                "SELECT COALESCE(MAX(landmark_id),-1)+1 FROM landmarks"
            ).fetchone()[0]
            next_row = processed
            active: dict[int, tuple[np.ndarray, int, int]] = {}
            if processed:
                minimum_row = next_row - config.max_inactive_frames
                for record in store.db.execute(
                    "SELECT landmark_id,resultant,observation_count,last_row FROM landmarks "
                    "WHERE last_row>=? ORDER BY landmark_id",
                    (minimum_row,),
                ):
                    active[record[0]] = (
                        np.frombuffer(record[1], dtype="<f4").copy(),
                        record[2],
                        record[3],
                    )

            bar = tqdm(total=total, initial=processed, unit="frame", disable=not progress)
            try:
                for row, frame in enumerate(cache.frames()):
                    if row < processed:
                        continue
                    if row >= total:
                        break
                    minimum_row = row - config.max_inactive_frames
                    active = {
                        landmark_id: state
                        for landmark_id, state in active.items()
                        if state[2] >= minimum_row
                    }
                    descriptors, valid = _normalized(frame.descriptors)
                    landmark_ids = np.array(sorted(active), dtype=np.int64)
                    means = np.empty((len(landmark_ids), 256), np.float32)
                    for index, landmark_id in enumerate(landmark_ids):
                        resultant = active[int(landmark_id)][0]
                        norm = np.linalg.norm(resultant)
                        means[index] = resultant / norm if norm else 0
                    best_indices, similarities, second_similarities = _best_two(
                        descriptors, means, config
                    )
                    margins = np.full_like(similarities, -np.inf)
                    finite_best = np.isfinite(similarities)
                    finite_second = np.isfinite(second_similarities)
                    margins[finite_best & ~finite_second] = np.inf
                    margins[finite_best & finite_second] = (
                        similarities[finite_best & finite_second]
                        - second_similarities[finite_best & finite_second]
                    )
                    candidates = np.flatnonzero(
                        valid
                        & (best_indices >= 0)
                        & (similarities >= config.min_similarity)
                        & (margins >= config.min_margin)
                    )
                    order = candidates[np.lexsort((candidates, -similarities[candidates]))]
                    assignments = np.full(len(descriptors), -1, np.int64)
                    used: set[int] = set()
                    for feature_id in order:
                        landmark_id = int(landmark_ids[best_indices[feature_id]])
                        if landmark_id not in used:
                            assignments[feature_id] = landmark_id
                            used.add(landmark_id)

                    matched_count = int((assignments >= 0).sum())
                    observation_rows = []
                    landmark_inserts = []
                    landmark_updates = []
                    for feature_id, descriptor in enumerate(descriptors):
                        landmark_id = int(assignments[feature_id])
                        state = "matched"
                        if landmark_id >= 0:
                            resultant, count, _ = active[landmark_id]
                            resultant = resultant + descriptor
                            count += 1
                            concentration = float(np.linalg.norm(resultant) / count)
                            active[landmark_id] = (resultant, count, row)
                            landmark_updates.append(
                                (
                                    count,
                                    np.asarray(resultant, dtype="<f4").tobytes(),
                                    concentration,
                                    row,
                                    landmark_id,
                                )
                            )
                        else:
                            state = "new"
                            landmark_id = next_landmark
                            next_landmark += 1
                            resultant = descriptor.copy()
                            concentration = 1.0 if valid[feature_id] else 0.0
                            active[landmark_id] = (resultant, 1, row)
                            landmark_inserts.append(
                                (
                                    landmark_id,
                                    1,
                                    np.asarray(resultant, dtype="<f4").tobytes(),
                                    concentration,
                                    row,
                                    row,
                                )
                            )
                        x, y = frame.keypoints[feature_id]
                        observation_rows.append(
                            (
                                row,
                                feature_id,
                                float(x),
                                float(y),
                                float(frame.scores[feature_id]),
                                landmark_id,
                                state,
                                _finite(similarities[feature_id]),
                                _finite(second_similarities[feature_id]),
                            )
                        )
                    matched_values = similarities[assignments >= 0]
                    mean_similarity = float(matched_values.mean()) if len(matched_values) else None
                    with store.db:
                        store.db.executemany(
                            "INSERT INTO landmarks VALUES (?,?,?,?,?,?)", landmark_inserts
                        )
                        store.db.executemany(
                            "UPDATE landmarks SET observation_count=?,resultant=?,"
                            "concentration=?,last_row=? WHERE landmark_id=?",
                            landmark_updates,
                        )
                        store.db.execute(
                            "INSERT INTO frames "
                            "(row_id,frame_index,timestamp_ns,width,height,keypoint_count,"
                            "new_count,matched_count,mean_similarity) VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                row,
                                frame.index,
                                frame.timestamp_ns,
                                frame.width,
                                frame.height,
                                len(frame.keypoints),
                                len(frame.keypoints) - matched_count,
                                matched_count,
                                mean_similarity,
                            ),
                        )
                        store.db.executemany(
                            "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                            observation_rows,
                        )
                        store.set(
                            phase="running",
                            processed_frames=row + 1,
                            updated_at=utc_now(),
                        )
                    bar.update(1)
                    bar.set_postfix(
                        active=len(active),
                        matched=matched_count,
                        new=len(frame.keypoints) - matched_count,
                    )
            except BaseException:
                with store.db:
                    store.set(phase="paused", updated_at=utc_now())
                raise
            finally:
                bar.close()

            with store.db:
                store.set(phase="complete", processed_frames=total, updated_at=utc_now())
            summary = online_track_status(output)
            summary["longest_landmark"] = store.db.execute(
                "SELECT COALESCE(MAX(observation_count),0) FROM landmarks"
            ).fetchone()[0]
            summary["landmarks_two_or_more"] = store.db.execute(
                "SELECT COUNT(*) FROM landmarks WHERE observation_count>=2"
            ).fetchone()[0]
            summary["landmarks_three_or_more"] = store.db.execute(
                "SELECT COUNT(*) FROM landmarks WHERE observation_count>=3"
            ).fetchone()[0]
            atomic_write_json(output / "summary.json", summary)
            store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            write_manifest(
                output,
                artifact_id=artifact_id,
                artifact_type="online_tracks",
                parent_artifact_id=cache_id,
                producer_job_id=producer_job_id,
                payloads=["tracks.sqlite3", "summary.json"],
                capabilities=[
                    "causal_descriptor_association",
                    "frame_observations",
                    "landmark_history",
                    "source_cache_images",
                    "spherical_cluster_statistics",
                ],
                origin={"source_cache": str(cache_path)},
            )
    return output
