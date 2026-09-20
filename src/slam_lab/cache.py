"""Transactional, self-contained frame caches. No ML or viewer imports here."""

import json
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from slam_lab.config import SCHEMA_VERSION


@dataclass
class CachedFrame:
    index: int
    timestamp_ns: int
    pts: int | None
    width: int
    height: int
    original_width: int
    original_height: int
    jpeg: bytes
    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray
    extraction_ms: float


class FrameCache(AbstractContextManager):
    def __init__(self, path: Path, *, create: bool = False):
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "ro"
        self.db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode={mode}", uri=True)
        if create:
            # Each frame is committed atomically. An interrupted transaction is rolled back.
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS frames (
                    frame_index INTEGER PRIMARY KEY,
                    timestamp_ns INTEGER NOT NULL,
                    pts INTEGER,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    original_width INTEGER NOT NULL,
                    original_height INTEGER NOT NULL,
                    jpeg BLOB NOT NULL,
                    features BLOB NOT NULL,
                    extraction_ms REAL NOT NULL,
                    keypoint_count INTEGER NOT NULL
                );
            """)
        schema = self.get("schema_version")
        if schema is not None and schema != SCHEMA_VERSION:
            self.db.close()
            raise ValueError(f"Unsupported cache schema {schema}: {self.path}")

    def __exit__(self, *args):
        self.db.close()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, **values):
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
            )

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def contains(self, index: int) -> bool:
        return (
            self.db.execute("SELECT 1 FROM frames WHERE frame_index = ?", (index,)).fetchone()
            is not None
        )

    def put(self, frame: CachedFrame):
        size = len(frame.keypoints)
        if frame.keypoints.shape != (size, 2) or frame.scores.shape != (size,):
            raise ValueError("Invalid keypoint or score shape")
        if frame.descriptors.shape != (size, 256):
            raise ValueError("SuperPoint descriptors must have shape (N, 256)")
        arrays = (frame.keypoints, frame.scores, frame.descriptors)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Non-finite feature values")
        features = BytesIO()
        np.savez_compressed(
            features,
            keypoints=np.asarray(frame.keypoints, dtype=np.float32),
            scores=np.asarray(frame.scores, dtype=np.float32),
            descriptors=np.asarray(frame.descriptors, dtype=np.float32),
        )
        with self.db:
            self.db.execute(
                "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    frame.index,
                    frame.timestamp_ns,
                    frame.pts,
                    frame.width,
                    frame.height,
                    frame.original_width,
                    frame.original_height,
                    frame.jpeg,
                    features.getvalue(),
                    frame.extraction_ms,
                    size,
                ),
            )

    def frames(self, *, descriptors: bool = True):
        cursor = self.db.execute(
            "SELECT frame_index, timestamp_ns, pts, width, height, original_width, "
            "original_height, jpeg, features, extraction_ms FROM frames ORDER BY frame_index"
        )
        for row in cursor:
            with np.load(BytesIO(row[8]), allow_pickle=False) as features:
                yield CachedFrame(
                    *row[:8],
                    keypoints=features["keypoints"],
                    scores=features["scores"],
                    descriptors=features["descriptors"] if descriptors else np.empty((0, 256)),
                    extraction_ms=row[9],
                )


def resolve_cache(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / "cache.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(f"Cache not found: {path}. Run 'slam-lab list' to find a cache.")
    return path
