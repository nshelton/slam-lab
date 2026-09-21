"""Single read-only interface for published SLAM Lab artifacts."""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
from pathlib import Path

import numpy as np

from slam_lab.artifacts import validate_manifest


class ArtifactReader:
    """Validate and read one immutable matching, geometry, or reconstruction artifact."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self._manifest = validate_manifest(self.path)
        self.artifact_type = self._manifest["artifact_type"]
        if self.artifact_type not in {"matches", "geometry", "reconstruction"}:
            raise ValueError(f"Unsupported artifact type: {self.artifact_type!r}")

    def manifest(self) -> dict:
        return dict(self._manifest)

    def status(self) -> dict:
        if self.artifact_type == "reconstruction" and self._manifest.get("producer_job_id"):
            from slam_lab.solve_jobs import solve_status

            return solve_status(self.path)
        return {
            "schema_version": 1,
            "artifact_id": self._manifest["artifact_id"],
            "state": "complete",
            "phase": "complete",
            "revision": self._manifest["revision"],
            "updated_at": self._manifest["updated_at"],
            "last_error": None,
            "worker_alive": False,
        }

    def _encode_cursor(self, resource: str, position) -> str:
        value = {
            "artifact_id": self._manifest["artifact_id"],
            "resource": resource,
            "position": position,
        }
        return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).decode()

    def _decode_cursor(self, cursor: str | None, resource: str, default):
        if cursor is None:
            return default
        try:
            value = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Invalid artifact cursor") from error
        if (
            value.get("artifact_id") != self._manifest["artifact_id"]
            or value.get("resource") != resource
        ):
            raise ValueError("Cursor belongs to a different artifact or resource")
        return value["position"]

    @staticmethod
    def _limit(limit: int) -> int:
        if not 1 <= limit <= 1000:
            raise ValueError("Page limit must be between 1 and 1000")
        return limit

    def list_frames(self, *, after: str | None = None, limit: int = 100) -> dict:
        limit = self._limit(limit)
        position = int(self._decode_cursor(after, "frames", -1))
        if self.artifact_type in {"matches", "geometry"}:
            database = "matches.sqlite3" if self.artifact_type == "matches" else "geometry.sqlite3"
            with sqlite3.connect(f"{(self.path / database).as_uri()}?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT row_id,frame_index,timestamp_ns,width,height,keypoint_count "
                    "FROM frames WHERE row_id>? ORDER BY row_id LIMIT ?",
                    (position, limit + 1),
                ).fetchall()
            items = [
                {
                    "row": row,
                    "matching_row": row,
                    "frame_index": frame,
                    "timestamp_ns": timestamp,
                    "width": width,
                    "height": height,
                    "keypoint_count": count,
                }
                for row, frame, timestamp, width, height, count in rows[:limit]
            ]
        else:
            with np.load(self.path / "reconstruction.npz", allow_pickle=False) as data:
                stop = min(position + 1 + limit + 1, len(data["frame_indices"]))
                indices = range(position + 1, stop)
                items = [
                    {
                        "row": row,
                        "matching_row": int(data["frame_rows"][row]),
                        "frame_index": int(data["frame_indices"][row]),
                        "timestamp_ns": int(data["timestamps_ns"][row]),
                        "registered": bool(data["registered"][row]),
                    }
                    for row in indices
                ]
            rows = items
            items = items[:limit]
        more = len(rows) > limit
        return {
            "items": items,
            "next_cursor": self._encode_cursor("frames", items[-1]["row"])
            if more and items
            else None,
        }

    def frame(self, row: int) -> dict:
        pages = self.list_frames(after=self._encode_cursor("frames", row - 1), limit=1)
        if not pages["items"] or pages["items"][0]["row"] != row:
            raise ValueError(f"Unknown frame row: {row}")
        result = pages["items"][0]
        if self.artifact_type in {"matches", "geometry"}:
            if self.artifact_type == "matches":
                from slam_lab.correspondence import MatchStore

                store_type, database = MatchStore, "matches.sqlite3"
            else:
                from slam_lab.verification import GeometryStore

                store_type, database = GeometryStore, "geometry.sqlite3"
            with store_type(self.path / database) as store:
                stored = store.frame(row)
            result = {**result, "keypoints": stored["keypoints"]}
        return result

    def list_pairs(self, *, after: str | None = None, limit: int = 100) -> dict:
        if self.artifact_type not in {"matches", "geometry"}:
            raise ValueError("Pair reading requires a matching or geometry artifact")
        limit = self._limit(limit)
        first, second = self._decode_cursor(after, "pairs", [-1, -1])
        database = "matches.sqlite3" if self.artifact_type == "matches" else "geometry.sqlite3"
        columns = (
            "first,second,count" if self.artifact_type == "matches" else "first,second,summary"
        )
        with sqlite3.connect(f"{(self.path / database).as_uri()}?mode=ro", uri=True) as db:
            rows = db.execute(
                f"SELECT {columns} FROM pairs WHERE first>? OR (first=? AND second>?) "
                "ORDER BY first,second LIMIT ?",
                (first, first, second, limit + 1),
            ).fetchall()
        items = []
        for a, b, value in rows[:limit]:
            item = {"first": a, "second": b}
            item["count" if self.artifact_type == "matches" else "summary"] = (
                value if self.artifact_type == "matches" else json.loads(value)
            )
            items.append(item)
        return {
            "items": items,
            "next_cursor": self._encode_cursor("pairs", [items[-1]["first"], items[-1]["second"]])
            if len(rows) > limit and items
            else None,
        }

    def pair(self, first: int, second: int) -> dict:
        if first >= second:
            raise ValueError("Pair keys must use canonical order: first < second")
        if self.artifact_type == "matches":
            from slam_lab.correspondence import MatchStore

            with MatchStore(self.path / "matches.sqlite3") as store:
                matches = store.pair(first, second)
            return {
                "first": first,
                "second": second,
                "indices": matches.indices,
                "scores": matches.scores,
                "distances": matches.distances,
            }
        if self.artifact_type == "geometry":
            from slam_lab.verification import GeometryStore

            with GeometryStore(self.path / "geometry.sqlite3") as store:
                arrays, summary = store.pair(first, second)
            return {"first": first, "second": second, "arrays": arrays, "summary": summary}
        raise ValueError("Pair reading requires a matching or geometry artifact")

    def load_reconstruction(self) -> dict[str, np.ndarray]:
        if self.artifact_type != "reconstruction":
            raise ValueError("Reconstruction loading requires a reconstruction artifact")
        with np.load(self.path / "reconstruction.npz", allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def load_snapshot(self, revision=None):
        raise ValueError("This artifact has no solve snapshots")
