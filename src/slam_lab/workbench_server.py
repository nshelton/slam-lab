"""Read-only HTTP adapter for the browser-based SLAM workbench."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from slam_lab.cache import FrameCache
from slam_lab.correspondence import MatchStore, match_status
from slam_lab.online_tracker import OnlineTrackStore, online_track_status
from slam_lab.workbench_pipeline import PipelineSupervisor


class WorkbenchData:
    """Translate an existing match artifact into browser-friendly values."""

    def __init__(self, run: Path, run_id: str | None = None):
        self.run = Path(run).expanduser().resolve()
        self.run_id = run_id or self.run.name
        if not (self.run / "matches.sqlite3").is_file():
            raise ValueError(f"Not a matching run: {self.run}")

    def overview(self):
        status = match_status(self.run)
        with MatchStore(self.run / "matches.sqlite3") as store:
            source = store.get("source")
            suggested = store.db.execute(
                "SELECT first,second,count FROM pairs WHERE second=first+1 "
                "ORDER BY count DESC,first LIMIT 1"
            ).fetchone()
            if suggested is None:
                suggested = store.db.execute(
                    "SELECT first,second,count FROM pairs ORDER BY count DESC,first,second LIMIT 1"
                ).fetchone()
        return {
            "run": str(self.run),
            "source": source,
            "status": status,
            "suggested_pair": (
                {"first": suggested[0], "second": suggested[1], "matches": suggested[2]}
                if suggested
                else None
            ),
        }

    def _frame_info(self, row: int, frame: dict):
        return {
            "row": row,
            "frame_index": frame["index"],
            "timestamp_ns": frame["timestamp_ns"],
            "width": frame["width"],
            "height": frame["height"],
            "keypoint_count": len(frame["keypoints"]),
            "video_url": f"/api/match/video?run={quote(self.run_id)}",
        }

    @staticmethod
    def _runtime_for_pair(store: MatchStore, first: int, second: int):
        columns = {row[1] for row in store.db.execute("PRAGMA table_info(pairs)")}
        if "runtime_id" not in columns:
            return None
        runtime = store.db.execute(
            "SELECT runtime_id FROM pairs WHERE first=? AND second=?",
            (min(first, second), max(first, second)),
        ).fetchone()
        if runtime is None or runtime[0] is None:
            return None
        exists = store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_runtimes'"
        ).fetchone()
        if not exists:
            return None
        provenance = store.db.execute(
            "SELECT provenance FROM match_runtimes WHERE runtime_id=?", (runtime[0],)
        ).fetchone()
        return json.loads(provenance[0]) if provenance else {"runtime_id": runtime[0]}

    def pair(self, first: int, second: int):
        if first == second:
            raise ValueError("Choose two different frame rows")
        with MatchStore(self.run / "matches.sqlite3") as store:
            first_frame, second_frame = store.frame(first), store.frame(second)
            matches = store.pair(first, second)
            runtime = self._runtime_for_pair(store, first, second)
            first_xy = first_frame["keypoints"][matches.indices[:, 0]]
            second_xy = second_frame["keypoints"][matches.indices[:, 1]]
        return {
            "first": self._frame_info(first, first_frame),
            "second": self._frame_info(second, second_frame),
            "matcher": match_status(self.run)["matcher"],
            "pair_runtime": runtime,
            "matches": {
                "count": len(matches.indices),
                "indices": matches.indices.tolist(),
                "scores": matches.scores.tolist(),
                "distances": matches.distances.tolist(),
                "first_xy": first_xy.tolist(),
                "second_xy": second_xy.tolist(),
            },
            "features": {
                "first_xy": first_frame["keypoints"].tolist(),
                "second_xy": second_frame["keypoints"].tolist(),
            },
        }

    def matrix(self):
        with MatchStore(self.run / "matches.sqlite3") as store:
            size = store.get("frames")
            counts = [-1] * (size * size)
            for row in range(size):
                counts[row * size + row] = -2
            maximum = 0
            for first, second, count in store.db.execute(
                "SELECT first,second,count FROM pairs ORDER BY first,second"
            ):
                counts[first * size + second] = count
                counts[second * size + first] = count
                maximum = max(maximum, count)
        return {
            "size": size,
            "counts": counts,
            "max_count": maximum,
            "matcher": match_status(self.run)["matcher"],
        }

    def video_path(self):
        with MatchStore(self.run / "matches.sqlite3") as store:
            source = store.get("source")
        if not isinstance(source, str) or not Path(source).expanduser().is_file():
            raise ValueError(f"Source video is unavailable: {source}")
        return Path(source).expanduser().resolve()


class DetectionData:
    """Expose a live feature cache without copying its frames or features."""

    def __init__(self, path: Path, detection_id: str | None = None):
        self.path = Path(path).expanduser().resolve()
        self.database = self.path / "cache.sqlite3"
        self.detection_id = detection_id or self.path.name
        if not self.database.is_file():
            raise ValueError(f"Not a feature cache: {self.path}")
        # Validate the schema during discovery so obsolete caches never enter the catalog.
        with FrameCache(self.database):
            pass

    def overview(self):
        with FrameCache(self.database) as cache:
            count, average_ms = cache.db.execute(
                "SELECT COUNT(*),AVG(extraction_ms) FROM frames"
            ).fetchone()
            return {
                "id": self.detection_id,
                "name": self.path.name,
                "path": str(self.path),
                "source": cache.get("source_path"),
                "source_cache_id": cache.get("run_id"),
                "complete": cache.get("complete", False),
                "selection_complete": cache.get(
                    "selection_complete", cache.get("updated_at") is not None
                ),
                "selection_max_frames": cache.get("selection_max_frames"),
                "frames": count,
                "latest_row": count - 1 if count else None,
                "average_detection_ms": average_ms,
                "extractor": cache.get("extractor", {}),
                "video": cache.get("video", {}),
                "updated_at": cache.get("updated_at"),
            }

    @staticmethod
    def _row(cache: FrameCache, row: int):
        if row < 0:
            raise ValueError(f"Unknown detection frame row: {row}")
        record = cache.db.execute(
            "SELECT frame_index,timestamp_ns,width,height,keypoints,scores,extraction_ms,"
            "keypoint_count FROM frames ORDER BY frame_index LIMIT 1 OFFSET ?",
            (row,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Unknown detection frame row: {row}")
        return record

    def frame(self, row: int):
        with FrameCache(self.database) as cache:
            record = self._row(cache, row)
            count = record[7]
            keypoints = np.frombuffer(record[4], dtype="<f4").reshape(count, 2).tolist()
            scores = np.frombuffer(record[5], dtype="<f4").reshape(count).tolist()
        return {
            "overview": self.overview(),
            "frame": {
                "row": row,
                "frame_index": record[0],
                "timestamp_ns": record[1],
                "width": record[2],
                "height": record[3],
                "keypoint_count": record[7],
                "extraction_ms": record[6],
                "video_url": f"/api/detection/video?detection={quote(self.detection_id)}",
            },
            "keypoints": keypoints,
            "scores": scores,
        }

    def video_path(self):
        with FrameCache(self.database) as cache:
            source = cache.get("source_path")
        if not isinstance(source, str) or not Path(source).expanduser().is_file():
            raise ValueError(f"Source video is unavailable: {source}")
        return Path(source).expanduser().resolve()


class OnlineTrackData:
    """Read a live or completed causal landmark artifact for the browser."""

    def __init__(self, path: Path, track_id: str | None = None):
        self.path = Path(path).expanduser().resolve()
        self.track_id = track_id or self.path.name
        if not (self.path / "tracks.sqlite3").is_file():
            raise ValueError(f"Not an online track artifact: {self.path}")

    def overview(self):
        status = online_track_status(self.path)
        return {
            "id": self.track_id,
            "name": self.path.name,
            "path": str(self.path),
            **status,
        }

    def frame(self, row: int, *, trail: int = 12):
        trail = min(100, max(0, trail))
        with OnlineTrackStore(self.path) as store:
            frame = store.db.execute(
                "SELECT row_id,frame_index,timestamp_ns,width,height,keypoint_count,"
                "new_count,matched_count,mean_similarity FROM frames WHERE row_id=?",
                (row,),
            ).fetchone()
            if frame is None:
                raise ValueError(f"Unknown online track frame: {row}")
            records = store.db.execute(
                "SELECT o.feature_id,o.x,o.y,o.score,o.landmark_id,o.state,"
                "o.similarity,o.second_similarity,l.observation_count,l.concentration,"
                "l.first_row,l.last_row FROM observations o JOIN landmarks l "
                "ON l.landmark_id=o.landmark_id WHERE o.row_id=? ORDER BY o.feature_id",
                (row,),
            ).fetchall()
            landmark_ids = [record[4] for record in records]
            trails: dict[int, list[list[float | int]]] = {}
            if trail and landmark_ids:
                placeholders = ",".join("?" for _ in landmark_ids)
                query = (
                    "SELECT landmark_id,row_id,x,y FROM observations WHERE landmark_id IN ("
                    + placeholders
                    + ") AND row_id BETWEEN ? AND ? ORDER BY landmark_id,row_id"
                )
                for item in store.db.execute(query, (*landmark_ids, max(0, row - trail), row)):
                    trails.setdefault(item[0], []).append([item[1], item[2], item[3]])
            observations = [
                {
                    "feature_id": record[0],
                    "x": record[1],
                    "y": record[2],
                    "score": record[3],
                    "landmark_id": record[4],
                    "state": record[5],
                    "similarity": record[6],
                    "second_similarity": record[7],
                    "observation_count": record[8],
                    "concentration": record[9],
                    "first_row": record[10],
                    "last_row": record[11],
                    "trail": trails.get(record[4], []),
                }
                for record in records
            ]
        return {
            "overview": self.overview(),
            "frame": {
                "row": frame[0],
                "frame_index": frame[1],
                "timestamp_ns": frame[2],
                "width": frame[3],
                "height": frame[4],
                "keypoint_count": frame[5],
                "new_count": frame[6],
                "matched_count": frame[7],
                "mean_similarity": frame[8],
                "video_url": f"/api/online-track/video?track={quote(self.track_id)}",
            },
            "observations": observations,
        }

    def landmark(self, landmark_id: int):
        with OnlineTrackStore(self.path) as store:
            landmark = store.db.execute(
                "SELECT landmark_id,observation_count,concentration,first_row,last_row "
                "FROM landmarks WHERE landmark_id=?",
                (landmark_id,),
            ).fetchone()
            if landmark is None:
                raise ValueError(f"Unknown landmark: {landmark_id}")
            observations = store.db.execute(
                "SELECT o.row_id,f.frame_index,f.timestamp_ns,o.feature_id,o.x,o.y,"
                "o.score,o.state,o.similarity,o.second_similarity FROM observations o "
                "JOIN frames f ON f.row_id=o.row_id WHERE o.landmark_id=? ORDER BY o.row_id",
                (landmark_id,),
            ).fetchall()
        return {
            "landmark_id": landmark[0],
            "observation_count": landmark[1],
            "concentration": landmark[2],
            "first_row": landmark[3],
            "last_row": landmark[4],
            "observations": [
                {
                    "row": item[0],
                    "frame_index": item[1],
                    "timestamp_ns": item[2],
                    "feature_id": item[3],
                    "x": item[4],
                    "y": item[5],
                    "score": item[6],
                    "state": item[7],
                    "similarity": item[8],
                    "second_similarity": item[9],
                }
                for item in observations
            ],
        }

    def video_path(self):
        with OnlineTrackStore(self.path) as store:
            source = store.get("source_path")
        if not isinstance(source, str) or not Path(source).expanduser().is_file():
            raise ValueError(f"Source video is unavailable: {source}")
        return Path(source).expanduser().resolve()


class ReconstructionData:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.archive = self.path / "reconstruction.npz"
        self.metadata_path = self.path / "metadata.json"
        if not self.archive.is_file() or not self.metadata_path.is_file():
            raise ValueError(f"Not a reconstruction artifact: {self.path}")

    def metadata(self):
        return json.loads(self.metadata_path.read_text())

    def overview(self):
        metadata = self.metadata()
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return {
            "name": self.path.name,
            "path": str(self.path),
            "artifact_id": metadata.get("artifact_id"),
            "parent_artifact_id": manifest.get("parent_artifact_id"),
            "updated_at": manifest.get("updated_at"),
            "backend": metadata.get("backend", "incremental-mapper"),
            "registered_frames": metadata.get("registered_frames"),
            "frames": metadata.get("frames"),
            "points": metadata.get("points"),
            "observations": metadata.get("observations"),
            "median_reprojection_error_px": metadata.get("median_reprojection_error_px"),
            "p95_reprojection_error_px": metadata.get("p95_reprojection_error_px"),
            "source_match_run": metadata.get("source_match_run"),
        }

    def scene(self):
        metadata = self.metadata()
        with np.load(self.archive, allow_pickle=False) as archive:
            registered = archive["registered"].astype(bool)
            poses = archive["T_world_camera"][registered]
            return {
                "overview": self.overview(),
                "points": archive["points"].tolist(),
                "colors": archive["colors"].tolist(),
                "track_lengths": archive["track_lengths"].tolist(),
                "poses": poses.tolist(),
                "frame_indices": archive["frame_indices"][registered].tolist(),
                "timestamps_ns": archive["timestamps_ns"][registered].tolist(),
                "K": archive["K"].tolist(),
                "pose_convention": metadata.get("pose_convention", "T_world_camera"),
                "scale": metadata.get("scale"),
            }


class GeometryData:
    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.manifest_path = self.path / "manifest.json"
        self.summary_path = self.path / "summary.json"
        if not self.manifest_path.is_file() or not self.summary_path.is_file():
            raise ValueError(f"Not a geometry artifact: {self.path}")
        if self.manifest().get("artifact_type") != "geometry":
            raise ValueError(f"Not a geometry artifact: {self.path}")

    def manifest(self):
        return json.loads(self.manifest_path.read_text())

    def overview(self):
        manifest = self.manifest()
        summary = json.loads(self.summary_path.read_text())
        return {
            "name": self.path.name,
            "path": str(self.path),
            "artifact_id": manifest.get("artifact_id"),
            "parent_artifact_id": manifest.get("parent_artifact_id"),
            "state": manifest.get("state"),
            "updated_at": manifest.get("updated_at"),
            "frames": summary.get("frames"),
            "pairs": summary.get("pairs"),
            "verified_edges": summary.get("verified_edges"),
            "tracks": summary.get("tracks_three_or_more"),
        }


def _unique_paths(paths):
    result = []
    seen = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _artifact_directories(roots: list[Path], payload: str):
    paths = []
    for root in roots:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            continue
        paths.extend(path.parent for path in sorted(resolved.rglob(payload)))
    return _unique_paths(paths)


class WorkbenchCatalog:
    def __init__(
        self,
        runs: list[Path],
        reconstructions: list[Path] | None = None,
        artifact_roots: list[Path] | None = None,
    ):
        self.explicit_runs = list(runs)
        self.explicit_reconstructions = list(reconstructions or [])
        self.artifact_roots = list(artifact_roots or [])
        roots = artifact_roots or []
        run_paths = _unique_paths([*runs, *_artifact_directories(roots, "matches.sqlite3")])
        self.runs = {}
        for path in run_paths:
            candidate = WorkbenchData(path)
            matcher = candidate.overview()["status"]["matcher"]["name"]
            base = {
                "lightglue-superpoint": "lightglue",
                "cosine-mutual-nearest-neighbor": "cosine",
                "mutual-nearest-neighbor": "nn",
            }.get(matcher, candidate.run.name)
            run_id, suffix = base, 2
            while run_id in self.runs:
                run_id, suffix = f"{base}-{suffix}", suffix + 1
            candidate.run_id = run_id
            self.runs[run_id] = candidate
        if not self.runs:
            raise ValueError("No matching artifacts found")
        self.default = next(iter(self.runs))

        self.reconstructions = {}
        self.reconstruction_matches = {}
        reconstruction_paths = _unique_paths(
            [
                *(reconstructions or []),
                *_artifact_directories(roots, "reconstruction.npz"),
            ]
        )
        for path in reconstruction_paths:
            reconstruction = ReconstructionData(path)
            base = reconstruction.path.name
            reconstruction_id, suffix = base, 2
            while reconstruction_id in self.reconstructions:
                reconstruction_id, suffix = f"{base}-{suffix}", suffix + 1
            self.reconstructions[reconstruction_id] = reconstruction
            source = reconstruction.metadata().get("source_match_run")
            for run_id, data in self.runs.items():
                if source and Path(source).expanduser().resolve() == data.run:
                    self.reconstruction_matches[reconstruction_id] = run_id
                    break

        self.geometries = []
        for path in _artifact_directories(roots, "manifest.json"):
            try:
                self.geometries.append(GeometryData(path))
            except (ValueError, json.JSONDecodeError, OSError):
                continue

        self.online_tracks = {}
        for path in _artifact_directories(roots, "tracks.sqlite3"):
            try:
                candidate = OnlineTrackData(path)
                base = path.parent.name if path.name == "online-tracks" else path.name
                track_id, suffix = base, 2
                while track_id in self.online_tracks:
                    track_id, suffix = f"{base}-{suffix}", suffix + 1
                candidate.track_id = track_id
                self.online_tracks[track_id] = candidate
            except (ValueError, json.JSONDecodeError, OSError, sqlite3.Error):
                continue

        self.detections = {}
        for path in _artifact_directories(roots, "cache.sqlite3"):
            try:
                candidate = DetectionData(path)
                base = path.parent.parent.name if path.parent.name == "cache" else path.name
                detection_id, suffix = base, 2
                while detection_id in self.detections:
                    detection_id, suffix = f"{base}-{suffix}", suffix + 1
                candidate.detection_id = detection_id
                self.detections[detection_id] = candidate
            except (ValueError, json.JSONDecodeError, OSError, sqlite3.Error):
                continue

    def refresh(self):
        fresh = type(self)(
            self.explicit_runs,
            self.explicit_reconstructions,
            self.artifact_roots,
        )
        self.runs = fresh.runs
        self.default = fresh.default
        self.reconstructions = fresh.reconstructions
        self.reconstruction_matches = fresh.reconstruction_matches
        self.geometries = fresh.geometries
        self.online_tracks = fresh.online_tracks
        self.detections = fresh.detections

    def get(self, run_id: str | None):
        if run_id is None:
            return self.runs[self.default]
        if run_id not in self.runs:
            raise ValueError(f"Unknown matching run: {run_id}")
        return self.runs[run_id]

    def list(self):
        return {
            "default": self.default,
            "runs": [
                {
                    "id": run_id,
                    "run": str(data.run),
                    "status": data.overview()["status"],
                    "reconstructions": [
                        reconstruction_id
                        for reconstruction_id, match_run_id in self.reconstruction_matches.items()
                        if match_run_id == run_id
                    ],
                }
                for run_id, data in self.runs.items()
            ],
            "reconstructions": [
                {
                    "id": reconstruction_id,
                    "match_run_id": self.reconstruction_matches.get(reconstruction_id),
                    **data.overview(),
                }
                for reconstruction_id, data in self.reconstructions.items()
            ],
            "geometry": [data.overview() for data in self.geometries],
            "online_tracks": [data.overview() for data in self.online_tracks.values()],
            "detections": [data.overview() for data in self.detections.values()],
        }

    def detection(self, detection_id: str | None):
        if detection_id is None:
            if not self.detections:
                raise ValueError("No feature caches are available")
            return next(iter(self.detections.values()))
        if detection_id not in self.detections:
            raise ValueError(f"Unknown feature cache: {detection_id}")
        return self.detections[detection_id]

    def online_track(self, track_id: str | None):
        if track_id is None:
            if not self.online_tracks:
                raise ValueError("No online track artifacts are available")
            return next(iter(self.online_tracks.values()))
        if track_id not in self.online_tracks:
            raise ValueError(f"Unknown online track artifact: {track_id}")
        return self.online_tracks[track_id]

    def reconstruction(self, reconstruction_id: str | None, run_id: str | None):
        if reconstruction_id is not None:
            if reconstruction_id not in self.reconstructions:
                raise ValueError(f"Unknown reconstruction: {reconstruction_id}")
            return self.reconstructions[reconstruction_id]
        resolved_run = run_id or self.default
        for candidate_id, candidate_run in self.reconstruction_matches.items():
            if candidate_run == resolved_run:
                return self.reconstructions[candidate_id]
        if self.reconstructions:
            return next(iter(self.reconstructions.values()))
        raise ValueError("No reconstruction artifacts are available")

    def sources(self):
        result = []
        for data in self.runs.values():
            source = data.overview().get("source")
            if source:
                resolved = str(Path(source).expanduser().resolve())
                if resolved not in result and Path(resolved).is_file():
                    result.append(resolved)
        for data in self.detections.values():
            source = data.overview().get("source")
            if source:
                resolved = str(Path(source).expanduser().resolve())
                if resolved not in result and Path(resolved).is_file():
                    result.append(resolved)
        return result


def make_handler(
    catalog: WorkbenchCatalog,
    static_root: Path,
    pipeline: PipelineSupervisor | None = None,
):
    static_root = static_root.resolve()

    class WorkbenchHandler(BaseHTTPRequestHandler):
        server_version = "SlamLabWorkbench/0.1"

        def _bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers routinely abandon superseded status/image requests.
                pass

        def _json(self, value, status=HTTPStatus.OK):
            self._bytes(
                json.dumps(value, separators=(",", ":")).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _error(self, status, message):
            self._json({"error": message}, status)

        def _video(self, video: Path):
            size = video.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            requested = self.headers.get("Range")
            if requested:
                if not requested.startswith("bytes=") or "," in requested:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid byte range")
                    return
                first, separator, last = requested[6:].partition("-")
                try:
                    if first:
                        start = int(first)
                        end = int(last) if separator and last else size - 1
                    elif last:
                        length = int(last)
                        start = max(0, size - length)
                    else:
                        raise ValueError
                except ValueError:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid byte range")
                    return
                if start < 0 or start >= size or end < start:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(video.name)[0] or "video/mp4")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            try:
                with video.open("rb") as source:
                    source.seek(start)
                    remaining = length
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            run_id = query.get("run", [None])[0]
            reconstruction_id = query.get("reconstruction", [None])[0]
            track_id = query.get("track", [None])[0]
            detection_id = query.get("detection", [None])[0]
            try:
                if path == "/api/runs":
                    catalog.refresh()
                    self._json(catalog.list())
                    return
                if path == "/api/pipeline":
                    if pipeline is None:
                        raise ValueError("Pipeline supervisor is disabled")
                    self._json({"sources": catalog.sources(), "jobs": pipeline.list()})
                    return
                if path.startswith("/api/pipeline/jobs/"):
                    if pipeline is None:
                        raise ValueError("Pipeline supervisor is disabled")
                    job_id = path.rsplit("/", 1)[-1]
                    job = next((item for item in pipeline.list() if item["job_id"] == job_id), None)
                    if job is None:
                        raise ValueError(f"Unknown pipeline job: {job_id}")
                    self._json(job)
                    return
                if path == "/api/online-track":
                    self._json(catalog.online_track(track_id).overview())
                    return
                if path == "/api/online-track/video":
                    self._video(catalog.online_track(track_id).video_path())
                    return
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[:3] == ["api", "online-track", "frame"]:
                    trail = int(query.get("trail", ["12"])[0])
                    self._json(catalog.online_track(track_id).frame(int(parts[3]), trail=trail))
                    return
                if len(parts) == 4 and parts[:3] == ["api", "online-track", "landmark"]:
                    self._json(catalog.online_track(track_id).landmark(int(parts[3])))
                    return
                if path == "/api/detection":
                    self._json(catalog.detection(detection_id).overview())
                    return
                if path == "/api/detection/video":
                    self._video(catalog.detection(detection_id).video_path())
                    return
                if len(parts) == 4 and parts[:3] == ["api", "detection", "frame"]:
                    self._json(catalog.detection(detection_id).frame(int(parts[3])))
                    return
                data = catalog.get(run_id)
                if path == "/api/match/video":
                    self._video(data.video_path())
                    return
                if path == "/api/run":
                    self._json(data.overview())
                    return
                if path == "/api/matrix":
                    self._json(data.matrix())
                    return
                if path == "/api/reconstruction":
                    self._json(catalog.reconstruction(reconstruction_id, run_id).scene())
                    return
                if len(parts) == 4 and parts[:2] == ["api", "pairs"]:
                    self._json(data.pair(int(parts[2]), int(parts[3])))
                    return
                self._static(path)
            except (ValueError, IndexError, KeyError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            except sqlite3.Error as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if pipeline is None:
                    raise ValueError("Pipeline supervisor is disabled")
                if path == "/api/pipeline/jobs":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 65536:
                        raise ValueError("Invalid request body")
                    request = json.loads(self.rfile.read(length))
                    self._json(
                        pipeline.start(request, catalog.sources()),
                        HTTPStatus.ACCEPTED,
                    )
                    return
                if path.startswith("/api/pipeline/jobs/") and path.endswith("/cancel"):
                    job_id = path.split("/")[-2]
                    self._json(pipeline.cancel(job_id), HTTPStatus.ACCEPTED)
                    return
                self._error(HTTPStatus.NOT_FOUND, "Not found")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            except OSError as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))

        def _static(self, path: str):
            relative = "index.html" if path == "/" else path.lstrip("/")
            candidate = (static_root / relative).resolve()
            if static_root not in candidate.parents and candidate != static_root:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if not candidate.is_file():
                # Client-side routes should still boot the workbench.
                candidate = static_root / "index.html"
            if not candidate.is_file():
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "Workbench frontend is not built. "
                    "Run npm install && npm run build in workbench/.",
                )
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._bytes(candidate.read_bytes(), content_type)

        def log_message(self, message, *args):
            print(f"{self.address_string()} - {message % args}")

    return WorkbenchHandler


def parser():
    default_static = Path(__file__).resolve().parents[2] / "workbench" / "dist"
    result = argparse.ArgumentParser(description="Serve a matching run in the SLAM workbench")
    result.add_argument(
        "runs", type=Path, nargs="*", help="One or more directories containing matches.sqlite3"
    )
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--static", type=Path, default=default_static)
    result.add_argument(
        "--reconstruction",
        type=Path,
        action="append",
        default=[],
        help="Completed reconstruction artifact; repeat for matcher alternatives",
    )
    result.add_argument(
        "--artifacts-root",
        type=Path,
        action="append",
        default=[Path("recordings")],
        help="Discover match, geometry, and reconstruction artifacts below this directory",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    catalog = WorkbenchCatalog(args.runs, args.reconstruction, args.artifacts_root)
    workspace = Path(__file__).resolve().parents[2]
    pipeline = PipelineSupervisor(workspace, workspace / "recordings")
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(catalog, args.static, pipeline)
    )
    print(f"SLAM workbench: http://{args.host}:{args.port}")
    for run_id, data in catalog.runs.items():
        print(f"{run_id}: {data.run}")
        linked = [
            data.path
            for reconstruction_id, data in catalog.reconstructions.items()
            if catalog.reconstruction_matches.get(reconstruction_id) == run_id
        ]
        for path in linked:
            print(f"  reconstruction: {path}")
    print(
        f"catalog: {len(catalog.detections)} feature caches, "
        f"{len(catalog.online_tracks)} online tracks, "
        f"{len(catalog.runs)} matches, {len(catalog.geometries)} geometry, "
        f"{len(catalog.reconstructions)} reconstructions"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
