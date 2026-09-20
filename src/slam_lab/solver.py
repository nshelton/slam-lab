"""Graph-based SfM consuming verified observations, never descriptors or matchers."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from slam_lab.cache import CachedFrame
from slam_lab.geometry import Camera, camera_center, estimate_pose, project, triangulate
from slam_lab.reconstruction import Mapper, ReconstructionConfig
from slam_lab.verification import GeometryStore, coverage


def geometry_frames(store):
    """Make descriptor-free frames from the self-contained geometry snapshot."""
    for (row,) in store.db.execute("SELECT row_id FROM frames ORDER BY row_id"):
        frame = store.frame(row)
        n = len(frame["keypoints"])
        yield CachedFrame(
            frame["index"],
            frame["timestamp_ns"],
            None,
            frame["width"],
            frame["height"],
            frame["width"],
            frame["height"],
            frame["jpeg"],
            frame["keypoints"],
            np.ones(n, np.float32),
            np.empty((n, 0), np.float32),
            0.0,
        )


@dataclass
class ReconstructionProblem:
    camera: Camera
    frames: list
    frame_rows: np.ndarray
    observation_tracks: list
    track_lengths: np.ndarray
    seed_candidates: list
    provenance: dict

    @classmethod
    def load(cls, path, *, max_seed_candidates=50):
        path = Path(path).expanduser().resolve()
        with GeometryStore(path / "geometry.sqlite3") as store:
            if store.get("phase") != "complete":
                raise ValueError("Finish pair verification and track building before solving")
            with np.load(path / "tracks.npz", allow_pickle=False) as tracks:
                if str(tracks["geometry_identity"]) != store.get("tracks_identity"):
                    raise ValueError("Tracks do not belong to this geometry snapshot")
                rows, offsets = tracks["frame_rows"], tracks["offsets"]
                per_frame = [
                    tracks["track_ids"][a:b].copy()
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True)
                ]
                lengths = tracks["track_lengths"].copy()
            mapping = {int(row): index for index, row in enumerate(rows)}
            ranked = []
            for first, second, value in store.db.execute("SELECT first,second,summary FROM pairs"):
                summary = json.loads(value)
                if summary["seed_eligible"]:
                    ranked.append((first, second, summary))
            ranked.sort(key=lambda edge: (-edge[2]["seed_score"], edge[0], edge[1]))
            seeds = []
            for first, second, summary in ranked[:max_seed_candidates]:
                data, _ = store.pair(first, second)
                seeds.append((mapping[first], mapping[second], data, summary))
            return cls(
                Camera(**store.get("identity")["camera"]),
                list(geometry_frames(store)),
                rows,
                per_frame,
                lengths,
                seeds,
                {
                    "geometry_run": str(path),
                    "geometry_identity": store.get("tracks_identity"),
                    "source_match_run": store.get("source_match_run"),
                    "geometry_settings": store.get("identity")["config"],
                    "source_complete": store.get("source_complete", False),
                    "feature_cache": store.get("feature_cache"),
                },
            )


class GraphMapper(Mapper):
    """Prototype backend: graph seed, all-track PnP, multiview checks and joint BA."""

    def __init__(self, problem, config=None):
        super().__init__(problem.frames, problem.camera, config or ReconstructionConfig())
        self.problem = problem
        self.track_points = np.full(len(problem.track_lengths), -1, int)
        self.track_observations = [{} for _ in problem.track_lengths]
        for frame, ids in enumerate(problem.observation_tracks):
            for keypoint, track in enumerate(ids):
                self.track_observations[track][frame] = keypoint
        self.seed_attempts = []

    def match(self, first, second):
        raise RuntimeError("Graph solver must never recompute appearance matches")

    def candidates(self, current):
        tracks = self.problem.observation_tracks[current]
        valid = self.track_points[tracks] >= 0
        return np.flatnonzero(valid), self.track_points[tracks[valid]]

    def register_from_tracks(self, current):
        state = self.frames[current]
        keys, ids = self.candidates(current)
        state.stats = {"pnp_candidates": len(keys)}
        if len(keys) < self.config.min_pnp_inliers:
            state.status = "insufficient_2d_3d_tracks"
            return False
        points = np.asarray(self.points)[ids]
        xy = state.frame.keypoints[keys]
        trace = {
            "keypoint_ids": keys.copy(),
            "landmark_ids": ids.copy(),
            "xy": xy.copy(),
            "points": points.copy(),
            "references": np.array(self.registered),
            "ransac_ran": False,
            "accepted": False,
        }
        state.pnp_diagnostics = trace
        result = estimate_pose(
            points,
            xy,
            self.camera,
            threshold=self.config.reprojection_threshold,
            min_inliers=self.config.min_pnp_inliers,
            diagnostics=trace,
        )
        if result is None:
            state.status = "pnp_rejected"
            return False
        pose, valid, _ = result
        if coverage(xy[valid], self.camera) < 0.2:
            state.status = "limited_spatial_support"
            trace["accepted"] = False
            return False
        state.pose, state.status = pose, "registered_from_tracks"
        state.stats["pnp_inliers"] = int(valid.sum())
        self.registered.append(current)
        self.incremental_poses[current] = pose.copy()
        for key, point in zip(keys[valid], ids[valid], strict=True):
            self.observations[point][current] = int(key)
            state.landmark_ids[key] = point
        return True

    def initialize_graph(self):
        for first, second, data, summary in self.problem.seed_candidates:
            self.points, self.observations, self.registered = [], [], [first, second]
            self.track_points[:] = -1
            for state in self.frames:
                state.pose = None
                state.landmark_ids[:] = -1
                state.pnp_diagnostics = {}
                state.stats = {}
                state.status = "unregistered"
            self.incremental_poses = {}
            pose = data["pose_second_from_first"]
            indices = data["indices"]
            first_xy = self.frames[first].frame.keypoints[indices[:, 0]]
            second_xy = self.frames[second].frame.keypoints[indices[:, 1]]
            points, valid, _ = triangulate(
                first_xy,
                second_xy,
                np.eye(4),
                pose,
                self.camera,
                min_angle_deg=self.config.min_parallax,
                max_error=self.config.reprojection_threshold,
            )
            tracks_a = self.problem.observation_tracks[first][indices[:, 0]]
            tracks_b = self.problem.observation_tracks[second][indices[:, 1]]
            valid &= data["triangulated"] & (tracks_a == tracks_b)
            if valid.sum() < self.config.min_initial_points:
                self.seed_attempts.append({"rows": [first, second], "reason": "track_conflicts"})
                continue
            self.frames[first].pose, self.frames[second].pose = np.eye(4), pose.copy()
            for (a, b), point, track in zip(
                indices[valid], points[valid], tracks_a[valid], strict=True
            ):
                self.track_points[track] = len(self.points)
                self.add_point(point, first, a, second, b)
            candidates = sorted(
                (row for row in range(len(self.frames)) if row not in (first, second)),
                key=lambda row: (-len(self.candidates(row)[0]), row),
            )
            third = next((row for row in candidates if self.register_from_tracks(row)), None)
            if third is None and len(self.frames) > 2:
                self.seed_attempts.append(
                    {"rows": [first, second], "reason": "no_third_view_support"}
                )
                continue
            self.anchors = (first, second)
            self.frames[first].status = self.frames[second].status = "seed"
            self.incremental_poses.update({first: np.eye(4), second: pose.copy()})
            self.initialization = {
                "frames": [first, second],
                "match_rows": self.problem.frame_rows[[first, second]].tolist(),
                "third_view": third,
                "candidate": summary,
                "rejected_candidates": self.seed_attempts,
            }
            self.seed_diagnostics = {
                "frames": np.array([first, second]),
                "matches": indices,
                "xy_first": first_xy,
                "xy_second": second_xy,
                "ransac_inliers": data["E_inliers"],
                "triangulated": valid,
                "homography_inliers": data["H_inliers"],
                "parallax_deg": data["parallax_deg"],
                "essential": data["E"],
            }
            return
        raise ValueError("No seed survived parallax, track consistency and third-view PnP checks")

    def extend_tracks(self, current):
        """Triangulate with the widest registered baseline; validate every supporting view."""
        state = self.frames[current]
        grouped = {}
        for key, track in enumerate(self.problem.observation_tracks[current]):
            if self.track_points[track] >= 0:
                continue
            observed = self.track_observations[track]
            references = [r for r in observed if r != current and self.frames[r].pose is not None]
            if not references:
                continue
            ref = max(
                references,
                key=lambda r: np.linalg.norm(
                    camera_center(self.frames[r].pose) - camera_center(state.pose)
                ),
            )
            grouped.setdefault(ref, []).append((track, observed[ref], key))
        for ref, entries in grouped.items():
            ids, keys_a, keys_b = np.asarray(entries).T
            points, valid, _ = triangulate(
                self.frames[ref].frame.keypoints[keys_a],
                state.frame.keypoints[keys_b],
                self.frames[ref].pose,
                state.pose,
                self.camera,
                min_angle_deg=self.config.min_parallax,
                max_error=self.config.reprojection_threshold,
            )
            for track, point in zip(ids[valid], points[valid], strict=True):
                support = {}
                for row, key in self.track_observations[track].items():
                    if self.frames[row].pose is None:
                        continue
                    xy, depth = project(point[None], self.frames[row].pose, self.camera)
                    error = np.linalg.norm(xy[0] - self.frames[row].frame.keypoints[key])
                    if depth[0] > 0 and error <= self.config.reprojection_threshold:
                        support[row] = key
                if len(support) < 2:
                    continue
                landmark = len(self.points)
                self.track_points[track] = landmark
                self.points.append(point)
                self.observations.append(support)
                for row, key in support.items():
                    self.frames[row].landmark_ids[key] = landmark

    def run(self, progress=True):
        cv2.setRNGSeed(self.config.seed)
        self.initialize_graph()
        for row in list(self.registered):
            self.extend_tracks(row)
        remaining = set(range(len(self.frames))) - set(self.registered)
        with tqdm(
            total=len(self.frames),
            initial=len(self.registered),
            desc="Solving graph",
            disable=not progress,
        ) as bar:
            while remaining:
                candidates = sorted(remaining, key=lambda row: (-len(self.candidates(row)[0]), row))
                accepted = next((row for row in candidates if self.register_from_tracks(row)), None)
                if accepted is None:
                    break
                remaining.remove(accepted)
                self.extend_tracks(accepted)
                bar.update(1)


def solve_geometry(path, output, *, config=None, progress=True):
    from slam_lab.reconstruction_io import save_reconstruction

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    problem = ReconstructionProblem.load(path)
    mapper = GraphMapper(problem, config)
    mapper.run(progress=progress)
    if progress:
        tqdm.write(
            f"Registered {len(mapper.registered)}/{len(mapper.frames)} views; refining points"
        )
    points, observations, xy, errors, adjustment = mapper.finalize()
    metadata = {
        "schema_version": 1,
        "backend": "verified-track-graph-incremental-v1",
        **problem.provenance,
        "camera": asdict(problem.camera),
        "camera_assumption": "shared fixed pinhole, zero distortion; "
        "focal length assumed unless supplied",
        "settings": asdict(mapper.config),
        "initialization": mapper.initialization,
        "bundle_adjustment": adjustment,
        "frames": len(mapper.frames),
        "registered_frames": len(mapper.registered),
        "points": len(points),
        "observations": len(observations),
        "median_reprojection_error_px": float(np.median(errors)),
        "p95_reprojection_error_px": float(np.percentile(errors, 95)),
        "pose_convention": "T_world_camera maps camera to world; axes right, down, forward",
        "scale": "arbitrary; seed baseline is one unit, not meters",
        "limitations": [
            "prototype graph-based incremental backend; final bundle adjustment only",
            "one connected component; disconnected or rejected images have no pose",
            "pairwise geometry does not guarantee static-scene matches",
            "no track splitting or alternate global solver yet",
        ],
    }
    save_reconstruction(output, mapper, points, observations, xy, errors, metadata)
    return output
