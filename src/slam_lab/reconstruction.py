"""Incremental sparse monocular reconstruction from cached SuperPoint observations."""

import json
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from slam_lab.cache import CachedFrame, FrameCache, resolve_cache
from slam_lab.geometry import (
    Camera,
    estimate_pose,
    initialize_pair,
    match_descriptors,
    project,
    triangulate,
)


@dataclass(frozen=True)
class ReconstructionConfig:
    ratio: float = 0.8
    essential_threshold: float = 1.5
    reprojection_threshold: float = 3.0
    min_parallax: float = 1.0
    min_initial_points: int = 30
    min_pnp_inliers: int = 20
    min_track_length: int = 3
    keyframe_interval: int = 5
    bundle_evaluations: int = 30
    seed: int = 7

    def __post_init__(self):
        if not 0 < self.ratio < 1:
            raise ValueError("ratio must be between 0 and 1")
        for name in ("essential_threshold", "reprojection_threshold", "min_parallax"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_parallax >= 90:
            raise ValueError("min_parallax must be less than 90 degrees")
        if self.min_initial_points < 8 or self.min_pnp_inliers < 6:
            raise ValueError("Need at least 8 initial points and 6 PnP inliers")
        if self.min_track_length < 2 or self.keyframe_interval < 1 or self.bundle_evaluations < 0:
            raise ValueError("Invalid track length, keyframe interval, or bundle evaluation limit")
        if not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("seed must be a nonnegative signed 32-bit integer")


@dataclass
class FrameState:
    frame: CachedFrame
    pose: np.ndarray | None = None
    landmark_ids: np.ndarray = field(init=False)
    status: str = "waiting_for_initialization"
    stats: dict = field(default_factory=dict)
    pnp_diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.landmark_ids = np.full(len(self.frame.keypoints), -1, dtype=int)


class Mapper:
    def __init__(self, frames, camera, config):
        self.frames = [FrameState(frame) for frame in frames]
        self.camera, self.config = camera, config
        self.points = []
        self.observations = []
        self.keyframes = []
        self.registered = []
        self.anchors = None
        self.initialization = None
        self.pair_counts = []
        self.seed_diagnostics = {}
        self.incremental_poses = {}

    def match(self, first, second):
        matches = match_descriptors(
            self.frames[first].frame.descriptors,
            self.frames[second].frame.descriptors,
            self.config.ratio,
        )
        self.pair_counts.append({"first": first, "second": second, "matches": len(matches)})
        return matches

    def add_point(self, point, first, first_keypoint, second, second_keypoint):
        index = len(self.points)
        self.points.append(point)
        self.observations.append({first: int(first_keypoint), second: int(second_keypoint)})
        self.frames[first].landmark_ids[first_keypoint] = index
        self.frames[second].landmark_ids[second_keypoint] = index

    def bootstrap(self, current):
        # Try wider baselines first. A close pair often has many matches but unstable depth.
        references = list(dict.fromkeys(max(0, current - delta) for delta in (10, 20, 30, 5, 1)))
        for reference in references:
            if reference == current:
                continue
            first, second = self.frames[reference], self.frames[current]
            matches = self.match(reference, current)
            diagnostics = {}
            result = initialize_pair(
                first.frame.keypoints[matches[:, 0]],
                second.frame.keypoints[matches[:, 1]],
                self.camera,
                threshold=self.config.essential_threshold,
                min_inliers=self.config.min_initial_points,
                min_angle_deg=self.config.min_parallax,
                max_error=self.config.reprojection_threshold,
                diagnostics=diagnostics,
            )
            if result is None:
                continue
            pose, points, valid, stats = result
            first.pose, second.pose = np.eye(4), pose
            first.status = second.status = "seed"
            self.registered = [reference, current]
            self.keyframes = [reference, current]
            self.anchors = (reference, current)
            self.initialization = {"frames": [reference, current], **stats}
            self.seed_diagnostics = {
                "frames": np.array([reference, current]),
                "matches": matches.copy(),
                **diagnostics,
            }
            self.incremental_poses.update(
                {reference: first.pose.copy(), current: second.pose.copy()}
            )
            for (a, b), point in zip(matches[valid], points[valid], strict=True):
                self.add_point(point, reference, a, current, b)
            return True
        return False

    def references(self):
        return list(
            dict.fromkeys([self.registered[-1], *self.keyframes[-3:], *self.keyframes[-6:-5]])
        )

    def register(self, current, *, references=None, triangulate_new=True):
        state = self.frames[current]
        if len(state.frame.keypoints) < self.config.min_pnp_inliers:
            state.status = "insufficient_features"
            return False
        references = self.references() if references is None else references
        pairs = {ref: self.match(ref, current) for ref in references if ref != current}
        # Conflicting identities are discarded; a descriptor is not a permanent ID.
        candidates = {}
        for ref, matches in pairs.items():
            ids = self.frames[ref].landmark_ids
            for previous, keypoint in matches:
                if ids[previous] >= 0:
                    candidates.setdefault(int(keypoint), set()).add(int(ids[previous]))
        by_point = {}
        for keypoint, ids in candidates.items():
            if len(ids) == 1:
                by_point.setdefault(next(iter(ids)), []).append(keypoint)
        associations = [(keys[0], point) for point, keys in by_point.items() if len(keys) == 1]
        candidates_array = np.asarray(associations, dtype=int).reshape(-1, 2)
        state.pnp_diagnostics = {
            "keypoint_ids": candidates_array[:, 0],
            "landmark_ids": candidates_array[:, 1],
            "xy": state.frame.keypoints[candidates_array[:, 0]].copy(),
            "points": np.asarray(self.points)[candidates_array[:, 1]].copy(),
            "references": np.array(list(pairs), dtype=int),
            "ransac_ran": False,
            "accepted": False,
        }
        state.stats = {
            "matches": sum(map(len, pairs.values())),
            "pnp_candidates": len(associations),
        }
        if len(associations) < self.config.min_pnp_inliers:
            state.status = "insufficient_2d_3d_matches"
            return False
        keypoints, point_ids = np.asarray(associations).T
        points = np.asarray(self.points)[point_ids]
        result = estimate_pose(
            points,
            state.frame.keypoints[keypoints],
            self.camera,
            threshold=self.config.reprojection_threshold,
            min_inliers=self.config.min_pnp_inliers,
            diagnostics=state.pnp_diagnostics,
        )
        state.stats["pnp_ransac_inliers"] = int(
            np.count_nonzero(state.pnp_diagnostics.get("ransac_inliers", []))
        )
        if result is None:
            state.status = "pnp_ransac_failed"
            return False
        state.pose, valid, errors = result
        self.incremental_poses[current] = state.pose.copy()
        state.status = "registered"
        state.stats.update(
            pnp_inliers=int(valid.sum()), median_error_px=float(np.median(errors[valid]))
        )
        for keypoint, point in zip(keypoints[valid], point_ids[valid], strict=True):
            state.landmark_ids[keypoint] = point
            self.observations[point][current] = int(keypoint)
        if triangulate_new:
            for ref, matches in pairs.items():
                self.extend_map(ref, current, matches)
        self.registered.append(current)
        if triangulate_new and current - self.keyframes[-1] >= self.config.keyframe_interval:
            self.keyframes.append(current)
        return True

    def extend_map(self, reference, current, matches):
        first, second = self.frames[reference], self.frames[current]
        # Existing landmarks can acquire observations from either side of a match.
        for source, target, source_col, target_col, target_index in (
            (first, second, 0, 1, current),
            (second, first, 1, 0, reference),
        ):
            used = set(target.landmark_ids[target.landmark_ids >= 0].tolist())
            for pair in matches:
                point = int(source.landmark_ids[pair[source_col]])
                keypoint = int(pair[target_col])
                if point < 0 or point in used or target.landmark_ids[keypoint] >= 0:
                    continue
                projected, depth = project(
                    np.asarray([self.points[point]]), target.pose, self.camera
                )
                error = np.linalg.norm(projected[0] - target.frame.keypoints[keypoint])
                if depth[0] > 0 and error <= self.config.reprojection_threshold:
                    target.landmark_ids[keypoint] = point
                    self.observations[point][target_index] = keypoint
                    used.add(point)
        unused = (first.landmark_ids[matches[:, 0]] < 0) & (second.landmark_ids[matches[:, 1]] < 0)
        matches = matches[unused]
        points, valid, _ = triangulate(
            first.frame.keypoints[matches[:, 0]],
            second.frame.keypoints[matches[:, 1]],
            first.pose,
            second.pose,
            self.camera,
            max_error=self.config.reprojection_threshold,
            min_angle_deg=self.config.min_parallax,
        )
        for (a, b), point in zip(matches[valid], points[valid], strict=True):
            self.add_point(point, reference, a, current, b)

    def run(self, progress=True):
        cv2.setRNGSeed(self.config.seed)
        for current, state in enumerate(
            tqdm(self.frames, desc="Reconstructing", disable=not progress)
        ):
            if len(state.frame.keypoints) < self.config.min_pnp_inliers:
                state.status = "insufficient_features"
                continue
            if self.anchors is None:
                if self.bootstrap(current):
                    # Extend the map backwards through local overlapping views. Direct PnP
                    # against a distant seed pair can accept weak, inconsistent early poses.
                    backward_keys = list(self.anchors)
                    for previous in range(current - 1, -1, -1):
                        if self.frames[previous].pose is None:
                            references = list(
                                dict.fromkeys([self.registered[-1], *backward_keys[-3:]])
                            )
                            if self.register(previous, references=references):
                                if (
                                    abs(previous - backward_keys[-1])
                                    >= self.config.keyframe_interval
                                ):
                                    backward_keys.append(previous)
                    self.registered = sorted(self.registered)
            else:
                self.register(current)
        if self.anchors is None:
            raise ValueError(
                "No reliable two-view initialization: need overlapping, textured views "
                "with translation and parallax. Try more frames or a different focal length."
            )

    def finalize(self):
        # Remove short-lived tentative points before optimization.
        minimum = 2 if len(self.registered) == 2 else self.config.min_track_length
        selected = [i for i, obs in enumerate(self.observations) if len(obs) >= minimum]
        if not selected:
            raise ValueError(
                "No multi-view tracks survived. Try a longer segment or different intrinsics."
            )
        points = np.asarray(self.points)[selected]
        observations = [self.observations[i] for i in selected]
        rows = np.array(
            [
                (frame, point, keypoint)
                for point, obs in enumerate(observations)
                for frame, keypoint in sorted(obs.items())
            ],
            dtype=int,
        )
        xy = np.array([self.frames[frame].frame.keypoints[keypoint] for frame, _, keypoint in rows])
        poses = {i: self.frames[i].pose for i in self.registered}
        report = {"enabled": False}
        if self.config.bundle_evaluations:
            from slam_lab.bundle import bundle_adjust

            poses, points, report = bundle_adjust(
                poses,
                points,
                rows[:, 0],
                rows[:, 1],
                xy,
                self.camera,
                anchors=self.anchors,
                max_evaluations=self.config.bundle_evaluations,
            )
        errors = np.empty(len(rows))
        positive = np.zeros(len(rows), bool)
        for frame, pose in poses.items():
            chosen = rows[:, 0] == frame
            projected, depth = project(points[rows[chosen, 1]], pose, self.camera)
            errors[chosen] = np.linalg.norm(projected - xy[chosen], axis=1)
            positive[chosen] = depth > 0
            self.frames[frame].pose = pose
        valid = positive & np.isfinite(errors) & (errors <= self.config.reprojection_threshold)
        counts = np.bincount(rows[valid, 1], minlength=len(points))
        keep = counts >= minimum
        remap = np.cumsum(keep) - 1
        valid &= keep[rows[:, 1]]
        rows, xy, errors = rows[valid], xy[valid], errors[valid]
        rows[:, 1] = remap[rows[:, 1]]
        self.final_point_ids = np.full(len(self.points), -1, dtype=int)
        self.final_point_ids[np.asarray(selected)[keep]] = np.arange(np.count_nonzero(keep))
        points = points[keep]
        if not len(points):
            raise ValueError("No points survived final reprojection and track-length filtering")
        return points, rows, xy, errors, report


def reconstruct(
    cache_path: Path,
    output: Path,
    *,
    config=None,
    camera_options=None,
    max_frames=300,
    frame_step=1,
    progress=True,
):
    from slam_lab.reconstruction_io import save_reconstruction

    if max_frames < 2 or frame_step < 1:
        raise ValueError("max-frames must be at least 2 and frame-step at least 1")
    config = config or ReconstructionConfig()
    cache_path = resolve_cache(cache_path.expanduser()).resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new reconstruction directory."
        )
    with FrameCache(cache_path) as cache:
        frames = list(islice(cache.frames(), 0, max_frames * frame_step, frame_step))
        source = cache.get("source_path")
        identity = cache.get("identity")
        available = cache.count()
    if len(frames) < 2:
        raise ValueError("Reconstruction requires at least two cached frames")
    size = frames[0].width, frames[0].height
    if any((frame.width, frame.height) != size for frame in frames):
        raise ValueError("Fixed-intrinsic reconstruction requires a constant cached image size")
    camera = Camera.from_size(*size, **(camera_options or {}))
    mapper = Mapper(frames, camera, config)
    mapper.run(progress=progress)
    if progress:
        tqdm.write(
            f"Registered {len(mapper.registered)}/{len(frames)} cameras; "
            f"refining {len(mapper.points)} tentative points"
        )
    points, observations, xy, errors, adjustment = mapper.finalize()
    metadata = {
        "schema_version": 1,
        "source": source,
        "feature_cache": str(cache_path),
        "extraction_identity": identity,
        "cached_frames_available": available,
        "selection": {"max_frames": max_frames, "frame_step": frame_step},
        "camera": asdict(camera),
        "camera_assumption": "fixed pinhole; zero distortion",
        "intrinsics_source": "explicit focal length"
        if (camera_options or {}).get("fx") is not None
        else "assumed horizontal field of view",
        "horizontal_fov_deg": float(np.rad2deg(2 * np.arctan(camera.width / (2 * camera.fx)))),
        "settings": asdict(config),
        "matcher": "mutual L2 nearest neighbors with symmetric ratio test",
        "opencv_version": cv2.__version__,
        "initialization": mapper.initialization,
        "bundle_adjustment": adjustment,
        "frames": len(frames),
        "registered_frames": len(mapper.registered),
        "points": len(points),
        "observations": len(observations),
        "median_reprojection_error_px": float(np.median(errors)),
        "p95_reprojection_error_px": float(np.percentile(errors, 95)),
        "scale": "arbitrary; seed-camera baseline is one unit, not meters",
        "pose_convention": "T_world_camera maps camera to world; camera axes right, down, forward",
        "limitations": [
            "intrinsics assumed, not calibrated",
            "no loop closure or global relocalization",
            "static-scene assumption; moving objects and video edits can break tracking",
            "failed frames have no pose; trajectory gaps are not interpolated",
        ],
    }
    save_reconstruction(output, mapper, points, observations, xy, errors, metadata)
    print(
        json.dumps(
            {
                key: metadata[key]
                for key in ("registered_frames", "frames", "points", "median_reprojection_error_px")
            },
            indent=2,
        )
    )
    return output
