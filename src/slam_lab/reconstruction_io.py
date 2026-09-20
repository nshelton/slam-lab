"""Portable reconstruction artifacts and synchronized 2D/3D Rerun inspection."""

import json
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from slam_lab.cache import FrameCache, resolve_cache
from slam_lab.ransac_view import load_diagnostics, log_pnp, log_seed, save_diagnostics


def reconstruction_frames(metadata, cache_path=None):
    if metadata.get("geometry_run") and cache_path is None:
        from slam_lab.solver import geometry_frames
        from slam_lab.verification import GeometryStore

        with GeometryStore(Path(metadata["geometry_run"]) / "geometry.sqlite3") as store:
            yield from geometry_frames(store)
    else:
        cache_path = resolve_cache(cache_path or Path(metadata["feature_cache"]))
        with FrameCache(cache_path) as cache:
            yield from cache.frames(descriptors=False)


def save_reconstruction(output, mapper, points, observations, xy, errors, metadata):
    states = mapper.frames
    transforms = np.full((len(states), 4, 4), np.nan)
    valid = np.array([state.pose is not None for state in states])
    for index in np.flatnonzero(valid):
        transforms[index] = np.linalg.inv(states[index].pose)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colored = np.zeros(len(points), bool)
    for frame_id, state in enumerate(states):
        selected = observations[:, 0] == frame_id
        ids, pixels = observations[selected, 1], xy[selected]
        fresh = ~colored[ids]
        if not fresh.any():
            continue
        image = np.asarray(Image.open(BytesIO(state.frame.jpeg)).convert("RGB"))
        pixels = np.rint(pixels[fresh]).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, image.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, image.shape[0] - 1)
        colors[ids[fresh]] = image[pixels[:, 1], pixels[:, 0]]
        colored[ids[fresh]] = True
    support = np.bincount(observations[:, 1], minlength=len(points))
    frame_support = np.bincount(observations[:, 0], minlength=len(states))
    poses = [
        {
            "frame_index": state.frame.index,
            "timestamp_ns": state.frame.timestamp_ns,
            "registered": bool(valid[index]),
            "status": state.status,
            "retained_observations": int(frame_support[index]),
            **state.stats,
            "T_world_camera": transforms[index].tolist() if valid[index] else None,
        }
        for index, state in enumerate(states)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        root = Path(temporary)
        np.savez_compressed(
            root / "reconstruction.npz",
            points=points,
            colors=colors,
            track_lengths=support,
            T_world_camera=transforms,
            registered=valid,
            frame_indices=np.array([state.frame.index for state in states]),
            timestamps_ns=np.array([state.frame.timestamp_ns for state in states], dtype=np.int64),
            K=mapper.camera.K,
            observations=observations,
            observation_xy=xy,
            reprojection_errors=errors,
        )
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (root / "poses.json").write_text(json.dumps(poses, indent=2) + "\n")
        (root / "matching.json").write_text(json.dumps(mapper.pair_counts, indent=2) + "\n")
        save_diagnostics(root, mapper)
        with (root / "points.ply").open("w") as file:
            file.write(
                f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
                "property double x\nproperty double y\nproperty double z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            )
            np.savetxt(file, np.column_stack((points, colors)), fmt="%.9g %.9g %.9g %d %d %d")
        # Publish only a complete artifact set. Existing runs are never overwritten.
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        root.rename(output)


def view_reconstruction(path: Path, *, output: Path | None = None, cache_path: Path | None = None):
    import rerun as rr
    import rerun.blueprint as rrb

    path = path.expanduser().resolve()
    metadata = json.loads((path / "metadata.json").read_text())
    poses = json.loads((path / "poses.json").read_text())
    diagnostics = load_diagnostics(path)
    with np.load(path / "reconstruction.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if output is not None:
        output = output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    from slam_lab.rerun_support import init_recording

    recording = init_recording("slam-lab-reconstruction", output=output)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Reconstruction (arbitrary scale)", origin="/world"),
                rrb.Vertical(
                    rrb.Tabs(
                        rrb.Spatial2DView(name="PnP RANSAC: green in / red out", origin="/ransac"),
                        rrb.Spatial2DView(name="Fit residuals (after LM)", origin="/fit"),
                        rrb.Spatial2DView(name="Retained tracks (after BA)", origin="/image"),
                        rrb.Spatial2DView(
                            name="Seed first: essential RANSAC", origin="/seed_first"
                        ),
                        rrb.Spatial2DView(
                            name="Seed second: essential RANSAC", origin="/seed_second"
                        ),
                    )
                    if diagnostics is not None
                    else rrb.Spatial2DView(name="Retained tracks", origin="/image"),
                    rrb.TimeSeriesView(name="Tracking support", origin="/metrics"),
                    rrb.TextDocumentView(
                        name="Frame status",
                        origin="/ransac_status" if diagnostics is not None else "/status",
                    ),
                    row_shares=[4, 1, 1],
                ),
                column_shares=[2, 3],
            ),
            rrb.TimePanel(timeline="video_time"),
        )
    )
    try:
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        rr.log(
            "world/points",
            rr.Points3D(data["points"], colors=data["colors"], radii=0.015),
            rr.AnyValues(track_length=data["track_lengths"]),
            static=True,
        )
        # Split the trajectory at missing poses so a lost interval is visible.
        strips, current = [], []
        for valid, transform in zip(data["registered"], data["T_world_camera"], strict=True):
            if valid:
                current.append(transform[:3, 3])
            else:
                if len(current) > 1:
                    strips.append(np.array(current))
                current = []
        if len(current) > 1:
            strips.append(np.array(current))
        if strips:
            rr.log("world/trajectory", rr.LineStrips3D(strips, colors=[255, 190, 40]), static=True)
        if diagnostics is not None:
            incremental = diagnostics["incremental_T_world_camera"]
            strips, current = [], []
            for pose in incremental:
                if np.isfinite(pose).all():
                    current.append(pose[:3, 3])
                else:
                    if len(current) > 1:
                        strips.append(np.array(current))
                    current = []
            if len(current) > 1:
                strips.append(np.array(current))
            if strips:
                rr.log(
                    "world/trajectory_before_ba",
                    rr.LineStrips3D(strips, colors=[80, 180, 255]),
                    static=True,
                )
            rr.log(
                "ransac_legend",
                rr.TextDocument(
                    "Green: raw RANSAC inliers. Red: rejected candidates. Gray: not tested.\n"
                    "Trajectory: cyan before bundle adjustment; yellow after.\n"
                    "An inlier fits a model; it does not establish that an object is static."
                ),
                static=True,
            )
        rr.log("info", rr.TextDocument(json.dumps(metadata, indent=2)), static=True)
        by_index = {int(index): row for row, index in enumerate(data["frame_indices"])}
        camera = metadata["camera"]
        for frame in reconstruction_frames(metadata, cache_path):
            row = by_index.get(frame.index)
            if row is None:
                continue
            rr.set_time("frame", sequence=frame.index)
            rr.set_time("video_time", duration=np.timedelta64(frame.timestamp_ns, "ns"))
            encoded = rr.EncodedImage(contents=frame.jpeg, media_type="image/jpeg")
            rr.log("image", encoded)
            if diagnostics is not None:
                log_pnp(diagnostics, row, frame, poses[row]["status"])
                log_seed(diagnostics, frame, row)
            observed = data["observations"][:, 0] == row
            point_ids = data["observations"][observed, 1]
            pixels = data["observation_xy"][observed]
            if len(pixels):
                rr.log(
                    "image/tracks",
                    rr.Points2D(pixels, colors=[50, 230, 100], radii=2),
                    rr.AnyValues(
                        landmark_id=point_ids,
                        reprojection_error=data["reprojection_errors"][observed],
                    ),
                )
            else:
                rr.log("image/tracks", rr.Clear(recursive=False))
            if data["registered"][row]:
                transform = data["T_world_camera"][row]
                rr.log(
                    "world/camera",
                    rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
                )
                rr.log(
                    "world/camera/pinhole",
                    rr.Pinhole(
                        image_from_camera=data["K"],
                        resolution=[camera["width"], camera["height"]],
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=0.3,
                    ),
                )
                rr.log("world/camera/pinhole/image", encoded)
            else:
                rr.log("world/camera", rr.Clear(recursive=True))
            rr.log("metrics/retained_observations", rr.Scalars(len(pixels)))
            rr.log("metrics/pnp_inliers", rr.Scalars(poses[row].get("pnp_inliers", 0)))
            rr.log("status", rr.TextDocument(json.dumps(poses[row], indent=2)))
        recording.flush()
    finally:
        rr.disconnect()
    return len(poses)
