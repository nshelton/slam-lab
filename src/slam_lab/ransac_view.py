"""Persist and display actual solver decisions, separately from post-BA tracks."""

import json

import numpy as np

GREEN = [40, 235, 95]
RED = [255, 65, 80]
GRAY = [160, 160, 160]


def save_diagnostics(root, mapper):
    arrays = {f"seed/{key}": value for key, value in mapper.seed_diagnostics.items()}
    incremental = np.full((len(mapper.frames), 4, 4), np.nan)
    for row, pose in mapper.incremental_poses.items():
        incremental[row] = np.linalg.inv(pose)
    arrays["incremental_T_world_camera"] = incremental
    for row, state in enumerate(mapper.frames):
        for key, value in state.pnp_diagnostics.items():
            arrays[f"pnp/{row}/{key}"] = np.asarray(value)
        if state.pnp_diagnostics:
            arrays[f"pnp/{row}/final_landmark_ids"] = mapper.final_point_ids[
                state.pnp_diagnostics["landmark_ids"]
            ]
    np.savez_compressed(root / "ransac.npz", **arrays)
    (root / "ransac.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": "Captured directly from OpenCV during this reconstruction",
                "pnp": "Per selected-frame row; 2D-3D inputs frozen before map extension/BA",
                "ransac_inliers": "Exact indices returned by solvePnPRansac, before LM refinement",
                "refined_inliers": "Reprojection/depth mask after LM, before map extension/BA",
                "seed": "Raw findEssentialMat mask; triangulation eligibility is separate",
                "landmark_ids": "Incremental IDs; final_landmark_ids maps to cloud (-1 if removed)",
                "poses": "raw_pose/refined_pose: world to camera; incremental_T: camera to world",
                "legend": {
                    "green": "RANSAC accepted",
                    "red": "RANSAC rejected",
                    "gray": "not tested",
                },
                "warning": "Inliers can include moving objects and fit incorrect poses",
            },
            indent=2,
        )
        + "\n"
    )


def load_diagnostics(path):
    if not (path / "ransac.npz").exists():
        return None
    with np.load(path / "ransac.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def pnp_record(data, row):
    prefix = f"pnp/{row}/"
    return {key[len(prefix) :]: value for key, value in data.items() if key.startswith(prefix)}


def log_pnp(data, row, frame, status):
    import rerun as rr

    trace = pnp_record(data, row)
    rr.log("ransac/image/decisions", rr.Clear(recursive=True))
    rr.log("fit/image/residuals", rr.Clear(recursive=True))
    xy = trace.get("xy", np.empty((0, 2)))
    tested = bool(trace.get("ransac_ran", False))
    raw = trace.get("ransac_inliers", np.zeros(len(xy), bool))
    refined = trace.get("refined_inliers", np.zeros(len(xy), bool))
    errors = trace.get("errors", np.full(len(xy), np.nan))
    raw_errors = trace.get("raw_errors", np.full(len(xy), np.nan))
    for name, chosen, color in (
        ("inliers", raw if tested else np.zeros(len(xy), bool), GREEN),
        ("outliers", ~raw if tested else np.zeros(len(xy), bool), RED),
        ("untested", np.zeros(len(xy), bool) if tested else np.ones(len(xy), bool), GRAY),
    ):
        if chosen.any():
            rr.log(
                f"ransac/image/decisions/{name}",
                rr.Points2D(xy[chosen], colors=color, radii=3),
                rr.AnyValues(
                    keypoint_id=trace["keypoint_ids"][chosen],
                    incremental_landmark_id=trace["landmark_ids"][chosen],
                    final_landmark_id=trace["final_landmark_ids"][chosen],
                    raw_error_px=raw_errors[chosen],
                    refined_error_px=errors[chosen],
                    refined_inlier=refined[chosen],
                ),
            )
    projected = trace.get("projected_xy")
    if projected is not None:
        # Only draw segments whose projections are inside the image; attributes retain all errors.
        visible = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < frame.width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < frame.height)
        )
        if visible.any():
            colors = np.where(refined[visible, None], np.array(GREEN), np.array(RED))
            rr.log(
                "fit/image/residuals/segments",
                rr.LineStrips2D(np.stack([xy[visible], projected[visible]], axis=1), colors=colors),
            )
            rr.log("fit/image/residuals/observed", rr.Points2D(xy[visible], colors=colors, radii=2))
    counts = {
        "candidates": len(xy),
        "inliers": int(raw.sum()) if tested else 0,
        "outliers": int((~raw).sum()) if tested else 0,
        "refined_inliers": int(refined.sum()),
    }
    for key, value in counts.items():
        rr.log(f"metrics/ransac/{key}", rr.Scalars(value))
    description = {
        "frame_index": frame.index,
        "time_seconds": frame.timestamp_ns / 1e9,
        "stage": "PnP RANSAC" if tested else "No PnP RANSAC for this frame",
        "status": status,
        "pose_accepted": bool(trace.get("accepted", False)),
        **counts,
        "reference_frame_rows": trace.get("references", np.empty(0, int)).tolist(),
        "legend": "GREEN = raw RANSAC inlier; RED = raw RANSAC outlier; GRAY = not tested",
        "trajectory_colors": "CYAN = before bundle adjustment; YELLOW = after",
        "note": "Click points for errors/IDs. Other tabs show retained tracks and LM residuals.",
    }
    rr.log("ransac_status", rr.TextDocument(json.dumps(description, indent=2)))


def log_seed(data, frame, row):
    import rerun as rr

    seed_frames = data.get("seed/frames", [])
    if row not in seed_frames:
        return
    side = "first" if row == seed_frames[0] else "second"
    root = f"seed_{side}/image"
    xy, raw = data[f"seed/xy_{side}"], data["seed/ransac_inliers"]
    for name, chosen, color in (("inliers", raw, GREEN), ("outliers", ~raw, RED)):
        if chosen.any():
            rr.log(
                f"{root}/{name}",
                rr.Points2D(xy[chosen], colors=color, radii=3),
                rr.AnyValues(
                    triangulated=data["seed/triangulated"][chosen],
                    homography_inlier=data["seed/homography_inliers"][chosen],
                    parallax_deg=data["seed/parallax_deg"][chosen],
                ),
                static=True,
            )
