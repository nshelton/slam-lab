"""Sparse robust bundle adjustment with fixed intrinsics and explicit gauge anchors."""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def normalize_baseline(poses, points, anchors, baseline=1.0):
    """Remove residual scale-gauge drift without changing any image projections."""
    first, second = anchors
    origin = -poses[first][:3, :3].T @ poses[first][:3, 3]
    center = -poses[second][:3, :3].T @ poses[second][:3, 3]
    scale = baseline / np.linalg.norm(center - origin)
    adjusted = origin + scale * (points - origin)
    matrices = {}
    for index, pose in poses.items():
        matrix = pose.copy()
        center = -pose[:3, :3].T @ pose[:3, 3]
        matrix[:3, 3] = -pose[:3, :3] @ (origin + scale * (center - origin))
        matrices[index] = matrix
    return matrices, adjusted


def bundle_adjust(
    poses,
    points,
    observation_frames,
    observation_points,
    xy,
    camera,
    *,
    anchors,
    max_evaluations=30,
):
    """Fix the first camera and constrain the seed baseline to preserve world scale."""
    fixed, scale_camera = anchors
    variable_frames = [index for index in sorted(poses) if index != fixed]
    columns = {index: row for row, index in enumerate(variable_frames)}
    camera_parameters = np.array(
        [
            np.r_[cv2.Rodrigues(poses[index][:3, :3])[0].ravel(), poses[index][:3, 3]]
            for index in variable_frames
        ]
    )
    camera_size = camera_parameters.size
    initial = np.r_[camera_parameters.ravel(), points.ravel()]
    frame_ids = np.asarray(observation_frames, int)
    point_ids = np.asarray(observation_points, int)
    groups = {index: np.flatnonzero(frame_ids == index) for index in poses}
    origin = -poses[fixed][:3, :3].T @ poses[fixed][:3, 3]
    baseline = np.linalg.norm(-poses[scale_camera][:3, :3].T @ poses[scale_camera][:3, 3] - origin)
    sparsity = lil_matrix((2 * len(xy) + 1, len(initial)), dtype=np.int8)
    rows = 2 * np.arange(len(xy))
    for component in range(3):
        sparsity[rows, camera_size + 3 * point_ids + component] = 1
        sparsity[rows + 1, camera_size + 3 * point_ids + component] = 1
    for index, selected in groups.items():
        if index == fixed:
            continue
        for component in range(6):
            sparsity[2 * selected, columns[index] * 6 + component] = 1
            sparsity[2 * selected + 1, columns[index] * 6 + component] = 1
    sparsity[-1, columns[scale_camera] * 6 : columns[scale_camera] * 6 + 6] = 1

    def unpack(parameters):
        matrices = {fixed: poses[fixed].copy()}
        for index, row in columns.items():
            vector = parameters[row * 6 : row * 6 + 6]
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = cv2.Rodrigues(vector[:3])[0], vector[3:]
            matrices[index] = matrix
        return matrices, parameters[camera_size:].reshape(-1, 3)

    def residual(parameters):
        matrices, world = unpack(parameters)
        error = np.empty((len(xy), 2))
        for index, selected in groups.items():
            pose = matrices[index]
            local = world[point_ids[selected]] @ pose[:3, :3].T + pose[:3, 3]
            depth = np.maximum(local[:, 2:3], 1e-6)
            projected = local[:, :2] / depth * [camera.fx, camera.fy] + [camera.cx, camera.cy]
            error[selected] = projected - xy[selected]
        scale_pose = matrices[scale_camera]
        length = np.linalg.norm(-scale_pose[:3, :3].T @ scale_pose[:3, 3] - origin)
        return np.r_[error.ravel(), 1000 * (length / baseline - 1)]

    before = residual(initial)
    fit = least_squares(
        residual,
        initial,
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_evaluations,
        ftol=1e-4,
        tr_solver="lsmr",
    )
    initial_cost = float(np.sum(np.sqrt(1 + before**2) - 1))
    accepted = bool(np.isfinite(fit.x).all() and fit.cost <= initial_cost)
    matrices, adjusted = unpack(fit.x if accepted else initial)
    # The robust loss also acts on the gauge penalty; enforce the exact output convention.
    matrices, adjusted = normalize_baseline(matrices, adjusted, anchors, baseline)
    after = residual(fit.x if accepted else initial)
    report = {
        "enabled": True,
        "accepted": accepted,
        "converged": bool(fit.success),
        "evaluations": int(fit.nfev),
        "message": str(fit.message),
        "median_error_before_px": float(
            np.median(np.linalg.norm(before[:-1].reshape(-1, 2), axis=1))
        ),
        "median_error_after_px": float(
            np.median(np.linalg.norm(after[:-1].reshape(-1, 2), axis=1))
        ),
        "gauge": "first seed pose fixed; seed baseline normalized exactly; intrinsics fixed",
    }
    return matrices, adjusted, report
