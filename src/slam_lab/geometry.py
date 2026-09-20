"""Fixed-intrinsic monocular geometry. Poses map world coordinates into cameras."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError("Camera dimensions must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera parameters must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")

    @classmethod
    def from_size(cls, width, height, *, fov_deg=60.0, fx=None, fy=None, cx=None, cy=None):
        if not np.isfinite(fov_deg) or not 1 < fov_deg < 179:
            raise ValueError("Horizontal field of view must be between 1 and 179 degrees")
        if fy is not None and fx is None:
            raise ValueError("Supply --fx together with --fy")
        focal = width / (2 * np.tan(np.deg2rad(fov_deg) / 2)) if fx is None else fx
        return cls(
            width,
            height,
            focal,
            focal if fy is None else fy,
            (width - 1) / 2 if cx is None else cx,
            (height - 1) / 2 if cy is None else cy,
        )

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def match_descriptors(first, second, ratio=0.8):
    """Symmetric ratio test and mutual nearest neighbors; indices are frame-local."""
    if len(first) < 2 or len(second) < 2:
        return np.empty((0, 2), dtype=np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(np.asarray(first, np.float32), np.asarray(second, np.float32), k=2)
    backward = matcher.knnMatch(np.asarray(second, np.float32), np.asarray(first, np.float32), k=2)
    reverse = {a.queryIdx: a.trainIdx for a, b in backward if a.distance < ratio * b.distance}
    return np.array(
        [
            (a.queryIdx, a.trainIdx)
            for a, b in forward
            if a.distance < ratio * b.distance and reverse.get(a.trainIdx) == a.queryIdx
        ],
        dtype=np.int32,
    ).reshape(-1, 2)


def camera_center(pose):
    return -pose[:3, :3].T @ pose[:3, 3]


def project(points, pose, camera):
    local = points @ pose[:3, :3].T + pose[:3, 3]
    # Keep the sign of invalid depths so callers can explicitly reject them.
    depth = local[:, 2]
    safe = np.where(np.abs(depth) < 1e-10, 1e-10, depth)
    xy = local[:, :2] / safe[:, None]
    return xy * [camera.fx, camera.fy] + [camera.cx, camera.cy], depth


def triangulate(
    first_xy, second_xy, first_pose, second_pose, camera, *, max_error=3.0, min_angle_deg=1.0
):
    if not len(first_xy):
        return np.empty((0, 3)), np.zeros(0, bool), np.zeros(0)
    homogeneous = cv2.triangulatePoints(
        camera.K @ first_pose[:3],
        camera.K @ second_pose[:3],
        np.asarray(first_xy, np.float64).T,
        np.asarray(second_xy, np.float64).T,
    )
    finite = np.abs(homogeneous[3]) > 1e-10
    points = (homogeneous[:3] / np.where(finite, homogeneous[3], 1)).T
    xy0, z0 = project(points, first_pose, camera)
    xy1, z1 = project(points, second_pose, camera)
    ray0, ray1 = points - camera_center(first_pose), points - camera_center(second_pose)
    denominator = np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1)
    cosine = np.sum(ray0 * ray1, axis=1) / np.maximum(denominator, 1e-12)
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    good = (
        finite
        & np.isfinite(points).all(axis=1)
        & (z0 > 0)
        & (z1 > 0)
        & (np.linalg.norm(xy0 - first_xy, axis=1) <= max_error)
        & (np.linalg.norm(xy1 - second_xy, axis=1) <= max_error)
        & (angles >= min_angle_deg)
        & (angles < 90)
    )
    return points, good, angles


def initialize_pair(
    first_xy,
    second_xy,
    camera,
    *,
    threshold=1.5,
    min_inliers=30,
    min_angle_deg=1.0,
    max_error=3.0,
    diagnostics=None,
):
    """Five-point RANSAC, cheirality, parallax and homography degeneracy checks."""
    if len(first_xy) < max(8, min_inliers):
        return None
    essential, mask = cv2.findEssentialMat(
        first_xy,
        second_xy,
        camera.K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold,
        maxIters=3000,
    )
    if essential is None or mask is None:
        return None
    _, homography_mask = cv2.findHomography(
        first_xy, second_xy, cv2.RANSAC, threshold, maxIters=2000, confidence=0.999
    )
    # Planar scenes and pure rotations do not reliably initialize general 3D.
    homography_fraction = 0 if homography_mask is None else float(homography_mask.mean())
    if homography_fraction > 0.85:
        return None
    best = None
    for matrix in essential.reshape(-1, 3, 3):
        _, rotation, translation, recovered = cv2.recoverPose(
            matrix, first_xy, second_xy, camera.K, mask=mask.copy()
        )
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation.ravel()
        points, valid, angles = triangulate(
            first_xy,
            second_xy,
            np.eye(4),
            pose,
            camera,
            max_error=max_error,
            min_angle_deg=min_angle_deg,
        )
        valid &= recovered.ravel() != 0
        count = int(valid.sum())
        if count < min_inliers or count < 0.25 * len(first_xy):
            continue
        # Enough points can pass a weak per-point gate while all depths remain unstable.
        # Seed the map with a wider baseline than is needed to extend an existing map.
        if np.median(angles[valid]) < max(2.0, min_angle_deg):
            continue
        if best is None or count > best[2].sum():
            if diagnostics is not None:
                diagnostics.update(
                    xy_first=np.array(first_xy, copy=True),
                    xy_second=np.array(second_xy, copy=True),
                    ransac_inliers=(mask.ravel() != 0).copy(),
                    triangulated=valid.copy(),
                    homography_inliers=(homography_mask.ravel() != 0)
                    if homography_mask is not None
                    else np.zeros(len(first_xy), bool),
                    essential=matrix.copy(),
                    parallax_deg=angles.copy(),
                )
            best = (
                pose,
                points,
                valid,
                {
                    "essential_inliers": int(np.count_nonzero(mask)),
                    "triangulated": count,
                    "homography_fraction": homography_fraction,
                    "median_parallax_deg": float(np.median(angles[valid])),
                },
            )
    return best


def estimate_pose(points, xy, camera, *, threshold=3.0, min_inliers=20, diagnostics=None):
    """PnP RANSAC followed by inlier-only pose refinement and revalidation."""
    if len(points) < max(6, min_inliers):
        return None
    points, xy = np.asarray(points, np.float64), np.asarray(xy, np.float64)
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        points,
        xy,
        camera.K,
        None,
        iterationsCount=2000,
        reprojectionError=threshold,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if diagnostics is not None:
        raw = np.zeros(len(points), bool)
        if inliers is not None:
            raw[inliers.ravel()] = True
        diagnostics.update(
            ransac_ran=True,
            ransac_inliers=raw,
            refined_inliers=np.zeros(len(points), bool),
            accepted=False,
        )
        if ok:
            raw_pose = np.eye(4)
            raw_pose[:3, :3] = cv2.Rodrigues(rotation)[0]
            raw_pose[:3, 3] = translation.ravel()
            raw_xy, _ = project(points, raw_pose, camera)
            diagnostics.update(
                raw_pose=raw_pose,
                raw_projected_xy=raw_xy,
                raw_errors=np.linalg.norm(raw_xy - xy, axis=1),
            )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    indices = inliers.ravel()
    rotation, translation = cv2.solvePnPRefineLM(
        points[indices], xy[indices], camera.K, None, rotation, translation
    )
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rotation)[0], translation.ravel()
    projected, depth = project(points, pose, camera)
    errors = np.linalg.norm(projected - xy, axis=1)
    valid = (errors <= threshold) & (depth > 0) & np.isfinite(errors)
    if diagnostics is not None:
        diagnostics.update(
            refined_pose=pose.copy(),
            projected_xy=projected.copy(),
            errors=errors.copy(),
            refined_inliers=valid.copy(),
        )
    if valid.sum() < min_inliers or valid.mean() < 0.25:
        return None
    # Avoid accepting support confined to a single tiny image region.
    span = np.ptp(xy[valid], axis=0)
    if span[0] < camera.width * 0.1 or span[1] < camera.height * 0.1:
        return None
    if diagnostics is not None:
        diagnostics["accepted"] = True
    return pose, valid, errors
