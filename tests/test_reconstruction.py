import json
from dataclasses import replace
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from slam_lab.bundle import normalize_baseline
from slam_lab.cache import CachedFrame, FrameCache
from slam_lab.config import SCHEMA_VERSION
from slam_lab.geometry import (
    Camera,
    camera_center,
    estimate_pose,
    initialize_pair,
    match_descriptors,
    project,
    triangulate,
)
from slam_lab.ransac_view import load_diagnostics, pnp_record
from slam_lab.reconstruction import Mapper, ReconstructionConfig, reconstruct
from slam_lab.reconstruction_io import view_reconstruction


@pytest.fixture
def scene():
    rng = np.random.default_rng(19)
    camera = Camera.from_size(640, 480, fx=500)
    points = rng.uniform([-2, -1.4, 5], [2, 1.4, 10], (250, 3))
    descriptors = rng.normal(size=(len(points), 256))
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    jpeg = BytesIO()
    Image.new("RGB", (640, 480), (120, 170, 200)).save(jpeg, format="JPEG")
    frames, poses = [], []
    for frame_id in range(12):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, frame_id * 0.005, 0.0]))[0]
        pose[:3, 3] = -pose[:3, :3] @ np.array([frame_id * 0.12, 0.01 * np.sin(frame_id), 0])
        xy, _ = project(points, pose, camera)
        xy += rng.normal(0, 0.15, xy.shape)
        # Descriptors vary, order varies, and 15% of coordinates are mismatches.
        desc = descriptors + rng.normal(0, 0.004, descriptors.shape)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True)
        bad = rng.choice(len(points), 38, replace=False)
        xy[bad] = rng.uniform([0, 0], [639, 479], (len(bad), 2))
        order = rng.permutation(len(points))
        if frame_id in (0, 6):
            order = order[:0]
        frames.append(
            CachedFrame(
                frame_id * 6,
                frame_id * 100_000_000,
                frame_id * 6000,
                640,
                480,
                1280,
                960,
                jpeg.getvalue(),
                xy[order].astype(np.float32),
                np.ones(len(order), np.float32),
                desc[order].astype(np.float32),
                0.0,
            )
        )
        poses.append(pose)
    return camera, points, frames, poses


def test_matching_rejects_ambiguity_and_accepts_descriptor_changes():
    first = np.eye(8, dtype=np.float32)
    second = np.roll(first, 3, axis=0) + 0.01
    matches = match_descriptors(first, second)
    np.testing.assert_array_equal(matches[:, 1], (matches[:, 0] + 3) % 8)
    assert len(match_descriptors(first[:0], second)) == 0
    assert len(match_descriptors(np.ones((4, 8), np.float32), np.ones((5, 8), np.float32))) == 0


def test_ransac_pose_and_triangulation_reject_outliers(scene):
    camera, points, _, poses = scene
    first, _ = project(points, poses[1], camera)
    second, _ = project(points, poses[8], camera)
    second[:50] = np.random.default_rng(1).uniform([0, 0], [640, 480], (50, 2))
    cv2.setRNGSeed(7)
    relative, triangulated, valid, _ = initialize_pair(first, second, camera)
    assert valid[50:].sum() > 190
    assert valid[:50].sum() < 3
    assert np.isfinite(triangulated[valid]).all()
    result = estimate_pose(points, second, camera)
    assert result is not None
    pose, inliers, errors = result
    np.testing.assert_allclose(pose, poses[8], atol=1e-5)
    assert inliers[:50].sum() < 3
    assert np.median(errors[inliers]) < 1e-4
    _, good, _ = triangulate(first, first, poses[1], poses[1], camera)
    assert not good.any()
    assert np.linalg.norm(relative[:3, 3]) == pytest.approx(1)


def test_diagnostics_preserve_exact_opencv_mask_and_estimation_inputs(scene, monkeypatch):
    camera, points, _, poses = scene
    xy, _ = project(points, poses[8], camera)
    xy[:50] = np.random.default_rng(1).uniform([0, 0], [640, 480], (50, 2))
    captured = {}
    original = cv2.solvePnPRansac

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["indices"] = result[3].ravel().copy()
        return result

    monkeypatch.setattr(cv2, "solvePnPRansac", capture)
    diagnostics = {}
    result = estimate_pose(points, xy, camera, diagnostics=diagnostics)
    assert result is not None and diagnostics["ransac_ran"] and diagnostics["accepted"]
    np.testing.assert_array_equal(
        np.flatnonzero(diagnostics["ransac_inliers"]), captured["indices"]
    )
    np.testing.assert_array_equal(diagnostics["refined_inliers"], result[1])
    np.testing.assert_allclose(diagnostics["errors"], result[2])
    np.testing.assert_allclose(
        project(points, diagnostics["raw_pose"], camera)[0], diagnostics["raw_projected_xy"]
    )
    saved_raw_pose = diagnostics["raw_pose"].copy()
    result[0][:] = 0
    np.testing.assert_array_equal(diagnostics["raw_pose"], saved_raw_pose)


def test_failed_ransac_keeps_rejection_diagnostics(scene, monkeypatch):
    camera, points, _, poses = scene
    xy, _ = project(points, poses[8], camera)
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **kw: (False, None, None, None))
    diagnostics = {}
    assert estimate_pose(points, xy, camera, diagnostics=diagnostics) is None
    assert diagnostics["ransac_ran"] and not diagnostics["accepted"]
    assert len(diagnostics["ransac_inliers"]) == len(points)
    assert not diagnostics["ransac_inliers"].any()


def test_pure_rotation_does_not_create_a_map(scene):
    camera, points, _, _ = scene
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([0.03, 0.12, 0.0]))[0]
    first, _ = project(points, np.eye(4), camera)
    second, _ = project(points, pose, camera)
    assert initialize_pair(first, second, camera) is None


def test_planar_seed_is_rejected(scene):
    camera, points, _, poses = scene
    points = points.copy()
    points[:, 2] = 7
    first, _ = project(points, poses[1], camera)
    second, _ = project(points, poses[8], camera)
    assert initialize_pair(first, second, camera) is None


def test_impossible_track_length_does_not_silently_relax_filter(scene):
    camera, _, frames, _ = scene
    mapper = Mapper(frames, camera, ReconstructionConfig(min_track_length=100))
    mapper.run(progress=False)
    with pytest.raises(ValueError, match="No multi-view tracks"):
        mapper.finalize()


def test_all_empty_frames_cannot_produce_poses(scene):
    camera, _, frames, _ = scene
    mapper = Mapper([frames[0], frames[0]], camera, ReconstructionConfig())
    with pytest.raises(ValueError, match="No reliable two-view"):
        mapper.run(progress=False)


def test_incremental_mapping_scale_pose_direction_and_bundle_adjustment(scene):
    camera, _, frames, truth = scene
    mapper = Mapper(frames, camera, ReconstructionConfig(bundle_evaluations=20))
    mapper.run(progress=False)
    assert len(mapper.registered) == 10
    points, observations, _, errors, report = mapper.finalize()
    assert len(points) > 150
    assert report["accepted"]
    assert report["median_error_after_px"] < report["median_error_before_px"]
    assert errors.max() <= 3
    assert np.bincount(observations[:, 1]).min() >= 3
    anchor, second = mapper.anchors
    assert np.linalg.norm(
        camera_center(mapper.frames[second].pose) - camera_center(mapper.frames[anchor].pose)
    ) == pytest.approx(1, abs=1e-10)
    baseline = np.linalg.norm(camera_center(truth[second]) - camera_center(truth[anchor]))
    for index in mapper.registered:
        expected_center = (
            truth[anchor][:3, :3]
            @ (camera_center(truth[index]) - camera_center(truth[anchor]))
            / baseline
        )
        np.testing.assert_allclose(
            camera_center(mapper.frames[index].pose), expected_center, atol=0.15
        )
        expected_rotation = truth[index][:3, :3] @ truth[anchor][:3, :3].T
        np.testing.assert_allclose(mapper.frames[index].pose[:3, :3], expected_rotation, atol=0.025)
    assert mapper.frames[0].pose is None and mapper.frames[6].pose is None


def test_scale_normalization_preserves_projections_and_anchor(scene):
    camera, points, _, poses = scene
    original = {i: pose for i, pose in enumerate(poses)}
    normalized, world = normalize_baseline(original, points, (2, 7))
    assert np.linalg.norm(
        camera_center(normalized[7]) - camera_center(normalized[2])
    ) == pytest.approx(1)
    np.testing.assert_allclose(normalized[2], poses[2], atol=1e-12)
    for index, pose in normalized.items():
        np.testing.assert_allclose(
            project(world, pose, camera)[0], project(points, poses[index], camera)[0], atol=1e-10
        )


def test_artifacts_and_rerun_export_from_cached_features(scene, tmp_path):
    camera, _, frames, _ = scene
    cache_path = tmp_path / "features" / "cache.sqlite3"
    with FrameCache(cache_path, create=True) as cache:
        cache.set(schema_version=SCHEMA_VERSION, source_path="missing-video.mp4", identity={})
        for frame in frames:
            cache.put(frame)
    output = reconstruct(
        cache_path,
        tmp_path / "result",
        config=ReconstructionConfig(bundle_evaluations=0),
        camera_options={"fx": camera.fx},
        progress=False,
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["registered_frames"] == 10
    diagnostics = load_diagnostics(output)
    assert diagnostics is not None
    first, second = diagnostics["seed/frames"]
    matches = diagnostics["seed/matches"]
    np.testing.assert_array_equal(
        diagnostics["seed/xy_first"], frames[first].keypoints[matches[:, 0]]
    )
    np.testing.assert_array_equal(
        diagnostics["seed/xy_second"], frames[second].keypoints[matches[:, 1]]
    )
    for row in range(len(frames)):
        trace = pnp_record(diagnostics, row)
        if not trace:
            continue
        np.testing.assert_array_equal(trace["xy"], frames[row].keypoints[trace["keypoint_ids"]])
        if trace["ransac_ran"]:
            assert trace["ransac_inliers"].dtype == bool
            assert len(trace["final_landmark_ids"]) == len(trace["xy"])
            assert trace["ransac_inliers"].sum() <= len(trace["xy"])
    with np.load(output / "reconstruction.npz", allow_pickle=False) as result:
        assert result["T_world_camera"].shape == (12, 4, 4)
        assert np.isnan(result["T_world_camera"][0]).all()
        np.testing.assert_array_equal(result["frame_indices"], np.arange(12) * 6)
        assert result["track_lengths"].min() >= 3
        assert result["reprojection_errors"].max() <= 3
    recording = tmp_path / "reconstruction.rrd"
    assert view_reconstruction(output, output=recording) == 12
    assert recording.stat().st_size > 1000
    with pytest.raises(FileExistsError):
        reconstruct(cache_path, output, progress=False)
    with pytest.raises(FileExistsError):
        view_reconstruction(output, output=recording)
    assert (output / "points.ply").read_text().startswith("ply\n")


@pytest.mark.parametrize(
    "settings",
    [
        {"ratio": 1},
        {"min_parallax": float("nan")},
        {"min_track_length": 1},
        {"bundle_evaluations": -1},
    ],
)
def test_invalid_reconstruction_settings(settings):
    with pytest.raises(ValueError):
        ReconstructionConfig(**settings)


def test_fixed_camera_defaults_and_validation():
    camera = Camera.from_size(1024, 576)
    assert camera.cx == 511.5 and camera.cy == 287.5
    assert camera.fx == camera.fy == pytest.approx(886.8100135)
    with pytest.raises(ValueError):
        replace(camera, fx=-1)
    with pytest.raises(ValueError):
        Camera.from_size(100, 100, fy=100)
