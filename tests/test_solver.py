import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from slam_lab.artifacts import validate_manifest
from slam_lab.cache import CachedFrame
from slam_lab.correspondence import MatchStore, finalize_matches, pack_matches
from slam_lab.geometry import Camera, camera_center, project
from slam_lab.geometry_view import view_geometry
from slam_lab.matching import PairMatches
from slam_lab.reader import ArtifactReader
from slam_lab.reconstruction import ReconstructionConfig
from slam_lab.reconstruction_io import view_reconstruction
from slam_lab.solve_jobs import solve_status
from slam_lab.solver import GraphMapper, ReconstructionProblem, solve_geometry
from slam_lab.verification import (
    GeometryStore,
    VerificationConfig,
    geometry_status,
    verify_matches,
    verify_pair,
)


@pytest.fixture
def scene():
    rng = np.random.default_rng(123)
    camera = Camera.from_size(640, 480, fx=500)
    points = rng.uniform([-2, -1.4, 5], [2, 1.4, 10], (250, 3))
    poses, xy = [], []
    for i in range(8):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(np.array([0.0, i * 0.008, 0.0]))[0]
        pose[:3, 3] = -pose[:3, :3] @ np.array([i * 0.2, 0.03 * np.sin(i), 0])
        poses.append(pose)
        xy.append(project(points, pose, camera)[0])
    return camera, points, poses, xy


@pytest.fixture
def match_run(scene, tmp_path):
    camera, _, _, xy = scene
    rng = np.random.default_rng(99)
    frames = []
    for i, pixels in enumerate(xy):
        noisy = pixels + rng.normal(0, 0.1, pixels.shape)
        if i == 0:
            noisy = noisy[:0]
        frames.append(
            CachedFrame(
                i * 6,
                i * 100_000_000,
                None,
                640,
                480,
                640,
                480,
                noisy.astype(np.float32),
                np.ones(len(noisy)),
                np.empty((len(noisy), 0)),
                0,
            )
        )
    run = tmp_path / "imported-matches"
    with MatchStore(run / "matches.sqlite3", create=True) as store:
        store.initialize(frames, {"matcher": {"name": "external-correspondences"}}, "synthetic")
        for first in range(len(frames)):
            for second in range(first + 1, len(frames)):
                if first == 0:
                    pair = PairMatches.empty()
                else:
                    indices = np.column_stack([np.arange(250), np.arange(250)]).astype(np.int32)
                    # Wrong but one-to-one associations, without descriptors of any kind.
                    indices[:40, 1] = np.roll(indices[:40, 1], 5)
                    pair = PairMatches(indices, np.ones(250, np.float32), np.zeros(250, np.float32))
                store.put_pair(first, second, pair, 0)
        finalize_matches(store, run)
    return run, camera


def test_pair_geometry_outliers_and_pose_convention(scene):
    camera, _, poses, xy = scene
    second = xy[7].copy()
    second[:50] = np.random.default_rng(1).uniform([0, 0], [640, 480], (50, 2))
    data, summary = verify_pair(xy[1], second, camera)
    assert summary["seed_eligible"]
    assert data["verified"][50:].sum() > 190
    assert data["verified"][:50].sum() < 3
    expected = poses[7] @ np.linalg.inv(poses[1])
    np.testing.assert_allclose(data["pose_second_from_first"][:3, :3], expected[:3, :3], atol=0.01)
    np.testing.assert_allclose(
        data["pose_second_from_first"][:3, 3],
        expected[:3, 3] / np.linalg.norm(expected[:3, 3]),
        atol=0.02,
    )
    assert np.median(data["E_error_px"][50:]) < 0.01


@pytest.mark.parametrize("mode", ["rotation", "planar", "tiny_baseline", "empty"])
def test_degenerate_pairs_are_not_seeds(scene, mode):
    camera, points, poses, _ = scene
    points, pose = points.copy(), poses[5].copy()
    if mode == "rotation":
        pose[:3, 3] = 0
    elif mode == "planar":
        points[:, 2] = 7
    elif mode == "tiny_baseline":
        pose[:3, 3] *= 1e-5
    else:
        points = points[:0]
    a, b = project(points, np.eye(4), camera)[0], project(points, pose, camera)[0]
    data, summary = verify_pair(a, b, camera)
    assert not summary["seed_eligible"]
    assert len(data["E_inliers"]) == len(points)


def test_raw_essential_mask_survives_recover_pose_mutation(scene, monkeypatch):
    camera, _, _, xy = scene
    original = cv2.findEssentialMat
    captured = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["mask"] = result[1].ravel().astype(bool).copy()
        return result

    monkeypatch.setattr(cv2, "findEssentialMat", capture)
    data, _ = verify_pair(xy[1], xy[7], camera)
    np.testing.assert_array_equal(data["E_inliers"], captured["mask"])
    assert not np.shares_memory(data["E_inliers"], data["cheirality"])


def test_resumable_verification_tracks_and_immutable_input(match_run, tmp_path):
    run, camera = match_run
    before = hashlib.sha256((run / "matches.sqlite3").read_bytes()).hexdigest()
    output = tmp_path / "geometry"
    summary = verify_matches(run, output, camera_options={"fx": camera.fx}, progress=False)
    assert summary["pairs"] == 28 and summary["seed_candidates"] > 0
    assert summary["verified_edges"] > 3000
    match_manifest = validate_manifest(run, artifact_type="matches")
    geometry_manifest = validate_manifest(output, artifact_type="geometry")
    assert geometry_manifest["parent_artifact_id"] == match_manifest["artifact_id"]
    match_reader = ArtifactReader(run)
    first_page = match_reader.list_frames(limit=3)
    second_page = match_reader.list_frames(after=first_page["next_cursor"], limit=10)
    assert [item["row"] for item in first_page["items"] + second_page["items"]] == list(range(8))
    assert match_reader.status()["state"] == "complete"
    assert match_reader.pair(1, 7)["indices"].shape == (250, 2)
    with pytest.raises(ValueError, match="Invalid artifact cursor"):
        match_reader.list_frames(after="not-a-cursor")
    with pytest.raises(ValueError, match="canonical order"):
        match_reader.pair(7, 1)
    geometry_reader = ArtifactReader(output)
    pairs = geometry_reader.list_pairs(limit=5)
    assert len(pairs["items"]) == 5 and pairs["next_cursor"]
    assert geometry_reader.pair(1, 7)["summary"]["seed_eligible"]
    with pytest.raises(ValueError, match="different artifact"):
        geometry_reader.list_pairs(after=match_reader.list_pairs(limit=1)["next_cursor"])
    status = geometry_status(output)
    assert status["phase"] == "complete" and not status["running"]
    assert status["available_pairs"] == status["completed_pairs"] == 28
    with GeometryStore(output / "geometry.sqlite3") as store:
        raw_before = store.db.execute(
            "SELECT arrays FROM pairs WHERE first=1 AND second=7"
        ).fetchone()[0]
    again = verify_matches(run, output, camera_options={"fx": camera.fx}, progress=False)
    assert again == summary
    with GeometryStore(output / "geometry.sqlite3") as store:
        assert (
            raw_before
            == store.db.execute("SELECT arrays FROM pairs WHERE first=1 AND second=7").fetchone()[0]
        )
    assert before == hashlib.sha256((run / "matches.sqlite3").read_bytes()).hexdigest()
    problem = ReconstructionProblem.load(output)
    for ids in problem.observation_tracks:
        assert len(ids) == len(np.unique(ids))
    with pytest.raises(ValueError, match="settings changed"):
        verify_matches(run, output, camera_options={"fx": camera.fx + 1}, progress=False)
    manifest_path = run / "manifest.json"
    changed_manifest = json.loads(manifest_path.read_text())
    changed_manifest["artifact_id"] = "matches-different-parent"
    manifest_path.write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="parent does not match"):
        verify_matches(run, output, camera_options={"fx": camera.fx}, progress=False)
    manifest_path.write_text(json.dumps(match_manifest))
    with MatchStore(run / "matches.sqlite3", create=True) as store:
        pair = store.pair(1, 7)
        pair.scores *= 0.5
        with store.db:
            store.db.execute(
                "UPDATE pairs SET matches=? WHERE first=1 AND second=7", (pack_matches(pair),)
            )
    with pytest.raises(ValueError, match="payload does not match manifest"):
        verify_matches(run, output, camera_options={"fx": camera.fx}, progress=False)


def test_graph_solver_ground_truth_and_self_contained_viewers(match_run, scene, tmp_path):
    run, camera = match_run
    _, _, truth, _ = scene
    geometry = tmp_path / "geometry"
    verify_matches(run, geometry, camera_options={"fx": camera.fx}, progress=False)
    problem = ReconstructionProblem.load(geometry)
    assert all(frame.descriptors.shape[1] == 0 for frame in problem.frames)
    mapper = GraphMapper(problem, ReconstructionConfig(bundle_evaluations=15))
    mapper.run(progress=False)
    assert len(mapper.registered) == 7
    points, obs, _, errors, report = mapper.finalize()
    assert len(points) > 190 and report["accepted"]
    assert errors.max() <= 3
    assert np.bincount(obs[:, 1]).min() >= 3
    first, second = mapper.anchors
    baseline = np.linalg.norm(camera_center(truth[second]) - camera_center(truth[first]))
    for row in mapper.registered:
        expected = (
            truth[first][:3, :3]
            @ (camera_center(truth[row]) - camera_center(truth[first]))
            / baseline
        )
        np.testing.assert_allclose(camera_center(mapper.frames[row].pose), expected, atol=0.06)
    assert mapper.initialization["third_view"] is not None
    assert mapper.frames[0].pose is None
    result = solve_geometry(
        geometry,
        tmp_path / "solution",
        config=ReconstructionConfig(bundle_evaluations=0),
        progress=False,
        invocation=["/test/python", "-m", "slam_lab", "solve", "geometry path"],
    )
    metadata = json.loads((result / "metadata.json").read_text())
    assert metadata["registered_frames"] == 7
    manifest = validate_manifest(result)
    assert manifest["artifact_id"] == metadata["artifact_id"]
    assert manifest["artifact_type"] == "reconstruction"
    geometry_manifest = validate_manifest(geometry, artifact_type="geometry")
    assert manifest["parent_artifact_id"] == geometry_manifest["artifact_id"]
    assert metadata["matching_artifact_id"] == geometry_manifest["parent_artifact_id"]
    assert set(manifest["capabilities"]) == {
        "matching_frame_rows",
        "source_track_lineage",
        "stable_point_ids",
    }
    with np.load(result / "reconstruction.npz", allow_pickle=False) as reconstruction:
        np.testing.assert_array_equal(reconstruction["frame_rows"], problem.frame_rows)
        point_ids = reconstruction["point_ids"]
        source_track_ids = reconstruction["source_track_ids"]
        observations = reconstruction["observations"]
        assert len(point_ids) == len(np.unique(point_ids)) == len(reconstruction["points"])
        assert len(source_track_ids) == len(reconstruction["points"])
        for frame, point, feature in observations:
            assert problem.observation_tracks[frame][feature] == source_track_ids[point]
    status = solve_status(result)
    assert status["state"] == status["phase"] == "complete"
    assert status["output_artifact_id"] == manifest["artifact_id"]
    assert status["registered_frames"] == 7
    assert status["points"] == metadata["points"]
    assert not status["worker_alive"]
    job = json.loads((Path(status["job_directory"]) / "job.json").read_text())
    assert job["input"]["artifact_id"] == geometry_manifest["artifact_id"]
    assert job["output"]["artifact_id"] == manifest["artifact_id"]
    assert job["effective_options"]["bundle_evaluations"] == 0
    assert isinstance(job["arguments"], list) and job["exit_code"] == 0
    assert job["arguments"][-1] == "geometry path"
    assert job["finished_at"] is not None
    reader = ArtifactReader(result)
    assert reader.manifest()["artifact_id"] == manifest["artifact_id"]
    assert reader.status()["state"] == "complete"
    assert "source_track_ids" in reader.load_reconstruction()
    assert len(reader.list_frames(limit=3)["items"]) == 3
    with pytest.raises(ValueError, match="Pair reading"):
        reader.list_pairs()
    assert view_reconstruction(result, output=tmp_path / "solution.rrd") == 8
    assert view_geometry(geometry, pair=[1, 7], output=tmp_path / "geometry.rrd") == [1, 7]
    assert (tmp_path / "geometry.rrd").stat().st_size > 1000


def test_failed_solve_has_structured_terminal_status(tmp_path):
    output = tmp_path / "failed-solution"
    with pytest.raises(ValueError, match="Missing artifact manifest"):
        solve_geometry(tmp_path / "missing-geometry", output, progress=False)
    status = solve_status(output)
    assert status["state"] == "failed"
    assert status["revision"] >= 2
    assert status["last_error"]["code"]
    job = json.loads((Path(status["job_directory"]) / "job.json").read_text())
    assert job["exit_code"] == 1 and job["finished_at"] is not None
    assert not output.exists()


def test_subsample_keeps_original_match_rows(match_run, tmp_path):
    run, camera = match_run
    output = tmp_path / "subsample"
    verify_matches(
        run, output, frame_step=2, max_frames=3, camera_options={"fx": camera.fx}, progress=False
    )
    problem = ReconstructionProblem.load(output)
    np.testing.assert_array_equal(problem.frame_rows, [0, 2, 4])
    assert [frame.index for frame in problem.frames] == [0, 12, 24]


@pytest.mark.parametrize(
    "settings",
    [
        {"threshold_px": float("nan")},
        {"min_parallax_deg": 90},
        {"min_coverage": 0},
        {"min_inliers": 2},
    ],
)
def test_invalid_verification_config(settings):
    with pytest.raises(ValueError):
        replace(VerificationConfig(), **settings)


def test_degenerate_opencv_failure_is_recorded_without_aborting(scene, monkeypatch):
    camera, _, _, xy = scene

    def fail(*args, **kwargs):
        raise cv2.error("degenerate model")

    monkeypatch.setattr(cv2, "findFundamentalMat", fail)
    data, summary = verify_pair(xy[1], xy[7], camera)
    assert "F" in summary["estimator_errors"]
    assert not data["verified"].any()
    assert not summary["seed_eligible"]
