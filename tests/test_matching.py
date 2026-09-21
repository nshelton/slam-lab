from types import SimpleNamespace

import numpy as np
import pytest

from slam_lab.artifacts import validate_manifest
from slam_lab.config import ExtractionConfig
from slam_lab.correspondence import MatchStore, associate_tracks, match_all, match_status
from slam_lab.match_view import view_matches
from slam_lab.matching import CosineMatcher, DescriptorMatcher, PairMatches
from slam_lab.pipeline import process_video


class CountingMatcher:
    provenance = {"name": "test"}

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, first, second):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        count = min(len(first.keypoints), len(second.keypoints))
        return PairMatches(
            np.column_stack([np.arange(count), np.arange(count)]).astype(np.int32),
            np.full(count, 0.9, np.float32),
            np.zeros(count, np.float32),
        )


def test_nn_matches_permutation_and_rejects_ambiguous_ties():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(30, 256)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    order = rng.permutation(len(a))
    matcher = DescriptorMatcher()
    matches = matcher(SimpleNamespace(descriptors=a), SimpleNamespace(descriptors=a[order]))
    np.testing.assert_array_equal(matches.indices[:, 0], np.arange(len(a)))
    np.testing.assert_array_equal(order[matches.indices[:, 1]], np.arange(len(a)))
    matches.validate(30, 30)
    zeros = SimpleNamespace(descriptors=np.zeros((3, 256), np.float32))
    assert len(matcher(zeros, zeros).indices) == 0
    assert len(matcher(SimpleNamespace(descriptors=a[:0]), zeros).indices) == 0


def test_tracks_enforce_unique_frame_and_keep_singletons():
    offsets, ids, lengths, summary = associate_tracks(
        [2, 1, 1], np.array([[0, 2], [2, 3], [1, 3], [0, 3]]), np.array([0.9, 0.8, 0.7, 0.6])
    )
    assert offsets.tolist() == [0, 2, 3, 4]
    assert ids[0] == ids[2] == ids[3]
    assert ids[0] != ids[1]
    assert sorted(lengths.tolist()) == [1, 3]
    assert summary["conflicting_edges"] == summary["consistent_cycle_edges"] == 1
    _, ids, lengths, summary = associate_tracks([0, 0], np.empty((0, 2), int), np.empty(0))
    assert not len(ids) and not len(lengths)
    assert summary["longest_track"] == 0


def test_all_pairs_resume_identity_and_self_contained_view(video, tmp_path, extractor):
    cached = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    output = tmp_path / "matches"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        match_all(
            cached.path, output, matcher=CountingMatcher(fail_at=4), workers=1, progress=False
        )
    partial = match_status(output)
    assert 0 < partial["completed_pairs"] < 6
    assert partial["running"] is False
    resumed = CountingMatcher()
    summary = match_all(cached.path, output, matcher=resumed, workers=1, progress=False)
    assert resumed.calls == 6 - partial["completed_pairs"]
    assert summary["unique_pairs"] == 6
    assert summary["geometry_verified"] is False
    manifest = validate_manifest(output, artifact_type="matches")
    assert manifest["parent_artifact_id"] is None
    assert manifest["capabilities"] == [
        "matching_observation_namespace",
        "video_referenced_frames",
    ]
    with MatchStore(output / "matches.sqlite3") as store:
        assert list(store.db.execute("SELECT first,second FROM pairs ORDER BY first,second")) == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]
        forward, reverse = store.pair(0, 2), store.pair(2, 0)
        np.testing.assert_array_equal(forward.indices, reverse.indices[:, ::-1])
        assert len(store.pair(0, 1).indices) == 0
        with pytest.raises(ValueError, match="not been processed"):
            store.pair(0, 0)
    with np.load(output / "tracks.npz") as archive:
        counts = archive["pair_match_counts"]
        np.testing.assert_array_equal(counts, counts.T)
        assert counts[0, 2] == 2
        assert not np.diag(counts).any()
        assert sorted(archive["track_lengths"]) == [2, 2]
    hit = CountingMatcher(fail_at=1)
    match_all(cached.path, output, matcher=hit, workers=1, progress=False)
    assert hit.calls == 0
    with pytest.raises(ValueError, match="different selection"):
        match_all(cached.path, output, matcher=hit, max_frames=3, progress=False)
    video.unlink()
    cached.path.unlink()
    assert view_matches(output, output=tmp_path / "pair.rrd", pair=(0, 2)) == 2
    assert view_matches(output, output=tmp_path / "empty.rrd", pair=(0, 1)) == 2
    assert view_matches(output, output=tmp_path / "tracks.rrd") == 4
    assert (tmp_path / "pair.rrd").stat().st_size > 1000


def test_rejects_duplicate_and_out_of_bounds_matches():
    with pytest.raises(ValueError, match="one-to-one"):
        PairMatches(np.array([[0, 0], [0, 1]]), np.ones(2), np.zeros(2)).validate(2, 2)
    with pytest.raises(ValueError, match="one-to-one"):
        PairMatches(np.array([[0, 2]]), np.ones(1), np.zeros(1)).validate(2, 2)


def test_writer_lock_is_reported_and_prevents_second_job(tmp_path):
    import fcntl

    output = tmp_path / "matches"
    with output.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert match_status(output) == {"phase": "starting", "running": True}
        with pytest.raises(RuntimeError, match="already running"):
            match_all(tmp_path / "absent-cache", output)


def test_cosine_uses_direction_not_magnitude_and_rejects_zero_or_tied_vectors():
    a = np.array([[2, 0], [0, 3], [0, 0]], np.float32)
    b = np.array([[0, 7], [4, 0], [0, 0]], np.float32)
    matches = CosineMatcher()(SimpleNamespace(descriptors=a), SimpleNamespace(descriptors=b))
    np.testing.assert_array_equal(matches.indices, [[0, 1], [1, 0]])
    np.testing.assert_allclose(matches.scores, [1, 1])
    np.testing.assert_allclose(matches.distances, [2, 4])
    matches.validate(3, 3)
    tied = SimpleNamespace(descriptors=np.array([[1, 0], [1, 0]], np.float32))
    assert len(CosineMatcher()(tied, tied).indices) == 0
    zero = SimpleNamespace(descriptors=np.zeros((1, 2), np.float32))
    assert len(CosineMatcher(-1)(zero, zero).indices) == 0
    empty = SimpleNamespace(descriptors=np.empty((0, 2), np.float32))
    assert len(CosineMatcher()(empty, tied).indices) == 0


def test_cosine_matches_independent_scalar_reference():
    rng = np.random.default_rng(123)
    a, b = rng.normal(size=(8, 5)).astype(np.float32), rng.normal(size=(11, 5)).astype(np.float32)
    reference = np.array(
        [
            [
                float(np.dot(x.astype(float), y.astype(float)))
                / (np.linalg.norm(x.astype(float)) * np.linalg.norm(y.astype(float)))
                for y in b
            ]
            for x in a
        ]
    )
    expected = [
        (i, j)
        for i in range(len(a))
        for j in range(len(b))
        if reference[i, j] >= 0.5
        and reference[i, j] == max(reference[i])
        and reference[i, j] == max(reference[:, j])
    ]
    matches = CosineMatcher(0.5)(SimpleNamespace(descriptors=a), SimpleNamespace(descriptors=b))
    np.testing.assert_array_equal(matches.indices, expected)
    np.testing.assert_allclose(matches.scores, [reference[i, j] for i, j in expected], atol=1e-6)
    with pytest.raises(ValueError, match="cosine-threshold"):
        CosineMatcher(float("nan"))


def test_cosine_run_resume_preserves_parameters(video, tmp_path, extractor):
    from slam_lab.cli import parser
    from slam_lab.match_jobs import resume_arguments

    cached = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    output = tmp_path / "cosine"
    match_all(
        cached.path,
        output,
        matcher_name="cosine",
        cosine_threshold=0.85,
        max_frames=4,
        workers=2,
        progress=False,
    )
    resumed = parser().parse_args(resume_arguments(output))
    assert resumed.matcher == "cosine"
    assert resumed.cosine_threshold == 0.85
    assert resumed.max_frames == 4 and resumed.workers == 2
    assert resumed.cache == cached.path.resolve()
    assert match_status(output)["completed_pairs"] == 6
    with pytest.raises(ValueError, match="different selection or matcher"):
        match_all(
            cached.path,
            output,
            matcher_name="cosine",
            cosine_threshold=0.8,
            max_frames=4,
            progress=False,
        )


@pytest.mark.parametrize("failed", [False, True])
def test_continuation_runs_only_after_successful_matching_and_export(tmp_path, monkeypatch, failed):
    from slam_lab.cli import main

    calls = []

    def fake_match(*args, **kwargs):
        assert kwargs["export"] is True
        calls.append("match and export")
        if failed:
            raise RuntimeError("failed matching")

    monkeypatch.setattr("slam_lab.correspondence.match_all", fake_match)
    monkeypatch.setattr("slam_lab.match_jobs.resume_arguments", lambda path: [])
    monkeypatch.setattr("slam_lab.match_jobs.resume_saved_run", lambda path: calls.append("resume"))
    code = main(
        [
            "match-all",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "cosine"),
            "--matcher",
            "cosine",
            "--resume-after",
            str(tmp_path / "lightglue"),
        ]
    )
    assert code == (1 if failed else 0)
    assert calls == (["match and export"] if failed else ["match and export", "resume"])


def test_explicit_runtime_switch_preserves_legacy_pairs_and_rejects_model_changes(
    video, tmp_path, extractor
):
    import sqlite3

    cached = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    output = tmp_path / "mixed-runtime"
    cpu = CountingMatcher(fail_at=4)
    cpu.provenance = {
        "name": "lightglue-superpoint",
        "device": "cpu",
        "torch": "cpu-test",
        "weights_sha256": "same-weights",
        "filter_threshold": 0.1,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        match_all(cached.path, output, matcher=cpu, workers=1, progress=False)
    with sqlite3.connect(output / "matches.sqlite3") as db:
        original = dict(
            ((a, b), blob) for a, b, blob in db.execute("SELECT first,second,matches FROM pairs")
        )
        # Exercise migration of an actual old six-column pair table.
        db.execute("ALTER TABLE pairs DROP COLUMN runtime_id")
        db.execute("DROP TABLE match_runtimes")
    assert original
    gpu = CountingMatcher()
    gpu.provenance = {**cpu.provenance, "device": "cuda", "torch": "cuda-test"}
    with pytest.raises(ValueError, match="allow-runtime-change"):
        match_all(cached.path, output, matcher=gpu, workers=1, progress=False)
    assert gpu.calls == 0
    gpu.provenance["weights_sha256"] = "different-weights"
    with pytest.raises(ValueError, match="different selection or matcher"):
        match_all(
            cached.path, output, matcher=gpu, workers=1, progress=False, allow_runtime_change=True
        )
    gpu.provenance["weights_sha256"] = "same-weights"
    summary = match_all(
        cached.path, output, matcher=gpu, workers=1, progress=False, allow_runtime_change=True
    )
    assert gpu.calls == 6 - len(original)
    assert summary["matcher"]["device"] == "cuda"
    counts = {item["matcher"]["device"]: item["pairs"] for item in summary["runtimes"]}
    assert counts == {"cpu": len(original), "cuda": 6 - len(original)}
    with MatchStore(output / "matches.sqlite3") as store:
        assert store.get("identity")["matcher"] == cpu.provenance
        for a, b, blob in store.db.execute("SELECT first,second,matches FROM pairs"):
            if (a, b) in original:
                assert blob == original[(a, b)]
    assert match_status(output)["matcher"] == gpu.provenance
