import json

import numpy as np

from slam_lab.cache import CachedFrame, FrameCache
from slam_lab.online_tracker import (
    OnlineTrackerConfig,
    OnlineTrackStore,
    online_track_status,
    track_online,
)
from slam_lab.workbench_server import OnlineTrackData


def _descriptor(index):
    value = np.zeros(256, np.float32)
    value[index] = 1
    return value


def _cache(path):
    frames = [
        ([_descriptor(0), _descriptor(1)], [[10, 10], [30, 10]]),
        ([_descriptor(0), _descriptor(2)], [[11, 10], [50, 20]]),
        ([_descriptor(0), _descriptor(1)], [[12, 10], [32, 10]]),
    ]
    with FrameCache(path, create=True) as cache:
        cache.set(run_id="cache-test", source_path="/tmp/test.mp4", complete=True)
        for row, (descriptors, keypoints) in enumerate(frames):
            cache.put(
                CachedFrame(
                    index=row * 2,
                    timestamp_ns=row * 100_000_000,
                    pts=row,
                    width=64,
                    height=48,
                    original_width=64,
                    original_height=48,
                    keypoints=np.asarray(keypoints, np.float32),
                    scores=np.full(2, 0.5, np.float32),
                    descriptors=np.asarray(descriptors, np.float32),
                    extraction_ms=1,
                )
            )


def test_online_tracker_updates_spherical_landmarks_and_persists_frames(tmp_path):
    cache = tmp_path / "cache.sqlite3"
    output = tmp_path / "online"
    _cache(cache)
    track_online(
        cache,
        output,
        config=OnlineTrackerConfig(
            min_similarity=0.9,
            min_margin=0.01,
            max_inactive_frames=3,
        ),
        progress=False,
    )

    status = online_track_status(output)
    assert status["phase"] == "complete"
    assert status["processed_frames"] == 3
    assert status["observations"] == 6
    assert status["matched_observations"] == 3
    assert status["landmarks"] == 3
    assert json.loads((output / "manifest.json").read_text())["artifact_type"] == "online_tracks"

    with OnlineTrackStore(output) as store:
        observations = store.db.execute(
            "SELECT row_id,feature_id,landmark_id,state FROM observations "
            "ORDER BY row_id,feature_id"
        ).fetchall()
        assert [tuple(row) for row in observations] == [
            (0, 0, 0, "new"),
            (0, 1, 1, "new"),
            (1, 0, 0, "matched"),
            (1, 1, 2, "new"),
            (2, 0, 0, "matched"),
            (2, 1, 1, "matched"),
        ]
        landmark = store.db.execute(
            "SELECT observation_count,concentration,first_row,last_row FROM landmarks "
            "WHERE landmark_id=0"
        ).fetchone()
        assert tuple(landmark) == (3, 1.0, 0, 2)
        columns = {row[1] for row in store.db.execute("PRAGMA table_info(frames)")}
        assert "jpeg" not in columns
    assert OnlineTrackData(output).frame(1)["frame"]["video_url"].startswith(
        "/api/online-track/video"
    )


def test_online_tracker_enforces_one_observation_per_landmark_per_frame(tmp_path):
    cache = tmp_path / "cache.sqlite3"
    with FrameCache(cache, create=True) as store:
        store.set(run_id="cache-test", source_path="/tmp/test.mp4", complete=True)
        for row, count in enumerate((1, 2)):
            store.put(
                CachedFrame(
                    index=row,
                    timestamp_ns=row,
                    pts=row,
                    width=10,
                    height=10,
                    original_width=10,
                    original_height=10,
                    keypoints=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
                    scores=np.ones(count, np.float32),
                    descriptors=np.tile(_descriptor(0), (count, 1)),
                    extraction_ms=1,
                )
            )
    output = tmp_path / "online"
    track_online(
        cache,
        output,
        config=OnlineTrackerConfig(min_similarity=0.9, min_margin=0),
        progress=False,
    )
    with OnlineTrackStore(output) as store:
        second = store.db.execute(
            "SELECT landmark_id,state FROM observations WHERE row_id=1 ORDER BY feature_id"
        ).fetchall()
        assert [tuple(row) for row in second] == [(0, "matched"), (1, "new")]
