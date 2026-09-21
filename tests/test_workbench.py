import json

import numpy as np

from slam_lab.config import ExtractionConfig
from slam_lab.correspondence import match_all
from slam_lab.matching import PairMatches
from slam_lab.pipeline import process_video
from slam_lab.workbench_pipeline import PipelineSupervisor
from slam_lab.workbench_server import ReconstructionData, WorkbenchCatalog, WorkbenchData


class WorkbenchMatcher:
    provenance = {
        "name": "lightglue-superpoint",
        "device": "cuda",
        "torch": "test",
        "weights_sha256": "test",
        "filter_threshold": 0.1,
        "score": "LightGlue match confidence",
    }
    device = "cuda"

    def __call__(self, first, second):
        count = min(len(first.keypoints), len(second.keypoints))
        return PairMatches(
            np.column_stack([np.arange(count), np.arange(count)]).astype(np.int32),
            np.full(count, 0.75, np.float32),
            np.full(count, 0.25, np.float32),
        )


def test_workbench_reads_images_matches_and_runtime(video, tmp_path, extractor):
    cached = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    output = tmp_path / "matches"
    match_all(cached.path, output, matcher=WorkbenchMatcher(), workers=1, progress=False)

    data = WorkbenchData(output)
    overview = data.overview()
    assert overview["status"]["completed_pairs"] == 6
    assert overview["status"]["matcher"]["name"] == "lightglue-superpoint"
    assert overview["suggested_pair"] is not None

    pair = data.pair(0, 2)
    assert pair["first"]["frame_index"] == 0
    assert pair["second"]["frame_index"] == 2
    assert pair["matches"]["indices"] == [[0, 0], [1, 1]]
    assert pair["matches"]["first_xy"] == [[12.5, 20.0], [12.5, 20.0]]
    assert pair["features"]["first_xy"] == [[12.5, 20.0], [12.5, 20.0]]
    assert pair["pair_runtime"]["device"] == "cuda"
    assert pair["first"]["video_url"].startswith("/api/match/video")
    assert data.video_path() == video.resolve()
    matrix = data.matrix()
    assert matrix["size"] == 4
    assert len(matrix["counts"]) == 16
    assert matrix["counts"][0] == -2
    assert matrix["counts"][2] == matrix["counts"][8] == 2
    catalog = WorkbenchCatalog([output])
    assert catalog.default == "lightglue"
    assert catalog.list()["runs"][0]["status"]["matcher"]["name"] == "lightglue-superpoint"

    solution = tmp_path / "solution"
    solution.mkdir()
    np.savez_compressed(
        solution / "reconstruction.npz",
        points=np.array([[1, 2, 3]], float),
        colors=np.array([[10, 20, 30]], np.uint8),
        track_lengths=np.array([3]),
        T_world_camera=np.array([np.eye(4), np.full((4, 4), np.nan)]),
        registered=np.array([True, False]),
        frame_indices=np.array([4, 8]),
        timestamps_ns=np.array([100, 200]),
        K=np.eye(3),
    )
    (solution / "metadata.json").write_text(
        json.dumps(
            {
                "source_match_run": str(output.resolve()),
                "registered_frames": 1,
                "frames": 2,
                "points": 1,
                "observations": 3,
            }
        )
    )
    reconstruction = ReconstructionData(solution)
    assert reconstruction.scene()["frame_indices"] == [4]
    catalog = WorkbenchCatalog([output], [solution])
    listed = catalog.list()
    assert listed["runs"][0]["reconstructions"] == ["solution"]
    assert listed["reconstructions"][0]["points"] == 1
    assert catalog.reconstruction("solution", None).scene()["frame_indices"] == [4]

    geometry = tmp_path / "geometry"
    geometry.mkdir()
    (geometry / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "geometry",
                "artifact_id": "geometry-test",
                "parent_artifact_id": "matches-test",
                "state": "complete",
            }
        )
    )
    (geometry / "summary.json").write_text(
        json.dumps({"frames": 4, "pairs": 6, "verified_edges": 12, "tracks_three_or_more": 2})
    )
    discovered = WorkbenchCatalog([], artifact_roots=[tmp_path])
    discovered_list = discovered.list()
    assert len(discovered_list["runs"]) == 1
    assert discovered_list["reconstructions"][0]["name"] == "solution"
    assert discovered_list["geometry"][0]["frames"] == 4


def test_pipeline_supervisor_validates_and_persists_job(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    recordings = workspace / "recordings"
    source = workspace / "osaka.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    for executable in (
        workspace / ".venv" / "bin" / "python",
        workspace / ".venv-cuda" / "bin" / "slam-lab",
    ):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("")

    launched = []

    class Child:
        pid = 1234

    def fake_popen(arguments, **options):
        launched.append((arguments, options))
        return Child()

    monkeypatch.setattr("slam_lab.workbench_pipeline.subprocess.Popen", fake_popen)
    supervisor = PipelineSupervisor(workspace, recordings)
    status = supervisor.start(
        {
            "source": str(source),
            "name": "Osaka 300",
            "frames": 300,
            "stride": 6,
            "matcher": "lightglue",
            "device": "cuda",
            "workflow": "full",
        },
        [str(source.resolve())],
    )
    assert status["name"] == "Osaka-300"
    assert status["config"]["frames"] == 300
    assert status["worker_pid"] == 1234
    assert len(status["stages"]) == 5
    assert launched[0][0][1:3] == ["-m", "slam_lab.workbench_pipeline"]
    persisted = supervisor.list()[0]
    assert persisted["job_id"] == status["job_id"]
    assert persisted["state"] == "starting"

    status_path = recordings / "pipeline-jobs" / status["job_id"] / "status.json"
    previous = json.loads(status_path.read_text())
    previous["state"] = "complete"
    status_path.write_text(json.dumps(previous))
    tracking = supervisor.start(
        {
            "source": str(source),
            "name": "Long tracks",
            "frames": 50_000,
            "stride": 1,
            "matcher": "lightglue",
            "device": "cuda",
            "workflow": "tracks",
        },
        [str(source.resolve())],
    )
    assert tracking["config"]["frames"] == 50_000
    assert [stage["id"] for stage in tracking["stages"]] == ["features", "online_tracks"]
