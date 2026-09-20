import numpy as np

from slam_lab.config import ExtractionConfig
from slam_lab.correspondence import match_all
from slam_lab.matching import PairMatches
from slam_lab.pipeline import process_video
from slam_lab.workbench_server import WorkbenchData


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
    assert pair["pair_runtime"]["device"] == "cuda"
    assert data.image(0).startswith(b"\xff\xd8")
