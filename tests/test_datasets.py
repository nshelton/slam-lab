import numpy as np
import pytest
from PIL import Image

from slam_lab.cache import FrameCache
from slam_lab.config import ExtractionConfig
from slam_lab.datasets import TUMReader, read_rgb_manifest
from slam_lab.pipeline import process_video
from slam_lab.video import discover_sources


@pytest.fixture
def tum(tmp_path):
    root = tmp_path / "sequence"
    (root / "rgb").mkdir(parents=True)
    rows = ["# timestamp filename"]
    for index, timestamp in enumerate(
        ["1305031102.000000001", "1305031102.033333337", "1305031102.100000009"]
    ):
        Image.fromarray(np.full((48, 64, 3), index * 40, dtype=np.uint8)).save(
            root / "rgb" / f"{index}.png"
        )
        rows.append(f"{timestamp} rgb/{index}.png")
    (root / "rgb.txt").write_text("\n".join(rows))
    (root / "depth.txt").write_text("# reserved for later RGB-D processing")
    (root / "groundtruth.txt").write_text("# reserved for trajectory evaluation")
    return root


def test_tum_preserves_pixels_and_precise_epoch_timestamps(tum):
    with TUMReader(tum) as reader:
        assert reader.total_frames == 3
        assert reader.metadata()["has_ground_truth"]
        frames = list(reader.frames(stride=1, max_side=1024))
    assert [frame.timestamp_ns for frame in frames] == [0, 33_333_336, 100_000_008]
    assert frames[0].pts == 1_305_031_102_000_000_001
    np.testing.assert_array_equal(frames[1].image, np.full((48, 64, 3), 40))
    assert discover_sources(tum) == [tum]
    assert discover_sources(tum.parent) == [tum]
    assert discover_sources(tum.parent, recursive=True) == [tum]


def test_tum_cache_resume_and_image_content_invalidation(tum, tmp_path, extractor):
    config = ExtractionConfig()
    first = process_video(tum, tmp_path / "cache", config, extractor, max_frames=1, progress=False)
    assert not first.complete
    resumed = process_video(tum, tmp_path / "cache", config, extractor, progress=False)
    assert (resumed.new_frames, resumed.cached_frames, resumed.complete) == (2, 1, True)
    with FrameCache(resumed.path) as cache:
        assert [frame.pts for frame in cache.frames()] == [
            1_305_031_102_000_000_001,
            1_305_031_102_033_333_337,
            1_305_031_102_100_000_009,
        ]
    (tum / "depth.txt").write_text("# depth changes do not invalidate RGB features")
    hit = process_video(tum, tmp_path / "cache", config, extractor, progress=False)
    assert hit.path == resumed.path
    assert hit.new_frames == 0
    Image.new("RGB", (64, 48), "red").save(tum / "rgb" / "0.png")
    changed = process_video(tum, tmp_path / "cache", config, extractor, progress=False)
    assert changed.path != resumed.path
    assert changed.new_frames == 3


@pytest.mark.parametrize(
    "manifest",
    ["NaN rgb/0.png", "1 rgb/0.png\n1 rgb/1.png", "1 ../outside.png", "1 rgb/missing.png"],
)
def test_rejects_invalid_tum_inputs(tum, manifest):
    (tum / "rgb.txt").write_text(manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_rgb_manifest(tum)
