import subprocess
import sys
from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from slam_lab.cache import FrameCache
from slam_lab.cli import main
from slam_lab.config import ExtractionConfig
from slam_lab.pipeline import process_video
from slam_lab.video import VideoReader, discover_sources
from slam_lab.viewer import view_cache


def test_preserves_variable_timestamps_and_resize(video):
    with VideoReader(video) as reader:
        frames = list(reader.frames(stride=1, max_side=64))
    assert [frame.timestamp_ns for frame in frames] == [0, 40_000_000, 120_000_000, 160_000_000]
    assert frames[0].pts == 1000
    assert frames[0].image.shape == (48, 64, 3)
    assert (frames[0].original_width, frames[0].original_height) == (128, 96)


def test_partial_resume_and_complete_hit(video, tmp_path, extractor):
    config = ExtractionConfig(max_side=64)
    first = process_video(
        video, tmp_path / "cache", config, extractor, max_frames=2, progress=False
    )
    assert not first.complete
    assert first.new_frames == extractor.calls == 2
    resumed = process_video(video, tmp_path / "cache", config, extractor, progress=False)
    assert resumed.path == first.path
    assert resumed.complete
    assert (resumed.new_frames, resumed.cached_frames, extractor.calls) == (2, 2, 4)

    def forbidden(_):
        pytest.fail("Complete cache hit invoked inference")

    hit = process_video(video, tmp_path / "cache", config, forbidden, progress=False)
    assert (hit.new_frames, hit.cached_frames) == (0, 4)
    with FrameCache(hit.path) as cache:
        frames = list(cache.frames())
        assert frames[0].descriptors.shape == (2, 256)
        assert frames[1].descriptors.shape == (0, 256)
        assert frames[0].keypoints.dtype == np.float32
        with Image.open(BytesIO(frames[0].jpeg)) as image:
            assert image.size == (frames[0].width, frames[0].height) == (64, 48)


def test_interruption_retains_completed_frames(video, tmp_path, extractor):
    config = ExtractionConfig()

    def interrupted(image):
        if extractor.calls == 2:
            raise KeyboardInterrupt
        return extractor(image)

    with pytest.raises(KeyboardInterrupt):
        process_video(video, tmp_path / "cache", config, interrupted, progress=False)
    result = process_video(video, tmp_path / "cache", config, extractor, progress=False)
    assert (result.new_frames, result.cached_frames) == (2, 2)


def test_cache_invalidation_and_stride(video, tmp_path, extractor):
    config = ExtractionConfig(stride=2)
    first = process_video(video, tmp_path / "cache", config, extractor, progress=False)
    changed = process_video(
        video,
        tmp_path / "cache",
        replace(config, detection_threshold=0.01),
        extractor,
        progress=False,
    )
    assert first.path != changed.path
    with FrameCache(first.path) as cache:
        frames = list(cache.frames())
    assert [frame.index for frame in frames] == [0, 2]
    assert [frame.timestamp_ns for frame in frames] == [0, 120_000_000]
    # Renaming identical content does not recompute it.
    renamed = video.with_name("renamed.mkv")
    video.rename(renamed)
    hit = process_video(renamed, tmp_path / "cache", config, extractor, progress=False)
    assert hit.path == first.path
    assert hit.new_frames == 0
    # Changing content under the same filename creates a new cache.
    with renamed.open("ab") as file:
        file.write(b"\0")
    edited = process_video(renamed, tmp_path / "cache", config, extractor, progress=False)
    assert edited.path != first.path


def test_export_is_self_contained_and_does_not_overwrite(video, tmp_path, extractor):
    result = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    video.unlink()
    recording = tmp_path / "inspection.rrd"
    assert view_cache(result.path, output=recording) == 4
    assert recording.stat().st_size > 1000
    assert view_cache(result.path, output=tmp_path / "filtered.rrd", min_score=1.0) == 4
    with pytest.raises(FileExistsError):
        view_cache(result.path, output=recording)


@pytest.mark.parametrize("first_frame_empty,min_score", [(True, 0.0), (False, 1.0)])
def test_export_empty_overlay_in_fresh_process(
    video, tmp_path, extractor, first_frame_empty, min_score
):
    # Rerun's custom-component type cache must not be primed by another test/export.
    extractor.calls = 1 if first_frame_empty else 0
    cached = process_video(video, tmp_path / "cache", ExtractionConfig(), extractor, progress=False)
    output = tmp_path / "empty-start.rrd"
    exported = subprocess.run(
        [
            sys.executable,
            "-m",
            "slam_lab",
            "view",
            str(cached.path),
            "--save",
            str(output),
            "--min-score",
            str(min_score),
        ],
        capture_output=True,
        text=True,
    )
    assert exported.returncode == 0, exported.stderr
    assert "Saved 4 frames" in exported.stdout
    assert output.stat().st_size > 1000


def test_discovery_and_folder_failure_isolation(video, tmp_path, extractor, monkeypatch, capsys):
    nested = tmp_path / "nested"
    nested.mkdir()
    moved = nested / "other.MKV"
    moved.write_bytes(video.read_bytes())
    assert discover_sources(tmp_path) == [video]
    assert set(discover_sources(tmp_path, recursive=True)) == {video, moved}
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    monkeypatch.setattr("slam_lab.extractor.SuperPointExtractor", lambda *a, **k: extractor)
    assert main(["process", str(tmp_path), "--cache-dir", str(tmp_path / "cache"), "--quiet"]) == 1
    output = capsys.readouterr()
    assert "Error processing" in output.err
    assert "4 new" in output.out


@pytest.mark.parametrize(
    "setting", [{"stride": 0}, {"max_side": 1}, {"detection_threshold": float("nan")}]
)
def test_rejects_invalid_extraction_settings(setting):
    with pytest.raises(ValueError):
        ExtractionConfig(**setting)
