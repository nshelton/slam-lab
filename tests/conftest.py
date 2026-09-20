from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "variable-rate.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "bgr0"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, milliseconds in enumerate([1000, 1040, 1120, 1160]):
            image = np.full((96, 128, 3), index * 40, dtype=np.uint8)
            image[20:40, 30:60] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = milliseconds
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def extractor():
    class CountingExtractor:
        calls = 0
        provenance = {"device": "test"}

        def __call__(self, image):
            self.calls += 1
            # Include featureless frames to exercise array round-tripping and overlay clearing.
            count = 0 if self.calls % 2 == 0 else 2
            return {
                "keypoints": np.tile([[12.5, 20.0]], (count, 1)).astype(np.float32),
                "scores": np.full(count, 0.02, dtype=np.float32),
                "descriptors": np.full((count, 256), 1 / 16, dtype=np.float32),
                "extraction_ms": 1.5,
            }

    return CountingExtractor()
