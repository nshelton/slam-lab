"""Sequential decoding with presentation timestamps, including variable frame rate."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".mts"}


def discover_sources(path: Path, *, recursive: bool = False) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return [path]  # Explicit files are decoded regardless of their extension.
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if (path / "rgb.txt").is_file():
        return [path]
    candidates = path.rglob("*") if recursive else path.iterdir()
    videos = sorted(
        p
        for p in candidates
        if (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        or (p.is_dir() and (p / "rgb.txt").is_file())
    )
    if not videos:
        raise ValueError(f"No videos or TUM RGB sequences found in {path}")
    return videos


@dataclass
class DecodedFrame:
    index: int
    timestamp_ns: int
    pts: int
    image: np.ndarray
    original_width: int
    original_height: int


def prepare_image(image: Image.Image, max_side: int) -> np.ndarray:
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    if min(image.size) < 8:
        raise ValueError(f"Image is too small for SuperPoint: {image.size}")
    return np.asarray(image.convert("RGB"))


class VideoReader(AbstractContextManager):
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.container = av.open(str(self.path))
        if not self.container.streams.video:
            self.container.close()
            raise ValueError(f"No video stream in {self.path}")
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.total_frames = self.stream.frames or None
        return self

    def __exit__(self, *args):
        self.container.close()

    def metadata(self) -> dict:
        return {
            "stream_index": self.stream.index,
            "width": self.stream.width,
            "height": self.stream.height,
            "frame_count_hint": self.stream.frames or None,
            "average_fps": float(self.stream.average_rate) if self.stream.average_rate else None,
            "time_base": str(self.stream.time_base),
            "start_pts": self.stream.start_time,
            "orientation": "coded pixels (rotation metadata is not applied)",
        }

    def frames(self, *, stride: int, max_side: int):
        origin = None
        previous = None
        for index, frame in enumerate(self.container.decode(self.stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame {index} has no presentation timestamp in {self.path}")
            if frame.time_base != self.stream.time_base:
                raise ValueError("Video time base changed while decoding")
            time = frame.pts * frame.time_base
            if previous is not None and time < previous:
                raise ValueError(f"Non-monotonic video timestamps at frame {index}")
            previous = time
            if origin is None:
                origin = time
            if index % stride:
                continue
            image = frame.to_image()
            original_width, original_height = image.size
            yield DecodedFrame(
                index=index,
                timestamp_ns=round((time - origin) * Fraction(1_000_000_000)),
                pts=frame.pts,
                image=prepare_image(image, max_side),
                original_width=original_width,
                original_height=original_height,
            )
