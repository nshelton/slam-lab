"""Read TUM RGB image sequences at their original timestamps."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PIL import Image

from slam_lab.config import fingerprint, hash_file
from slam_lab.video import DecodedFrame, VideoReader, prepare_image


@dataclass(frozen=True)
class RGBEntry:
    timestamp_ns: int
    path: Path


def read_rgb_manifest(source: Path) -> list[RGBEntry]:
    source = source.resolve()
    entries = []
    for number, line in enumerate((source / "rgb.txt").read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Invalid RGB manifest row {number}: {source / 'rgb.txt'}")
        try:
            nanos = Decimal(fields[0]) * 1_000_000_000
            if not nanos.is_finite() or nanos != nanos.to_integral_value():
                raise ValueError("timestamp must resolve to whole nanoseconds")
            timestamp_ns = int(nanos)
            if not 0 <= timestamp_ns < 2**63:
                raise ValueError("timestamp is outside the supported range")
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Invalid RGB timestamp on row {number}: {fields[0]}") from error
        if entries and timestamp_ns <= entries[-1].timestamp_ns:
            raise ValueError(f"RGB timestamps must be strictly increasing (row {number})")
        path = (source / fields[1]).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"RGB image must be inside the dataset directory: {fields[1]}")
        if not path.is_file():
            raise FileNotFoundError(f"RGB image missing: {path}")
        entries.append(RGBEntry(timestamp_ns, path))
    if not entries:
        raise ValueError(f"No RGB frames listed in {source / 'rgb.txt'}")
    return entries


def source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return [source / "rgb.txt", *(entry.path for entry in read_rgb_manifest(source))]


def source_state(source: Path) -> list[tuple[str, int, int]]:
    """Snapshot only the files used for extraction, excluding depth and ground truth."""
    state = []
    for path in source_files(source):
        stat = path.stat()
        relative = path.name if source.is_file() else str(path.relative_to(source))
        state.append((relative, stat.st_size, stat.st_mtime_ns))
    return state


def hash_source(source: Path) -> str:
    if source.is_file():
        return hash_file(source)  # Keep existing video cache identities unchanged.
    return fingerprint(
        {
            "format": "tum-rgb-v1",
            "files": [
                (str(path.relative_to(source)), hash_file(path)) for path in source_files(source)
            ],
        }
    )


class TUMReader(AbstractContextManager):
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.entries = read_rgb_manifest(self.path)
        self.total_frames = len(self.entries)
        return self

    def __exit__(self, *args):
        pass

    def metadata(self) -> dict:
        with Image.open(self.entries[0].path) as image:
            width, height = image.size
        return {
            "format": "tum-rgb",
            "width": width,
            "height": height,
            "frame_count_hint": self.total_frames,
            "time_base": "1/1000000000",
            "start_pts": self.entries[0].timestamp_ns,
            "manifest": "rgb.txt",
            "has_depth": (self.path / "depth.txt").is_file(),
            "has_ground_truth": (self.path / "groundtruth.txt").is_file(),
        }

    def frames(self, *, stride: int, max_side: int):
        origin = self.entries[0].timestamp_ns
        for index in range(0, len(self.entries), stride):
            entry = self.entries[index]
            with Image.open(entry.path) as image:
                yield DecodedFrame(
                    index=index,
                    timestamp_ns=entry.timestamp_ns - origin,
                    pts=entry.timestamp_ns,
                    image=prepare_image(image, max_side),
                    original_width=image.width,
                    original_height=image.height,
                )


def open_source(source: Path):
    return TUMReader(source) if source.is_dir() else VideoReader(source)
