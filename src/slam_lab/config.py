"""Settings that affect cached data, independent of the viewer."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LIGHTGLUE_REVISION = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
WEIGHTS_RELEASE = "cvg/LightGlue/v0.1_arxiv/superpoint_v1.pth"
WEIGHTS_SHA256 = "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExtractionConfig:
    stride: int = 1
    max_side: int = 1024
    max_keypoints: int = 2048
    detection_threshold: float = 0.0005
    jpeg_quality: int = 90

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.max_side < 8:
            raise ValueError("max-side must be at least 8")
        if self.max_keypoints < 1:
            raise ValueError("max-keypoints must be at least 1")
        if not 0 <= self.detection_threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")

    def identity(self) -> dict:
        packages = ("av", "numpy", "Pillow", "torch", "torchvision", "kornia", "lightglue")
        versions = {}
        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor": "superpoint",
            "implementation_revision": LIGHTGLUE_REVISION,
            "weights_release": WEIGHTS_RELEASE,
            "weights_sha256": WEIGHTS_SHA256,
            "preprocessing": "rgb-pillow-lanczos-downscale-v1",
            "settings": asdict(self),
            "packages": versions,
        }


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def hash_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
