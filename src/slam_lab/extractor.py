"""Lazy SuperPoint loading: viewing and complete cache hits do not initialize Torch."""

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from slam_lab.config import WEIGHTS_SHA256, ExtractionConfig, hash_file


class SuperPointExtractor:
    def __init__(self, config: ExtractionConfig, *, device: str, model_dir: Path):
        self.config = config
        self.requested_device = device
        self.model_dir = model_dir
        self.model = None

    def _load(self):
        try:
            import torch
            from lightglue import SuperPoint
        except ImportError as error:
            raise RuntimeError(
                "SuperPoint dependencies are missing. Install with "
                "uv pip install --torch-backend=auto -e '.[superpoint]'"
            ) from error

        device = self.requested_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Use --device cpu or auto.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable. Use --device cpu or auto.")
        self.device = device
        self.torch = torch
        if device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        weights = self.model_dir / "checkpoints" / "superpoint_v1.pth"
        if weights.exists() and hash_file(weights) != WEIGHTS_SHA256:
            raise ValueError(f"SuperPoint weights checksum mismatch: {weights}")
        previous_hub_dir = torch.hub.get_dir()
        try:
            # Keep weights alongside the workspace cache, not in a global user directory.
            torch.hub.set_dir(str(self.model_dir))
            self.model = (
                SuperPoint(
                    max_num_keypoints=self.config.max_keypoints,
                    detection_threshold=self.config.detection_threshold,
                )
                .eval()
                .to(device)
            )
        finally:
            torch.hub.set_dir(previous_hub_dir)
        if hash_file(weights) != WEIGHTS_SHA256:
            self.model = None
            raise ValueError(f"Downloaded SuperPoint weights checksum mismatch: {weights}")
        self.provenance = {"device": device, "weights_sha256": hash_file(weights)}

    def __call__(self, image: np.ndarray) -> dict:
        if self.model is None:
            self._load()
        torch = self.torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255).to(self.device)
        self._synchronize()
        start = perf_counter()
        with torch.inference_mode():
            # The pipeline already resized the frame; keypoints stay in preview coordinates.
            features = self.model.extract(tensor, resize=None)
        self._synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
        return {
            "keypoints": features["keypoints"][0].cpu().numpy(),
            "scores": features["keypoint_scores"][0].cpu().numpy(),
            "descriptors": features["descriptors"][0].cpu().numpy(),
            "extraction_ms": elapsed_ms,
        }

    def _synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()
