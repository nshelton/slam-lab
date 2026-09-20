"""Pairwise appearance matchers; no intrinsics, poses, or geometric filtering."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slam_lab.config import LIGHTGLUE_REVISION, hash_file


@dataclass
class PairMatches:
    indices: np.ndarray
    scores: np.ndarray
    distances: np.ndarray

    def validate(self, first_count, second_count):
        count = len(self.indices)
        if self.indices.shape != (count, 2) or self.indices.dtype.kind not in "iu":
            raise ValueError("Match indices must have integer shape (N, 2)")
        if self.scores.shape != (count,) or self.distances.shape != (count,):
            raise ValueError("Match scores and distances must have shape (N,)")
        if not np.isfinite(self.scores).all() or not np.isfinite(self.distances).all():
            raise ValueError("Non-finite match score or distance")
        if (
            (self.indices < 0).any()
            or (self.indices >= [first_count, second_count]).any()
            or len(np.unique(self.indices[:, 0])) != count
            or len(np.unique(self.indices[:, 1])) != count
        ):
            raise ValueError("Matches must be in bounds and one-to-one")

    @classmethod
    def empty(cls):
        return cls(np.empty((0, 2), np.int32), np.empty(0, np.float32), np.empty(0, np.float32))


class DescriptorMatcher:
    def __init__(self, ratio=0.8):
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")
        self.ratio = ratio
        self.provenance = {
            "name": "mutual-nearest-neighbor",
            "ratio": ratio,
            "metric": "L2",
            "implementation": "numpy-full-distance-v1",
            "numpy": np.__version__,
            "score": "1 - max(bidirectional distance ratios)",
        }

    def __call__(self, first, second):
        a, b = first.descriptors, second.descriptors
        if len(a) < 2 or len(b) < 2:
            return PairMatches.empty()
        # Compute all descriptor distances once and query both directions from the same matrix.
        distance = a @ b.T
        distance *= -2
        distance += np.sum(a * a, axis=1)[:, None]
        distance += np.sum(b * b, axis=1)[None, :]
        np.maximum(distance, 0, out=distance)
        row, col = np.arange(len(a)), np.arange(len(b))
        nearest_b, nearest_a = distance.argmin(axis=1), distance.argmin(axis=0)
        best_a, best_b = distance[row, nearest_b].copy(), distance[nearest_a, col].copy()
        distance[row, nearest_b] = np.inf
        next_a = distance.min(axis=1)
        distance[row, nearest_b] = best_a
        distance[nearest_a, col] = np.inf
        next_b = distance.min(axis=0)
        valid = (
            (nearest_a[nearest_b] == row)
            & (best_a < self.ratio**2 * next_a)
            & (best_b[nearest_b] < self.ratio**2 * next_b[nearest_b])
        )
        rows, cols = row[valid], nearest_b[valid]
        ratios = np.sqrt(
            np.maximum(
                best_a[valid] / np.maximum(next_a[valid], 1e-20),
                best_b[cols] / np.maximum(next_b[cols], 1e-20),
            )
        )
        return PairMatches(
            np.column_stack([rows, cols]).astype(np.int32),
            (1 - ratios).astype(np.float32),
            np.sqrt(best_a[valid]),
        )


class CosineMatcher:
    """Exact all-descriptor cosine search with unique, mutual nearest neighbors."""

    def __init__(self, min_similarity=0.8):
        if not -1 <= min_similarity <= 1:
            raise ValueError("cosine-threshold must be between -1 and 1")
        self.min_similarity = min_similarity
        self.provenance = {
            "name": "cosine-mutual-nearest-neighbor",
            "implementation": "numpy-full-cosine-v1",
            "numpy": np.__version__,
            "min_similarity": min_similarity,
            "normalization": "L2 per descriptor; zero-norm descriptors excluded",
            "selection": "unique mutual best; exact ties rejected; no ratio test",
            "score": "cosine similarity in [-1, 1]",
            "distance": "L2 distance between original cached descriptors",
        }

    def __call__(self, first, second):
        a, b = first.descriptors, second.descriptors
        if not len(a) or not len(b):
            return PairMatches.empty()
        a_norm, b_norm = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        # Invalid descriptors must not compete with valid descriptors in either direction.
        valid_a, valid_b = a_norm > 0, b_norm > 0
        similarity = (a / np.where(valid_a, a_norm, 1)[:, None]) @ (
            b / np.where(valid_b, b_norm, 1)[:, None]
        ).T
        np.clip(similarity, -1, 1, out=similarity)
        similarity[~valid_a, :] = -np.inf
        similarity[:, ~valid_b] = -np.inf
        rows, cols = np.arange(len(a)), np.arange(len(b))
        nearest_b, nearest_a = similarity.argmax(axis=1), similarity.argmax(axis=0)
        best_a = similarity[rows, nearest_b].copy()
        best_b = similarity[nearest_a, cols].copy()
        similarity[rows, nearest_b] = -np.inf
        second_a = similarity.max(axis=1)
        similarity[rows, nearest_b] = best_a
        similarity[nearest_a, cols] = -np.inf
        second_b = similarity.max(axis=0)
        keep = (
            valid_a
            & valid_b[nearest_b]
            & (nearest_a[nearest_b] == rows)
            & (best_a >= self.min_similarity)
            & (best_a > second_a)
            & (best_b[nearest_b] > second_b[nearest_b])
        )
        left, right = rows[keep], nearest_b[keep]
        return PairMatches(
            np.column_stack([left, right]).astype(np.int32),
            best_a[keep].astype(np.float32),
            np.linalg.norm(a[left] - b[right], axis=1).astype(np.float32),
        )


class LightGlueMatcher:
    """Match existing SuperPoint arrays without loading an image extractor."""

    def __init__(self, model_dir: Path, *, device="cpu", threshold=0.1, threads=4):
        if not 0 <= threshold <= 1:
            raise ValueError("LightGlue threshold must be between 0 and 1")
        import torch
        from lightglue import LightGlue

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select cpu or auto")
        if device == "cpu":
            torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        model_dir.mkdir(parents=True, exist_ok=True)
        previous = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(model_dir))
            self.model = (
                LightGlue(features="superpoint", filter_threshold=threshold).eval().to(device)
            )
        finally:
            torch.hub.set_dir(previous)
        weights = model_dir / "checkpoints" / "superpoint_lightglue_v0-1_arxiv.pth"
        self.provenance = {
            "name": "lightglue-superpoint",
            "implementation_revision": LIGHTGLUE_REVISION,
            "weights_sha256": hash_file(weights),
            "filter_threshold": threshold,
            "device": device,
            "torch": torch.__version__,
            "score": "LightGlue match confidence",
            "n_layers": 9,
            "depth_confidence": 0.95,
            "width_confidence": 0.99,
        }

    def __call__(self, first, second):
        if not len(first.keypoints) or not len(second.keypoints):
            return PairMatches.empty()
        torch = self.torch

        def features(frame):
            return {
                "keypoints": torch.from_numpy(frame.keypoints)[None].to(self.device),
                "descriptors": torch.from_numpy(frame.descriptors)[None].to(self.device),
                "image_size": torch.tensor([[frame.width, frame.height]], device=self.device),
            }

        with torch.inference_mode():
            result = self.model({"image0": features(first), "image1": features(second)})
        indices = result["matches"][0].cpu().numpy().astype(np.int32)
        scores = result["scores"][0].cpu().numpy().astype(np.float32)
        distances = np.linalg.norm(
            first.descriptors[indices[:, 0]] - second.descriptors[indices[:, 1]], axis=1
        )
        order = np.argsort(indices[:, 0], kind="stable")
        return PairMatches(indices[order], scores[order], distances[order])
