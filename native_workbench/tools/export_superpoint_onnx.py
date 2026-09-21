#!/usr/bin/env python3
"""Export fixed-shape SuperPoint with fixed top-k outputs for TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
import onnx
from lightglue import SuperPoint
from lightglue.superpoint import simple_nms


class ExportableSuperPoint(torch.nn.Module):
    def __init__(self, model: SuperPoint, max_keypoints: int):
        super().__init__()
        self.model = model
        self.max_keypoints = max_keypoints

    def forward(self, image: torch.Tensor):
        model = self.model
        x = model.relu(model.conv1a(image))
        x = model.relu(model.conv1b(x))
        x = model.pool(x)
        x = model.relu(model.conv2a(x))
        x = model.relu(model.conv2b(x))
        x = model.pool(x)
        x = model.relu(model.conv3a(x))
        x = model.relu(model.conv3b(x))
        x = model.pool(x)
        x = model.relu(model.conv4a(x))
        x = model.relu(model.conv4b(x))

        detector = model.convPb(model.relu(model.convPa(x)))
        detector = functional.softmax(detector, 1)[:, :-1]
        batch, _, height, width = detector.shape
        scores = detector.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch, height * 8, width * 8)
        scores = simple_nms(scores, model.conf.nms_radius)
        border = model.conf.remove_borders
        scores[:, :border] = -1
        scores[:, :, :border] = -1
        scores[:, -border:] = -1
        scores[:, :, -border:] = -1
        top_scores, indices = torch.topk(scores.flatten(1), self.max_keypoints, dim=1)
        keypoints = torch.stack((indices % scores.shape[2], indices // scores.shape[2]), dim=-1)
        keypoints = keypoints.float()

        dense_descriptors = model.convDb(model.relu(model.convDa(x)))
        dense_descriptors = functional.normalize(dense_descriptors, p=2, dim=1)
        sampling_points = keypoints - 3.5
        sampling_points = sampling_points / torch.tensor(
            [scores.shape[2] - 4.5, scores.shape[1] - 4.5], device=image.device
        )
        sampling_points = sampling_points * 2 - 1
        descriptors = functional.grid_sample(
            dense_descriptors,
            sampling_points.view(batch, 1, self.max_keypoints, 2),
            mode="bilinear",
            align_corners=True,
        )
        descriptors = functional.normalize(
            descriptors.reshape(batch, 256, self.max_keypoints), p=2, dim=1
        ).transpose(1, 2)
        return keypoints, top_scores, descriptors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".slam-cache/models"),
        help="Torch hub directory containing checkpoints/superpoint_v1.pth",
    )
    args = parser.parse_args()
    if args.width % 8 or args.height % 8:
        parser.error("width and height must be divisible by 8")
    if args.max_keypoints > args.width * args.height:
        parser.error("max-keypoints exceeds the score-map size")

    previous_hub_dir = torch.hub.get_dir()
    try:
        torch.hub.set_dir(str(args.model_dir))
        model = SuperPoint(max_num_keypoints=args.max_keypoints).eval()
    finally:
        torch.hub.set_dir(previous_hub_dir)
    exportable = ExportableSuperPoint(model, args.max_keypoints).eval()
    sample = torch.zeros(1, 1, args.height, args.width, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exportable,
        (sample,),
        args.output,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        opset_version=args.opset,
        dynamo=True,
    )
    # Keep the graph self-contained so trtexec can consume a single model file.
    exported = onnx.load(args.output)
    onnx.save_model(exported, args.output, save_as_external_data=False)
    args.output.with_suffix(args.output.suffix + ".data").unlink(missing_ok=True)
    onnx.checker.check_model(onnx.load(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
