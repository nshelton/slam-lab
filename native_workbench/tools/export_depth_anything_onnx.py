#!/usr/bin/env python3
"""Export fixed-shape Depth Anything V2 (metric) to ONNX for TensorRT.

The graph takes ImageNet-normalized RGB (float32, 1x3xHxW; the workbench's
preprocessing kernel normalizes) and returns metric depth in metres (float32,
1xHxW). Weights and compute are float16 unless --fp32: TensorRT builds
strongly typed networks, so the ONNX types decide the engine's precision.

Needs a checkout of https://github.com/DepthAnything/Depth-Anything-V2 and a
metric checkpoint, e.g. depth_anything_v2_metric_hypersim_vits.pth (indoor,
max depth 20 m) or depth_anything_v2_metric_vkitti_vits.pth (outdoor, 80 m).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch

CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class Exportable(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.to(self.dtype)).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(".slam-cache/Depth-Anything-V2"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".slam-cache/models/depth_anything_v2/depth_anything_v2_metric_hypersim_vits.pth"),
    )
    parser.add_argument("--encoder", choices=sorted(CONFIGS), default="vits")
    parser.add_argument("--max-depth", type=float, default=20.0, help="20 hypersim, 80 vkitti")
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=294)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if args.width % 14 or args.height % 14:
        parser.error("width and height must be multiples of 14 (the ViT patch size)")

    sys.path.insert(0, str(args.repo / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**CONFIGS[args.encoder], max_depth=args.max_depth)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    dtype = torch.float32 if args.fp32 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exportable = Exportable(model.to(device, dtype).eval(), dtype).eval()
    sample = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            exportable,
            (sample,),
            args.output,
            input_names=["image"],
            output_names=["depth"],
            opset_version=args.opset,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
