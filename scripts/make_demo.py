"""Create a small textured, moving test video without downloading sample footage."""

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def make_demo(path: Path, frames: int = 30):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=15)
        stream.width, stream.height, stream.pix_fmt = 480, 320, "bgr0"
        background = Image.fromarray(rng.integers(20, 65, (320, 480, 3), dtype=np.uint8))
        for index in range(frames):
            image = background.copy()
            draw = ImageDraw.Draw(image)
            for y in range(40, 290, 45):
                for x in range(40, 440, 45):
                    offset = index % 20
                    draw.rectangle((x + offset, y, x + offset + 19, y + 19), fill=(220, 190, 80))
            draw.ellipse((150 + index, 90, 280 + index, 220), fill=(50, 170, 210))
            draw.text((12, 12), f"SLAM LAB / frame {index:03d}", fill="white")
            frame = av.VideoFrame.from_image(image)
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo.mkv"))
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    make_demo(args.output, args.frames)
    print(args.output.resolve())
