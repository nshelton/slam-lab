"""Replay cached previews and features into a Rerun viewer or recording."""

import json
from pathlib import Path

import numpy as np

from slam_lab.cache import FrameCache, resolve_cache
from slam_lab.rerun_support import init_recording


def view_cache(
    path: Path, *, output: Path | None = None, min_score: float = 0.0, point_radius: float = 2.0
) -> int:
    if not 0 <= min_score <= 1:
        raise ValueError("min-score must be between 0 and 1")
    if not np.isfinite(point_radius) or point_radius <= 0:
        raise ValueError("point-radius must be positive and finite")
    path = resolve_cache(path)
    if output is not None:
        output = output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}. Choose a new --save path.")
        output.parent.mkdir(parents=True, exist_ok=True)

    import rerun as rr
    import rerun.blueprint as rrb

    with FrameCache(path) as cache:
        count = cache.count()
        if not count:
            raise ValueError(f"Cache has no frames yet: {path}")
        recording = init_recording("slam-lab", output=output)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="SuperPoint", origin="/camera"),
                    rrb.Vertical(
                        rrb.TimeSeriesView(name="Keypoints", origin="/metrics/keypoints"),
                        rrb.TimeSeriesView(name="Inference (ms)", origin="/metrics/extraction_ms"),
                        rrb.TextDocumentView(name="Recording", origin="/info"),
                        row_shares=[1, 1, 2],
                    ),
                    column_shares=[3, 1],
                ),
                rrb.TimePanel(timeline="video_time"),
            )
        )
        info = {
            "source": cache.get("source_path"),
            "input_modality": "monocular RGB",
            "frames": count,
            "complete": cache.get("complete"),
            "extraction": cache.get("identity"),
            "runtime": cache.get("extractor"),
            "display_min_score": min_score,
            "point_colors": "red (score 0) to green (score >= 0.05)",
            "coordinates": "cached preview pixels; original size stored per frame",
        }
        rr.log("info", rr.TextDocument(json.dumps(info, indent=2)), static=True)
        try:
            for frame in cache.frames(descriptors=False):
                rr.set_time("frame", sequence=frame.index)
                rr.set_time("video_time", duration=np.timedelta64(frame.timestamp_ns, "ns"))
                rr.log(
                    "camera/image", rr.EncodedImage(contents=frame.jpeg, media_type="image/jpeg")
                )
                visible = frame.scores >= min_score
                scores = frame.scores[visible]
                if len(scores):
                    strength = np.clip(scores / 0.05, 0, 1)
                    colors = np.column_stack((1 - strength, strength, np.full(len(scores), 0.2)))
                    rr.log(
                        "camera/image/keypoints",
                        rr.Points2D(
                            frame.keypoints[visible],
                            colors=(colors * 255).astype(np.uint8),
                            radii=point_radius,
                        ),
                        rr.AnyValues(confidence=scores),
                    )
                else:
                    # An initial empty AnyValues cannot establish its component type.
                    # Clear all point data, including confidence, for empty overlays.
                    rr.log("camera/image/keypoints", rr.Clear(recursive=False))
                rr.log("metrics/keypoints/detected", rr.Scalars(len(frame.keypoints)))
                rr.log("metrics/keypoints/visible", rr.Scalars(int(visible.sum())))
                rr.log("metrics/extraction_ms", rr.Scalars(frame.extraction_ms))
            recording.flush()
        finally:
            rr.disconnect()
    return count
