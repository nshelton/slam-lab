"""Inspect cached raw RANSAC decisions without re-estimating geometry."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from slam_lab.correspondence import MatchStore
from slam_lab.rerun_support import init_recording
from slam_lab.verification import GeometryStore
from slam_lab.video import VideoReader


def view_geometry(path, *, pair=None, output=None):
    import rerun as rr
    import rerun.blueprint as rrb

    path = Path(path).expanduser().resolve()
    if output is not None:
        output = Path(output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    with GeometryStore(path / "geometry.sqlite3") as store:
        summaries = [
            (a, b, json.loads(s))
            for a, b, s in store.db.execute(
                "SELECT first,second,summary FROM pairs ORDER BY first,second"
            )
        ]
        if not summaries:
            raise ValueError("No verified pairs to view")
        if pair is None:
            a, b, _ = max(summaries, key=lambda p: p[2]["seed_score"])
        else:
            a, b = sorted(pair)
        data, summary = store.pair(a, b)
        first, second = store.frame(a), store.frame(b)
        match_run = Path(store.get("source_match_run"))
        with MatchStore(match_run / "matches.sqlite3") as matches:
            source = Path(matches.get("source"))
        decoded = {}
        if source.is_file():
            wanted = {first["index"], second["index"]}
            with VideoReader(source) as reader:
                for frame in reader.frames(
                    stride=1, max_side=max(first["width"], first["height"])
                ):
                    if frame.index in wanted:
                        decoded[frame.index] = Image.fromarray(frame.image)
                    if len(decoded) == 2:
                        break
        left = decoded.get(first["index"], Image.new("RGB", (first["width"], first["height"])))
        right = decoded.get(
            second["index"], Image.new("RGB", (second["width"], second["height"]))
        )
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        xy_a = first["keypoints"][data["indices"][:, 0]]
        xy_b = second["keypoints"][data["indices"][:, 1]] + [left.width, 0]
        models = [
            ("F", "F_inliers"),
            ("E", "E_inliers"),
            ("H", "H_inliers"),
            ("Verified", "verified"),
            ("Triangulation", "triangulated"),
        ]
        recording = init_recording("slam-lab-pair-geometry", output=output)
        try:
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Tabs(
                            *[rrb.Spatial2DView(name=name, origin=f"/{name}") for name, _ in models]
                        ),
                        rrb.Vertical(
                            rrb.TensorView(name="Verified support", origin="/graph"),
                            rrb.TextDocumentView(name="Pair diagnostics", origin="/info"),
                        ),
                        column_shares=[3, 1],
                    )
                )
            )
            for name, mask_name in models:
                keep = data[mask_name]
                colors = np.where(keep[:, None], [40, 235, 95], [255, 65, 80]).astype(np.uint8)
                rr.log(name, rr.Image(np.asarray(canvas)), static=True)
                rr.log(
                    f"{name}/lines",
                    rr.LineStrips2D(np.stack([xy_a, xy_b], axis=1), colors=colors),
                    static=True,
                )
                for side, xy, column in (("first", xy_a, 0), ("second", xy_b, 1)):
                    rr.log(
                        f"{name}/{side}",
                        rr.Points2D(xy, colors=colors, radii=2),
                        rr.AnyValues(
                            match_row=np.arange(len(keep)),
                            inlier=keep,
                            keypoint_id=data["indices"][:, column],
                            F_error_px=data["F_error_px"],
                            E_error_px=data["E_error_px"],
                            parallax_deg=data["parallax_deg"],
                        ),
                        static=True,
                    )
            rows = [r[0] for r in store.db.execute("SELECT row_id FROM frames ORDER BY row_id")]
            positions = {row: i for i, row in enumerate(rows)}
            matrix = np.full((len(rows), len(rows)), -1, np.int32)
            for first_row, second_row, stats in summaries:
                i, j = positions[first_row], positions[second_row]
                matrix[i, j] = matrix[j, i] = stats["verified"]
            rr.log("graph", rr.Tensor(matrix), static=True)
            rr.log(
                "info",
                rr.TextDocument(
                    json.dumps(
                        {
                            "pair_match_rows": [a, b],
                            "timestamps_seconds": [
                                first["timestamp_ns"] / 1e9,
                                second["timestamp_ns"] / 1e9,
                            ],
                            **summary,
                            "camera": store.get("identity")["camera"],
                            "matrix_row_ids": rows,
                            "legend": "Green: accepted by selected model. Red: rejected. "
                            "All raw matches shown.",
                            "masks": "F/E/H are raw RANSAC masks. "
                            "Verified=F AND E with minimum support. "
                            "Triangulation additionally checks depth, parallax and reprojection.",
                            "warning": "Inliers do not establish a static scene "
                            "or a correct calibration.",
                        },
                        indent=2,
                    )
                ),
                static=True,
            )
            recording.flush()
        finally:
            rr.disconnect()
    return [a, b]
