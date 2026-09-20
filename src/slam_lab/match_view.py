"""Inspect tentative pair matches and global appearance tracks in Rerun."""

import json
from pathlib import Path

import numpy as np

from slam_lab.correspondence import MatchStore
from slam_lab.rerun_support import init_recording


def colors_for_ids(ids):
    values = np.asarray(ids, dtype=np.uint64) * 2654435761
    return (64 + np.column_stack([(values >> shift) % 192 for shift in (0, 8, 16)])).astype(
        np.uint8
    )


def view_matches(path, *, output=None, pair=None, max_links=None, min_track_length=2):
    import rerun as rr
    import rerun.blueprint as rrb

    path = Path(path).expanduser().resolve()
    if (max_links is not None and max_links < 0) or min_track_length < 1:
        raise ValueError("max-links must be nonnegative; min-track-length must be positive")
    if output is not None:
        output = Path(output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Recording already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    with MatchStore(path / "matches.sqlite3") as store:
        size = store.get("frames")
        counts = np.full((size, size), -1, np.int32)
        for first, second, count in store.db.execute("SELECT first,second,count FROM pairs"):
            counts[first, second] = counts[second, first] = count
        # Unknown pairs are gray; processed zero-match pairs are black; diagonal is white.
        strength = np.log1p(np.maximum(counts, 0)) / np.log1p(max(int(counts.max()), 1))
        heatmap = np.stack([strength * 30, strength * 220, strength * 255], axis=-1).astype(
            np.uint8
        )
        heatmap[counts < 0] = [75, 75, 75]
        heatmap[np.arange(size), np.arange(size)] = [220, 220, 220]
        if pair is None and not (path / "tracks.npz").exists():
            latest = store.db.execute(
                "SELECT first,second FROM pairs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if latest is None:
                raise ValueError("No pairs ready yet; check match-status shortly")
            pair = latest
        if pair is not None:
            matches = store.pair(*pair)
            frames = [store.frame(row) for row in pair]
            chosen = np.argsort(-matches.scores, kind="stable")[:max_links]
            bounds = rrb.VisualBounds2D(
                x_range=[0, frames[0]["width"] + 32 + frames[1]["width"]],
                y_range=[0, max(f["height"] for f in frames)],
            )
        recording = init_recording("slam-lab-correspondences", output=output)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Tabs(
                        rrb.Spatial2DView(
                            name="Match coverage (green matched / red unmatched)",
                            origin="/coverage",
                            visual_bounds=bounds,
                        ),
                        rrb.Spatial2DView(
                            name=f"Correspondences ({len(chosen)} / {len(matches.indices)} lines)",
                            origin="/pair",
                            visual_bounds=bounds,
                        ),
                        active_tab=0,
                    )
                    if pair is not None
                    else rrb.Spatial2DView(name="Appearance tracks", origin="/camera"),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="N × N match counts", origin="/matrix"),
                        rrb.TextDocumentView(name="Matching status", origin="/info"),
                    ),
                    column_shares=[3, 1],
                ),
                rrb.TimePanel(timeline="video_time")
                if pair is None
                else rrb.TimePanel(state="collapsed"),
            )
        )
        try:
            rr.log("matrix/counts", rr.Image(heatmap), static=True)
            info = {
                "matcher": store.get("identity")["matcher"]["name"],
                "frames": size,
                "completed_pairs": int(np.count_nonzero(np.triu(counts >= 0, k=1))),
                "total_pairs": store.get("total_pairs"),
                "matrix": "row/column = selected-frame row; gray = pending; brighter = more",
                "geometry": "Tentative appearance matches. No RANSAC, poses or triangulation.",
                "snapshot": "Reopen to refresh results from a running job.",
            }
            if pair is not None:
                shift = np.array([frames[0]["width"] + 32, 0], dtype=np.float32)
                colors = colors_for_ids(np.arange(len(matches.indices)))
                for side, frame in enumerate(frames):
                    root = f"pair/{side}"
                    coverage = f"coverage/{side}"
                    if side:
                        rr.log(root, rr.Transform3D(translation=[*shift, 0]), static=True)
                        rr.log(coverage, rr.Transform3D(translation=[*shift, 0]), static=True)
                    rr.log(
                        root + "/image",
                        rr.EncodedImage(contents=frame["jpeg"], media_type="image/jpeg"),
                        static=True,
                    )
                    rr.log(
                        coverage + "/image",
                        rr.EncodedImage(contents=frame["jpeg"], media_type="image/jpeg"),
                        static=True,
                    )
                    unmatched = np.ones(len(frame["keypoints"]), dtype=bool)
                    unmatched[matches.indices[:, side]] = False
                    if unmatched.any():
                        rr.log(
                            coverage + "/unmatched",
                            rr.Points2D(
                                frame["keypoints"][unmatched], colors=[255, 65, 80], radii=2.5
                            ),
                            rr.AnyValues(feature_id=np.flatnonzero(unmatched)),
                            static=True,
                        )
                    rr.log(
                        root + "/all_features",
                        rr.Points2D(frame["keypoints"], colors=[100, 100, 100], radii=1),
                        static=True,
                    )
                    if len(matches.indices):
                        rr.log(
                            coverage + "/matched",
                            rr.Points2D(
                                frame["keypoints"][matches.indices[:, side]],
                                colors=[40, 235, 95],
                                radii=2.5,
                            ),
                            rr.AnyValues(
                                match_id=np.arange(len(matches.indices)),
                                feature_id=matches.indices[:, side],
                                confidence=matches.scores,
                            ),
                            static=True,
                        )
                        rr.log(
                            root + "/matched",
                            rr.Points2D(
                                frame["keypoints"][matches.indices[:, side]],
                                colors=colors,
                                radii=2.5,
                            ),
                            rr.AnyValues(
                                match_id=np.arange(len(matches.indices)),
                                feature_id=matches.indices[:, side],
                                confidence=matches.scores,
                                descriptor_distance=matches.distances,
                            ),
                            static=True,
                        )
                if len(chosen):
                    endpoints = np.stack(
                        [
                            frames[0]["keypoints"][matches.indices[chosen, 0]],
                            frames[1]["keypoints"][matches.indices[chosen, 1]] + shift,
                        ],
                        axis=1,
                    )
                    rr.log(
                        "pair/links",
                        rr.LineStrips2D(endpoints, colors=colors[chosen], radii=0.5),
                        static=True,
                    )
                info.update(
                    selected_frame_rows=list(pair),
                    times_seconds=[f["timestamp_ns"] / 1e9 for f in frames],
                    matches=len(matches.indices),
                    detected_per_image=[len(f["keypoints"]) for f in frames],
                    unmatched_per_image=[
                        len(f["keypoints"]) - len(matches.indices) for f in frames
                    ],
                    displayed_links=len(chosen),
                    hidden_links=len(matches.indices) - len(chosen),
                    coverage="GREEN = appearance match; RED = unmatched detection. No RANSAC yet.",
                    empty_regions="No point means no cached SuperPoint detection at that location.",
                    colors="Same color and match_id identify corresponding points",
                    features="Gray = unmatched; all matched points are shown",
                )
                count = 2
            else:
                with np.load(path / "tracks.npz", allow_pickle=False) as archive:
                    offsets, track_ids, lengths = (
                        archive[key] for key in ("offsets", "track_ids", "track_lengths")
                    )
                for row in range(size):
                    frame = store.frame(row)
                    ids = track_ids[offsets[row] : offsets[row + 1]]
                    visible = lengths[ids] >= min_track_length
                    rr.set_time("frame", sequence=frame["index"])
                    rr.set_time("video_time", duration=np.timedelta64(frame["timestamp_ns"], "ns"))
                    rr.log(
                        "camera/image",
                        rr.EncodedImage(contents=frame["jpeg"], media_type="image/jpeg"),
                    )
                    rr.log("camera/image/tracks", rr.Clear(recursive=False))
                    if visible.any():
                        rr.log(
                            "camera/image/tracks",
                            rr.Points2D(
                                frame["keypoints"][visible],
                                colors=colors_for_ids(ids[visible]),
                                radii=2,
                            ),
                            rr.AnyValues(track_id=ids[visible], track_length=lengths[ids[visible]]),
                        )
                info.update(
                    summary=store.get("summary"),
                    min_displayed_track_length=min_track_length,
                    colors="Stable color per global track; click points for track ID/length",
                )
                count = size
            rr.log("info", rr.TextDocument(json.dumps(info, indent=2)), static=True)
            recording.flush()
        finally:
            rr.disconnect()
        return count
