"""Matcher-independent two-view verification, stored separately from raw matches."""

import fcntl
import hashlib
import json
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from tqdm import tqdm

from slam_lab.config import fingerprint
from slam_lab.correspondence import MatchStore, associate_tracks, unpack_matches
from slam_lab.geometry import Camera, triangulate


@dataclass(frozen=True)
class VerificationConfig:
    threshold_px: float = 1.5
    reprojection_px: float = 3.0
    min_inliers: int = 30
    min_parallax_deg: float = 2.0
    min_coverage: float = 0.2
    max_homography_fraction: float = 0.9
    max_iterations: int = 3000
    seed: int = 7

    def __post_init__(self):
        for value in (self.threshold_px, self.reprojection_px, self.min_parallax_deg):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Geometry thresholds must be finite and positive")
        if self.min_parallax_deg >= 90 or self.min_inliers < 8 or self.max_iterations < 1:
            raise ValueError("Invalid parallax, minimum support, or iteration limit")
        if not 0 < self.min_coverage <= 1 or not 0 < self.max_homography_fraction <= 1:
            raise ValueError("Coverage and homography fraction must be in (0, 1]")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must fit a nonnegative signed 32-bit integer")


def sampson_errors(matrix, first, second):
    """Square-root Sampson distance, in pixels for a pixel-coordinate F."""
    a, b = np.c_[first, np.ones(len(first))], np.c_[second, np.ones(len(second))]
    fa, ftb = a @ matrix.T, b @ matrix
    denominator = np.sum(fa[:, :2] ** 2 + ftb[:, :2] ** 2, axis=1)
    return np.abs(np.sum(b * fa, axis=1)) / np.sqrt(np.maximum(denominator, 1e-30))


def coverage(xy, camera):
    """Fraction of occupied cells in a 4 by 4 image grid."""
    if not len(xy):
        return 0.0
    cells = np.clip((xy / [camera.width, camera.height] * 4).astype(int), 0, 3)
    return len(np.unique(cells[:, 1] * 4 + cells[:, 0])) / 16


def verify_pair(first_xy, second_xy, camera, config=None):
    config = config or VerificationConfig()
    a, b = np.asarray(first_xy, np.float64), np.asarray(second_xy, np.float64)
    if a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape:
        raise ValueError("Correspondences must be equally sized Nx2 pixel arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Correspondences must be finite")
    n = len(a)
    data = {
        name: np.zeros(n, bool)
        for name in (
            "F_inliers",
            "E_inliers",
            "H_inliers",
            "cheirality",
            "triangulated",
            "verified",
        )
    }
    data.update({name: np.full((3, 3), np.nan) for name in ("F", "E", "H")})
    data.update(
        {
            name: np.full(n, np.nan)
            for name in ("F_error_px", "E_error_px", "H_error_px", "parallax_deg")
        }
    )
    data["pose_second_from_first"] = np.full((4, 4), np.nan)
    summary = {"matches": n, "seed_eligible": False, "reasons": []}

    def fit(name, estimator, *args, **kwargs):
        try:
            return estimator(*args, **kwargs)
        except cv2.error as exc:
            # Degenerate real inputs can make OpenCV throw instead of returning no model.
            summary.setdefault("estimator_errors", {})[name] = str(exc)
            return None, None

    cv2.setRNGSeed(config.seed)
    if n >= 8:
        fundamental, mask = fit(
            "F",
            cv2.findFundamentalMat,
            a,
            b,
            cv2.USAC_MAGSAC,
            config.threshold_px,
            0.999,
            config.max_iterations,
        )
        if fundamental is not None and fundamental.shape == (3, 3) and mask is not None:
            data["F"] = fundamental
            data["F_inliers"] = mask.ravel().astype(bool).copy()
            data["F_error_px"] = sampson_errors(fundamental, a, b)
        homography, mask = fit(
            "H",
            cv2.findHomography,
            a,
            b,
            cv2.RANSAC,
            config.threshold_px,
            maxIters=config.max_iterations,
            confidence=0.999,
        )
        if homography is not None and mask is not None:
            data["H"] = homography
            data["H_inliers"] = mask.ravel().astype(bool).copy()
            projected = cv2.perspectiveTransform(a[:, None], homography)[:, 0]
            data["H_error_px"] = np.linalg.norm(projected - b, axis=1)
        essential, mask = fit(
            "E",
            cv2.findEssentialMat,
            a,
            b,
            camera.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=config.threshold_px,
            maxIters=config.max_iterations,
        )
        if essential is not None and mask is not None:
            # recoverPose mutates its mask: retain the estimator output independently.
            data["E_inliers"] = mask.ravel().astype(bool).copy()
            best_count = -1
            for matrix in essential.reshape(-1, 3, 3):
                if not np.isfinite(matrix).all():
                    continue
                try:
                    _, rotation, translation, recovered = cv2.recoverPose(
                        matrix, a, b, camera.K, mask=mask.copy()
                    )
                except cv2.error as exc:
                    summary.setdefault("estimator_errors", {})["recoverPose"] = str(exc)
                    continue
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = rotation, translation.ravel()
                _, valid, angles = triangulate(
                    a,
                    b,
                    np.eye(4),
                    pose,
                    camera,
                    max_error=config.reprojection_px,
                    min_angle_deg=config.min_parallax_deg,
                )
                cheirality = recovered.ravel().astype(bool)
                valid &= cheirality & data["F_inliers"]
                if int(valid.sum()) > best_count:
                    best_count = int(valid.sum())
                    inverse = np.linalg.inv(camera.K)
                    data.update(
                        E=matrix,
                        E_error_px=sampson_errors(inverse.T @ matrix @ inverse, a, b),
                        cheirality=cheirality,
                        triangulated=valid,
                        parallax_deg=angles,
                        pose_second_from_first=pose,
                    )
    verified = data["F_inliers"] & data["E_inliers"]
    # Weak fits remain inspectable but do not become track edges.
    if verified.sum() >= config.min_inliers:
        data["verified"] = verified
    stable = data["triangulated"] & data["verified"]
    support = int(stable.sum())
    cov = min(coverage(a[stable], camera), coverage(b[stable], camera))
    h_fraction = float(np.count_nonzero(data["H_inliers"] & verified) / max(1, verified.sum()))
    median_angle = float(np.median(data["parallax_deg"][stable])) if support else 0.0
    reasons = summary["reasons"]
    if n < 8:
        reasons.append("insufficient_matches")
    if support < config.min_inliers:
        reasons.append("insufficient_positive_depth_parallax_support")
    if cov < config.min_coverage:
        reasons.append("limited_spatial_coverage")
    if h_fraction >= config.max_homography_fraction:
        reasons.append("homography_dominated")
    summary.update(
        F_inliers=int(data["F_inliers"].sum()),
        E_inliers=int(data["E_inliers"].sum()),
        H_inliers=int(data["H_inliers"].sum()),
        verified=int(data["verified"].sum()),
        triangulated=support,
        coverage=cov,
        homography_fraction=h_fraction,
        positive_depth_fraction=float(data["cheirality"].sum() / max(1, data["E_inliers"].sum())),
        median_parallax_deg=median_angle,
        seed_eligible=not reasons,
        seed_score=float(support * cov * min(median_angle, 15.0)) if not reasons else 0.0,
    )
    return data, summary


class GeometryStore(AbstractContextManager):
    def __init__(self, path, *, create=False):
        self.path = Path(path)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "ro"
        self.db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode={mode}", uri=True)
        if create:
            self.db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS frames (
                    row_id INTEGER PRIMARY KEY, frame_index INTEGER, timestamp_ns INTEGER,
                    width INTEGER, height INTEGER, jpeg BLOB, keypoints BLOB,
                    keypoint_count INTEGER);
                CREATE TABLE IF NOT EXISTS pairs (
                    first INTEGER, second INTEGER, source_hash TEXT, arrays BLOB, summary TEXT,
                    PRIMARY KEY(first,second));
            """)

    def __exit__(self, *args):
        self.db.close()

    get = MatchStore.get
    set = MatchStore.set
    frame = MatchStore.frame

    def pair(self, first, second):
        row = self.db.execute(
            "SELECT arrays,summary FROM pairs WHERE first=? AND second=?", (first, second)
        ).fetchone()
        if row is None:
            raise ValueError("Pair is not verified; supply its canonical selected-frame rows")
        with np.load(BytesIO(row[0]), allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        return data, json.loads(row[1])


def build_verified_tracks(store, output):
    rows = store.db.execute("SELECT row_id,keypoint_count FROM frames ORDER BY row_id").fetchall()
    frame_ids, counts = np.array(rows, dtype=int).T
    offsets = np.r_[0, np.cumsum(counts)]
    positions = {int(row): index for index, row in enumerate(frame_ids)}
    edges, scores = [], []
    digest = hashlib.sha256()
    for first, second, source_hash in store.db.execute(
        "SELECT first,second,source_hash FROM pairs ORDER BY first,second"
    ):
        data, _ = store.pair(first, second)
        chosen = data["verified"]
        edges.append(data["indices"][chosen] + offsets[[positions[first], positions[second]]])
        # Geometric residual is comparable across matchers; appearance scores are not.
        scores.append(-np.maximum(data["F_error_px"][chosen], data["E_error_px"][chosen]))
        digest.update(f"{first}:{second}:{source_hash}\n".encode())
    edge_array = np.concatenate(edges) if edges else np.empty((0, 2), int)
    score_array = np.concatenate(scores) if scores else np.empty(0)
    offsets, track_ids, lengths, summary = associate_tracks(counts, edge_array, score_array)
    identity = fingerprint({"geometry": store.get("identity"), "pairs": digest.hexdigest()})
    temp = output / "tracks.npz.tmp"
    with temp.open("wb") as file:
        np.savez_compressed(
            file,
            frame_rows=frame_ids,
            offsets=offsets,
            track_ids=track_ids,
            track_lengths=lengths,
            geometry_identity=np.array(identity),
        )
    temp.replace(output / "tracks.npz")
    summaries = [json.loads(row[0]) for row in store.db.execute("SELECT summary FROM pairs")]
    summary.update(
        geometry_verified=True,
        verified_edges=len(edge_array),
        frames=len(rows),
        pairs=len(summaries),
        seed_candidates=sum(s["seed_eligible"] for s in summaries),
        identity=identity,
        track_policy="residual-ordered-union-unique-frame-v1",
    )
    store.set(phase="complete", tracks_identity=identity, summary=summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify_matches(
    match_run,
    output,
    *,
    config=None,
    camera_options=None,
    frame_step=1,
    max_frames=None,
    progress=True,
):
    config = config or VerificationConfig()
    if frame_step < 1 or (max_frames is not None and max_frames < 2):
        raise ValueError("Need a positive frame step and at least two frames")
    source, output = Path(match_run).expanduser().resolve(), Path(output).expanduser().resolve()
    if source == output or (output.exists() and not (output / "geometry.sqlite3").exists()):
        raise FileExistsError("Choose a new geometry output or resume an existing geometry run")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This geometry run is already active") from exc
        with (
            MatchStore(source / "matches.sqlite3") as matches,
            GeometryStore(output / "geometry.sqlite3", create=True) as store,
        ):
            rows = matches.db.execute("SELECT * FROM frames ORDER BY row_id").fetchall()
            rows = rows[::frame_step][:max_frames]
            if len(rows) < 2:
                raise ValueError("Need at least two selected frames")
            size = rows[0][3:5]
            if any(row[3:5] != size for row in rows):
                raise ValueError("The fixed camera model requires equal image sizes")
            camera = Camera.from_size(*size, **(camera_options or {}))
            identity = {
                "schema": 1,
                "source": matches.get("identity"),
                "frame_rows": [r[0] for r in rows],
                "camera": asdict(camera),
                "config": asdict(config),
                "opencv": cv2.__version__,
                "estimators": "F-USAC_MAGSAC_E-RANSAC_H-RANSAC-v1",
            }
            if store.get("identity") not in (None, identity):
                raise ValueError("Geometry inputs or settings changed; choose a new --output")
            with store.db:
                store.db.executemany("INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?,?,?)", rows)
            store.set(
                identity=identity,
                source_match_run=str(source),
                phase="verifying",
                feature_cache=matches.get("feature_cache"),
            )
            selected = set(identity["frame_rows"])
            pending = [
                (a, b)
                for a, b in matches.db.execute("SELECT first,second FROM pairs")
                if a in selected and b in selected
            ]
            store.set(
                available_pairs=len(pending),
                possible_pairs=len(rows) * (len(rows) - 1) // 2,
                source_complete=matches.get("phase") == "complete",
                error=None,
            )
            done = dict(
                ((a, b), h)
                for a, b, h in store.db.execute("SELECT first,second,source_hash FROM pairs")
            )
            frames = {row[0]: store.frame(row[0]) for row in rows}
            started = perf_counter()
            try:
                for first, second in tqdm(
                    sorted(pending), desc="Verifying geometry", disable=not progress
                ):
                    blob = matches.db.execute(
                        "SELECT matches FROM pairs WHERE first=? AND second=?", (first, second)
                    ).fetchone()[0]
                    source_hash = hashlib.sha256(blob).hexdigest()
                    if (first, second) in done:
                        if done[first, second] != source_hash:
                            raise ValueError(
                                "Previously verified raw matches changed; use a new output"
                            )
                        continue
                    pair = unpack_matches(blob)
                    a, b = frames[first], frames[second]
                    pair.validate(len(a["keypoints"]), len(b["keypoints"]))
                    data, summary = verify_pair(
                        a["keypoints"][pair.indices[:, 0]],
                        b["keypoints"][pair.indices[:, 1]],
                        camera,
                        config,
                    )
                    data.update(indices=pair.indices, scores=pair.scores)
                    archive = BytesIO()
                    np.savez_compressed(archive, **data)
                    with store.db:
                        store.db.execute(
                            "INSERT INTO pairs VALUES (?,?,?,?,?)",
                            (first, second, source_hash, archive.getvalue(), json.dumps(summary)),
                        )
                store.set(phase="building_tracks")
                summary = build_verified_tracks(store, output)
                store.set(elapsed_seconds=perf_counter() - started)
                return summary
            except KeyboardInterrupt:
                store.set(phase="paused")
                raise
            except Exception as exc:
                store.set(phase="failed", error=str(exc))
                raise


def geometry_status(path):
    path = Path(path).expanduser().resolve()
    running = False
    if path.with_suffix(".lock").exists():
        with path.with_suffix(".lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                running = True
    if not (path / "geometry.sqlite3").exists():
        return {"phase": "starting" if running else "not_started", "running": running}
    with GeometryStore(path / "geometry.sqlite3") as store:
        return {
            "running": running,
            "phase": store.get("phase"),
            "completed_pairs": store.db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],
            "available_pairs": store.get("available_pairs"),
            "possible_pairs": store.get("possible_pairs"),
            "source_complete": store.get("source_complete"),
            "error": store.get("error"),
            "summary": store.get("summary") if store.get("phase") == "complete" else None,
        }
