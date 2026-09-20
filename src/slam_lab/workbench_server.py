"""Read-only HTTP adapter for the browser-based SLAM workbench."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from slam_lab.correspondence import MatchStore, match_status


class WorkbenchData:
    """Translate an existing match artifact into browser-friendly values."""

    def __init__(self, run: Path):
        self.run = Path(run).expanduser().resolve()
        if not (self.run / "matches.sqlite3").is_file():
            raise ValueError(f"Not a matching run: {self.run}")

    def overview(self):
        status = match_status(self.run)
        with MatchStore(self.run / "matches.sqlite3") as store:
            source = store.get("source")
            suggested = store.db.execute(
                "SELECT first,second,count FROM pairs WHERE second=first+1 "
                "ORDER BY count DESC,first LIMIT 1"
            ).fetchone()
            if suggested is None:
                suggested = store.db.execute(
                    "SELECT first,second,count FROM pairs ORDER BY count DESC,first,second LIMIT 1"
                ).fetchone()
        return {
            "run": str(self.run),
            "source": source,
            "status": status,
            "suggested_pair": (
                {"first": suggested[0], "second": suggested[1], "matches": suggested[2]}
                if suggested
                else None
            ),
        }

    @staticmethod
    def _frame_info(row: int, frame: dict):
        return {
            "row": row,
            "frame_index": frame["index"],
            "timestamp_ns": frame["timestamp_ns"],
            "width": frame["width"],
            "height": frame["height"],
            "keypoint_count": len(frame["keypoints"]),
            "image_url": f"/api/frames/{row}/image",
        }

    @staticmethod
    def _runtime_for_pair(store: MatchStore, first: int, second: int):
        columns = {row[1] for row in store.db.execute("PRAGMA table_info(pairs)")}
        if "runtime_id" not in columns:
            return None
        runtime = store.db.execute(
            "SELECT runtime_id FROM pairs WHERE first=? AND second=?",
            (min(first, second), max(first, second)),
        ).fetchone()
        if runtime is None or runtime[0] is None:
            return None
        exists = store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_runtimes'"
        ).fetchone()
        if not exists:
            return None
        provenance = store.db.execute(
            "SELECT provenance FROM match_runtimes WHERE runtime_id=?", (runtime[0],)
        ).fetchone()
        return json.loads(provenance[0]) if provenance else {"runtime_id": runtime[0]}

    def pair(self, first: int, second: int):
        if first == second:
            raise ValueError("Choose two different frame rows")
        with MatchStore(self.run / "matches.sqlite3") as store:
            first_frame, second_frame = store.frame(first), store.frame(second)
            matches = store.pair(first, second)
            runtime = self._runtime_for_pair(store, first, second)
            first_xy = first_frame["keypoints"][matches.indices[:, 0]]
            second_xy = second_frame["keypoints"][matches.indices[:, 1]]
        return {
            "first": self._frame_info(first, first_frame),
            "second": self._frame_info(second, second_frame),
            "matcher": match_status(self.run)["matcher"],
            "pair_runtime": runtime,
            "matches": {
                "count": len(matches.indices),
                "indices": matches.indices.tolist(),
                "scores": matches.scores.tolist(),
                "distances": matches.distances.tolist(),
                "first_xy": first_xy.tolist(),
                "second_xy": second_xy.tolist(),
            },
        }

    def image(self, row: int):
        with MatchStore(self.run / "matches.sqlite3") as store:
            return store.frame(row)["jpeg"]


def make_handler(data: WorkbenchData, static_root: Path):
    static_root = static_root.resolve()

    class WorkbenchHandler(BaseHTTPRequestHandler):
        server_version = "SlamLabWorkbench/0.1"

        def _bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status=HTTPStatus.OK):
            self._bytes(
                json.dumps(value, separators=(",", ":")).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _error(self, status, message):
            self._json({"error": message}, status)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            path = unquote(urlparse(self.path).path)
            try:
                if path == "/api/run":
                    self._json(data.overview())
                    return
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["api", "pairs"]:
                    self._json(data.pair(int(parts[2]), int(parts[3])))
                    return
                if len(parts) == 4 and parts[:2] == ["api", "frames"] and parts[3] == "image":
                    self._bytes(data.image(int(parts[2])), "image/jpeg")
                    return
                self._static(path)
            except (ValueError, IndexError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
            except sqlite3.Error as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))

        def _static(self, path: str):
            relative = "index.html" if path == "/" else path.lstrip("/")
            candidate = (static_root / relative).resolve()
            if static_root not in candidate.parents and candidate != static_root:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if not candidate.is_file():
                # Client-side routes should still boot the workbench.
                candidate = static_root / "index.html"
            if not candidate.is_file():
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "Workbench frontend is not built. "
                    "Run npm install && npm run build in workbench/.",
                )
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._bytes(candidate.read_bytes(), content_type)

        def log_message(self, message, *args):
            print(f"{self.address_string()} - {message % args}")

    return WorkbenchHandler


def parser():
    default_static = Path(__file__).resolve().parents[2] / "workbench" / "dist"
    result = argparse.ArgumentParser(description="Serve a matching run in the SLAM workbench")
    result.add_argument("run", type=Path, help="Directory containing matches.sqlite3")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--static", type=Path, default=default_static)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    data = WorkbenchData(args.run)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data, args.static))
    print(f"SLAM workbench: http://{args.host}:{args.port}")
    print(f"Matching run: {data.run}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
