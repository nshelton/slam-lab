"""Structured, atomically published status for solver jobs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from slam_lab import __version__
from slam_lab.artifacts import atomic_write_json, opaque_id, utc_now


class SolveJob:
    def __init__(
        self,
        geometry: Path,
        output: Path,
        artifact_id: str,
        *,
        invocation: list[str] | None = None,
    ):
        self.geometry = Path(geometry).resolve()
        self.output = Path(output).resolve()
        self.job_id = opaque_id("job")
        self.root = self.output.parent / "jobs" / self.job_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.revision = 0
        created_at = utc_now()
        self.job = {
            "schema_version": 1,
            "job_id": self.job_id,
            "job_type": "solve",
            "arguments": invocation
            or [
                sys.executable,
                "-m",
                "slam_lab",
                "solve",
                str(self.geometry),
                "--output",
                str(self.output),
            ],
            "cwd": str(Path.cwd().resolve()),
            "interpreter": sys.executable,
            "python_version": sys.version.split()[0],
            "slam_lab_version": __version__,
            "input": {"artifact_id": None, "location": str(self.geometry)},
            "output": {"artifact_id": artifact_id, "location": str(self.output)},
            "effective_options": None,
            "environment": {
                key: os.environ[key]
                for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                if key in os.environ
            },
            "created_at": created_at,
            "started_at": created_at,
            "finished_at": None,
            "exit_code": None,
            "termination_signal": None,
            "retry_of_job_id": None,
        }
        self._write_job()
        self.update(
            state="starting",
            phase="loading_inputs",
            output_artifact_id=artifact_id,
            registered_frames=0,
            total_frames=None,
            points=0,
            current_frame=None,
            bundle_adjustment_iteration=None,
            latest_snapshot=None,
            last_error=None,
        )

    def _write_job(self) -> None:
        atomic_write_json(self.root / "job.json", self.job)

    def configure(self, *, input_artifact_id: str, effective_options: dict) -> None:
        self.job["input"]["artifact_id"] = input_artifact_id
        self.job["effective_options"] = effective_options
        self._write_job()

    def finish(self, *, exit_code: int, termination_signal: str | None = None) -> None:
        self.job.update(
            finished_at=utc_now(), exit_code=exit_code, termination_signal=termination_signal
        )
        self._write_job()

    def update(self, *, state: str | None = None, phase: str | None = None, **values) -> dict:
        path = self.root / "status.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        self.revision += 1
        current.update(values)
        current.update(
            {
                "schema_version": 1,
                "job_id": self.job_id,
                "state": state if state is not None else current.get("state", "running"),
                "phase": phase if phase is not None else current.get("phase"),
                "revision": self.revision,
                "updated_at": utc_now(),
                "worker_pid": os.getpid(),
            }
        )
        atomic_write_json(path, current)
        return current

    def fail(self, error: BaseException, *, canceled: bool = False) -> None:
        self.update(
            state="canceled" if canceled else "failed",
            phase="canceled" if canceled else "failed",
            current_frame=None,
            last_error={"code": type(error).__name__, "message": str(error)},
        )
        self.finish(
            exit_code=130 if canceled else 1,
            termination_signal="SIGINT" if canceled else None,
        )


def _job_root_for(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    if (path / "status.json").is_file() and (path / "job.json").is_file():
        return path
    if (path / "metadata.json").is_file():
        metadata = json.loads((path / "metadata.json").read_text())
        job_id = metadata.get("producer_job_id")
        if job_id and (path.parent / "jobs" / job_id / "status.json").is_file():
            return path.parent / "jobs" / job_id
    jobs = path.parent / "jobs"
    if jobs.is_dir():
        for job_file in jobs.glob("*/job.json"):
            job = json.loads(job_file.read_text())
            location = job.get("output", {}).get("location")
            if location and Path(location).resolve() == path:
                return job_file.parent
    return None


def solve_status(path: Path) -> dict:
    root = _job_root_for(Path(path))
    if root is None:
        return {
            "schema_version": 1,
            "state": "not_started",
            "phase": "not_started",
            "revision": 0,
            "last_error": None,
        }
    status = json.loads((root / "status.json").read_text())
    pid = status.get("worker_pid")
    alive = False
    if pid and status.get("state") in {"starting", "running"}:
        try:
            os.kill(pid, 0)
            alive = True
        except (OSError, ValueError):
            pass
    status["worker_alive"] = alive
    status["job_directory"] = str(root)
    return status
