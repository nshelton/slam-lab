"""Supervised CLI pipeline jobs launched by the browser workbench."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

from slam_lab.artifacts import atomic_write_json, utc_now

STAGES = (
    ("features", "Load + detect"),
    ("online_tracks", "Track online"),
    ("matches", "Match all pairs"),
    ("geometry", "Verify geometry"),
    ("reconstruction", "Reconstruct"),
)


def _stages(workflow: str):
    return STAGES[:2] if workflow == "tracks" else STAGES


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _tail(path: Path, limit: int = 12000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - limit))
        return file.read().decode(errors="replace")


def _stage_status(stage_id: str, label: str) -> dict:
    return {
        "id": stage_id,
        "label": label,
        "state": "queued",
        "current": 0,
        "total": None,
        "detail": None,
    }


def _update(status_path: Path, status: dict, **values) -> None:
    status.update(values)
    status["revision"] = int(status.get("revision", 0)) + 1
    status["updated_at"] = utc_now()
    atomic_write_json(status_path, status)


def _probe_features(cache_root: Path, total: int) -> dict:
    databases = sorted(cache_root.glob("*/cache.sqlite3"))
    if not databases:
        return {"current": 0, "total": total, "detail": "Opening video"}
    with sqlite3.connect(databases[-1]) as database:
        count = database.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    return {"current": count, "total": total, "detail": f"{count}/{total} frames detected"}


def _probe_matches(path: Path) -> dict:
    from slam_lab.correspondence import match_status

    if not (path / "matches.sqlite3").is_file():
        return {"current": 0, "total": None, "detail": "Preparing match database"}
    current = match_status(path)
    return {
        "current": current.get("completed_pairs", 0),
        "total": current.get("total_pairs"),
        "detail": current.get("phase"),
    }


def _probe_online_tracks(path: Path) -> dict:
    from slam_lab.online_tracker import online_track_status

    if not (path / "tracks.sqlite3").is_file():
        return {"current": 0, "total": None, "detail": "Preparing landmark database"}
    current = online_track_status(path)
    return {
        "current": current.get("processed_frames", 0),
        "total": current.get("total_frames"),
        "detail": (
            f"{current.get('landmarks', 0)} landmarks · "
            f"{current.get('matched_observations', 0)} matched"
        ),
    }


def _probe_geometry(path: Path) -> dict:
    from slam_lab.verification import geometry_status

    if not (path / "geometry.sqlite3").is_file():
        return {"current": 0, "total": None, "detail": "Preparing geometry database"}
    current = geometry_status(path)
    return {
        "current": current.get("completed_pairs", 0),
        "total": current.get("possible_pairs"),
        "detail": current.get("phase"),
    }


def _probe_reconstruction(path: Path) -> dict:
    from slam_lab.solve_jobs import solve_status

    current = solve_status(path)
    return {
        "current": current.get("registered_frames", 0),
        "total": current.get("total_frames"),
        "detail": current.get("phase"),
    }


def _run_stage(
    *,
    status_path: Path,
    status: dict,
    stage_id: str,
    command: list[str],
    workspace: Path,
    log_path: Path,
    probe,
) -> None:
    stage = next(item for item in status["stages"] if item["id"] == stage_id)
    stage.update(state="running", current=0, total=None, detail="Starting")
    _update(status_path, status, state="running", phase=stage_id, command=command)
    with log_path.open("ab", buffering=0) as log:
        log.write(("\n$ " + " ".join(command) + "\n").encode())
        child = subprocess.Popen(
            command,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        _update(
            status_path,
            status,
            child_pid=child.pid,
            child_pgid=os.getpgid(child.pid),
        )
        try:
            while child.poll() is None:
                try:
                    stage.update(probe())
                except Exception:  # noqa: BLE001 - progress probes are best-effort
                    pass
                _update(status_path, status)
                time.sleep(1)
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
            raise
        if child.returncode != 0:
            raise RuntimeError(f"{stage['label']} exited with status {child.returncode}")
    try:
        stage.update(probe())
    except Exception:  # noqa: BLE001 - progress probes are best-effort
        pass
    stage["state"] = "complete"
    stage["detail"] = "Complete"
    if stage.get("total") is not None:
        stage["current"] = stage["total"]
    _update(status_path, status, child_pid=None, child_pgid=None)


def run_pipeline(config_path: Path) -> int:
    config_path = config_path.expanduser().resolve()
    config = _read_json(config_path)
    job_root = config_path.parent
    status_path = job_root / "status.json"
    log_path = job_root / "pipeline.log"
    workspace = Path(config["workspace"])
    run_root = Path(config["run_root"])
    cache_root = run_root / "cache"
    matches = run_root / "matches"
    online_tracks = run_root / "online-tracks"
    geometry = run_root / "geometry"
    reconstruction = run_root / "reconstruction"
    run_root.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir()
    shared_models = workspace / ".slam-cache" / "models"
    if shared_models.is_dir():
        (cache_root / "models").symlink_to(shared_models, target_is_directory=True)

    frames = int(config["frames"])
    stride = int(config["stride"])
    device = config["device"]
    matcher = config["matcher"]
    workflow = config.get("workflow", "full")
    cpu_cli = workspace / ".venv" / "bin" / "slam-lab"
    cuda_cli = workspace / ".venv-cuda" / "bin" / "slam-lab"
    feature_cli = cuda_cli if device == "cuda" else cpu_cli
    match_cli = cuda_cli if device == "cuda" and matcher == "lightglue" else cpu_cli
    status = {
        "schema_version": 1,
        "job_id": config["job_id"],
        "name": config["name"],
        "state": "starting",
        "phase": "starting",
        "revision": 0,
        "created_at": config["created_at"],
        "updated_at": config["created_at"],
        "worker_pid": os.getpid(),
        "child_pid": None,
        "last_error": None,
        "config": {
            **{key: config[key] for key in ("source", "frames", "stride", "matcher", "device")},
            "workflow": workflow,
        },
        "paths": {
            "run": str(run_root),
            "matches": str(matches),
            "online_tracks": str(online_tracks),
            "geometry": str(geometry),
            "reconstruction": str(reconstruction),
            "log": str(log_path),
        },
        "stages": [_stage_status(*stage) for stage in _stages(workflow)],
        "log_tail": "",
    }
    _update(status_path, status)
    try:
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="features",
            command=[
                str(feature_cli),
                "process",
                config["source"],
                "--cache-dir",
                str(cache_root),
                "--stride",
                str(stride),
                "--max-frames",
                str(frames),
                "--device",
                device,
                "--quiet",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_features(cache_root, frames),
        )
        cache_databases = sorted(cache_root.glob("*/cache.sqlite3"))
        if len(cache_databases) != 1:
            raise RuntimeError(f"Expected one feature cache, found {len(cache_databases)}")
        cache = cache_databases[0]
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="online_tracks",
            command=[
                str(feature_cli),
                "track-online",
                str(cache),
                "--output",
                str(online_tracks),
                "--max-frames",
                str(frames),
                "--device",
                device,
                "--quiet",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_online_tracks(online_tracks),
        )
        if workflow == "tracks":
            status["log_tail"] = _tail(log_path)
            _update(status_path, status, state="complete", phase="complete", command=None)
            return 0
        workers = "1" if matcher == "lightglue" else "4"
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="matches",
            command=[
                str(match_cli),
                "match-all",
                str(cache),
                "--output",
                str(matches),
                "--matcher",
                matcher,
                "--max-frames",
                str(frames),
                "--frame-step",
                "1",
                "--workers",
                workers,
                "--device",
                device if matcher == "lightglue" else "cpu",
                "--quiet",
                "--no-rerun",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_matches(matches),
        )
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="geometry",
            command=[
                str(cpu_cli),
                "verify-matches",
                str(matches),
                "--output",
                str(geometry),
                "--frame-step",
                "1",
                "--max-frames",
                str(frames),
                "--quiet",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_geometry(geometry),
        )
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="reconstruction",
            command=[
                str(cpu_cli),
                "solve",
                str(geometry),
                "--output",
                str(reconstruction),
                "--quiet",
                "--no-rerun",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_reconstruction(reconstruction),
        )
        status["log_tail"] = _tail(log_path)
        _update(status_path, status, state="complete", phase="complete", command=None)
        return 0
    except KeyboardInterrupt:
        for stage in status["stages"]:
            if stage["state"] == "running":
                stage["state"] = "canceled"
        status["log_tail"] = _tail(log_path)
        _update(status_path, status, state="canceled", phase="canceled", command=None)
        return 130
    except Exception as error:  # noqa: BLE001 - persisted boundary for supervised jobs
        for stage in status["stages"]:
            if stage["state"] == "running":
                stage["state"] = "failed"
        status["log_tail"] = _tail(log_path)
        _update(
            status_path,
            status,
            state="failed",
            phase="failed",
            command=None,
            last_error={"code": type(error).__name__, "message": str(error)},
        )
        return 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.is_file() or stat.read_text().split()[2] != "Z"
    except (OSError, ValueError):
        return False


def recover_pipeline(config_path: Path) -> int:
    """Reattach supervision after a probe failure left the CLI worker running."""
    config_path = config_path.expanduser().resolve()
    config = _read_json(config_path)
    status_path = config_path.parent / "status.json"
    status = _read_json(status_path)
    workspace = Path(config["workspace"])
    run_root = Path(config["run_root"])
    matches = run_root / "matches"
    geometry = run_root / "geometry"
    reconstruction = run_root / "reconstruction"
    log_path = config_path.parent / "pipeline.log"
    child_pid = status.get("child_pid")
    if not isinstance(child_pid, int) or not _pid_alive(child_pid):
        raise RuntimeError("The matching process is no longer running")
    child_pgid = status.get("child_pgid") or status.get("worker_pid")
    match_stage = next(item for item in status["stages"] if item["id"] == "matches")
    match_stage["state"] = "running"
    match_stage["detail"] = "Reattached to CUDA matcher"
    _update(
        status_path,
        status,
        state="running",
        phase="matches",
        worker_pid=os.getpid(),
        child_pgid=child_pgid,
        last_error=None,
    )
    try:
        while _pid_alive(child_pid):
            try:
                match_stage.update(_probe_matches(matches))
            except Exception:  # noqa: BLE001 - progress probes are best-effort
                pass
            _update(status_path, status)
            time.sleep(1)
        match_state = _probe_matches(matches)
        if not (matches / "manifest.json").is_file():
            raise RuntimeError("The matching process exited without publishing its artifact")
        match_stage.update(match_state, state="complete", detail="Complete")
        if match_stage.get("total") is not None:
            match_stage["current"] = match_stage["total"]
        _update(status_path, status, child_pid=None, child_pgid=None)

        frames = int(config["frames"])
        cpu_cli = workspace / ".venv" / "bin" / "slam-lab"
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="geometry",
            command=[
                str(cpu_cli),
                "verify-matches",
                str(matches),
                "--output",
                str(geometry),
                "--frame-step",
                "1",
                "--max-frames",
                str(frames),
                "--quiet",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_geometry(geometry),
        )
        _run_stage(
            status_path=status_path,
            status=status,
            stage_id="reconstruction",
            command=[
                str(cpu_cli),
                "solve",
                str(geometry),
                "--output",
                str(reconstruction),
                "--quiet",
                "--no-rerun",
            ],
            workspace=workspace,
            log_path=log_path,
            probe=lambda: _probe_reconstruction(reconstruction),
        )
        status["log_tail"] = _tail(log_path)
        _update(status_path, status, state="complete", phase="complete", command=None)
        return 0
    except KeyboardInterrupt:
        for stage in status["stages"]:
            if stage["state"] == "running":
                stage["state"] = "canceled"
        status["log_tail"] = _tail(log_path)
        _update(status_path, status, state="canceled", phase="canceled", command=None)
        return 130
    except Exception as error:  # noqa: BLE001 - persisted boundary for supervised jobs
        for stage in status["stages"]:
            if stage["state"] == "running":
                stage["state"] = "failed"
        status["log_tail"] = _tail(log_path)
        _update(
            status_path,
            status,
            state="failed",
            phase="failed",
            command=None,
            last_error={"code": type(error).__name__, "message": str(error)},
        )
        return 1


class PipelineSupervisor:
    def __init__(self, workspace: Path, recordings: Path):
        self.workspace = workspace.resolve()
        self.recordings = recordings.resolve()
        self.jobs_root = self.recordings / "pipeline-jobs"
        self.runs_root = self.recordings / "pipeline"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        jobs = []
        for status_path in sorted(
            self.jobs_root.glob("*/status.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                status = _read_json(status_path)
                status["log_tail"] = _tail(status_path.parent / "pipeline.log")
                worker_pid = status.get("worker_pid")
                child_pid = status.get("child_pid")
                status["worker_alive"] = isinstance(worker_pid, int) and _pid_alive(worker_pid)
                status["child_alive"] = isinstance(child_pid, int) and _pid_alive(child_pid)
                jobs.append(status)
            except (OSError, json.JSONDecodeError):
                continue
        return jobs

    def start(self, request: dict, allowed_sources: list[str]) -> dict:
        if any(job.get("state") in {"starting", "running"} for job in self.list()):
            raise ValueError("A pipeline job is already running")
        source = str(Path(request.get("source", "")).expanduser().resolve())
        if source not in allowed_sources or not Path(source).is_file():
            raise ValueError("Choose an available source video")
        workflow = request.get("workflow", "full")
        if workflow not in {"tracks", "full"}:
            raise ValueError("Workflow must be tracks or full")
        frames = int(request.get("frames", 300))
        stride = int(request.get("stride", 6))
        if frames < 2:
            raise ValueError("Frames must be at least 2")
        if workflow == "full" and frames > 1000:
            raise ValueError("Full reconstruction is limited to 1000 frames")
        if not 1 <= stride <= 120:
            raise ValueError("Stride must be between 1 and 120")
        matcher = request.get("matcher", "lightglue")
        if matcher not in {"lightglue", "cosine"}:
            raise ValueError("Matcher must be lightglue or cosine")
        device = request.get("device", "cuda")
        if device not in {"cuda", "cpu"}:
            raise ValueError("Device must be cuda or cpu")
        if device == "cuda" and not (self.workspace / ".venv-cuda" / "bin" / "slam-lab").is_file():
            raise ValueError("CUDA environment is not available")
        raw_name = str(request.get("name", "osaka-300")).strip()
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_name).strip("-")[:48]
        if not name:
            raise ValueError("Run name is empty")
        job_id = f"pipeline-{uuid.uuid4()}"
        short_id = job_id.rsplit("-", 1)[-1][:8]
        job_root = self.jobs_root / job_id
        run_root = self.runs_root / f"{name}-{short_id}"
        created_at = utc_now()
        job_root.mkdir(parents=True, exist_ok=False)
        config = {
            "schema_version": 1,
            "job_id": job_id,
            "name": name,
            "source": source,
            "frames": frames,
            "stride": stride,
            "matcher": matcher,
            "device": device,
            "workflow": workflow,
            "workspace": str(self.workspace),
            "run_root": str(run_root),
            "created_at": created_at,
        }
        config_path = job_root / "job.json"
        atomic_write_json(config_path, config)
        starting = {
            "schema_version": 1,
            "job_id": job_id,
            "name": name,
            "state": "starting",
            "phase": "starting",
            "revision": 1,
            "updated_at": created_at,
            "worker_pid": None,
            "child_pid": None,
            "last_error": None,
            "config": {
                key: config[key]
                for key in ("source", "frames", "stride", "matcher", "device", "workflow")
            },
            "paths": {"run": str(run_root), "log": str(job_root / "pipeline.log")},
            "stages": [_stage_status(*stage) for stage in _stages(workflow)],
            "log_tail": "",
        }
        atomic_write_json(job_root / "status.json", starting)
        with (job_root / "supervisor.log").open("ab") as log:
            child = subprocess.Popen(
                [
                    str(self.workspace / ".venv" / "bin" / "python"),
                    "-m",
                    "slam_lab.workbench_pipeline",
                    str(config_path),
                ],
                cwd=self.workspace,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        starting["worker_pid"] = child.pid
        return starting

    def cancel(self, job_id: str) -> dict:
        if not re.fullmatch(r"pipeline-[0-9a-f-]+", job_id):
            raise ValueError("Unknown pipeline job")
        status_path = self.jobs_root / job_id / "status.json"
        if not status_path.is_file():
            raise ValueError("Unknown pipeline job")
        status = _read_json(status_path)
        worker_pid = status.get("worker_pid")
        child_pid = status.get("child_pid")
        worker_alive = isinstance(worker_pid, int) and _pid_alive(worker_pid)
        child_alive = isinstance(child_pid, int) and _pid_alive(child_pid)
        if status.get("state") not in {"starting", "running"} and not (worker_alive or child_alive):
            return status
        if not worker_alive and not child_alive:
            raise ValueError("Pipeline worker has no process ID")
        groups = set()
        if worker_alive:
            groups.add(worker_pid)
        child_group = status.get("child_pgid")
        if child_alive and isinstance(child_group, int) and child_group > 0:
            groups.add(child_group)
        elif child_alive:
            groups.add(child_pid)
        for group in groups:
            try:
                os.killpg(group, signal.SIGINT)
            except ProcessLookupError:
                pass
        for stage in status.get("stages", []):
            if stage.get("state") in {"running", "failed"}:
                stage["state"] = "canceled"
                stage["detail"] = "Stop requested; committed work retained"
        _update(
            status_path,
            status,
            state="canceled",
            phase="canceled",
            child_pid=None,
            child_pgid=None,
            last_error=None,
        )
        return status


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("config", type=Path)
    result.add_argument("--recover", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return recover_pipeline(args.config) if args.recover else run_pipeline(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
