"""Artifact envelopes and atomic JSON helpers shared by backend stages."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from slam_lab.config import hash_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def opaque_id(kind: str) -> str:
    return f"{kind}-{uuid4()}"


def atomic_write_json(path: Path, value: dict) -> None:
    """Replace a JSON document atomically without exposing a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def payload_inventory(root: Path, names: list[str]) -> list[dict]:
    payloads = []
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise ValueError(f"Missing artifact payload: {name}")
        payloads.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": hash_file(path)}
        )
    return payloads


def write_manifest(
    root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    parent_artifact_id: str | None,
    producer_job_id: str | None,
    payloads: list[str],
    capabilities: list[str],
    origin: dict | None = None,
) -> dict:
    manifest = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "parent_artifact_id": parent_artifact_id,
        "producer_job_id": producer_job_id,
        "state": "complete",
        "revision": 1,
        "updated_at": utc_now(),
        "last_error": None,
        "capabilities": sorted(capabilities),
        "payloads": payload_inventory(root, payloads),
    }
    if origin is not None:
        manifest["origin"] = origin
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def read_manifest(root: Path) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing artifact manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete":
        raise ValueError(f"Unsupported or incomplete artifact manifest: {path}")
    return manifest


def validate_manifest(
    root: Path, manifest: dict | None = None, *, artifact_type: str | None = None
) -> dict:
    root = Path(root)
    manifest = read_manifest(root) if manifest is None else manifest
    if artifact_type is not None and manifest.get("artifact_type") != artifact_type:
        raise ValueError(
            f"Expected {artifact_type} artifact, found {manifest.get('artifact_type')!r}: {root}"
        )
    for payload in manifest.get("payloads", []):
        path = root / payload["path"]
        if not path.is_file():
            raise ValueError(f"Missing artifact payload: {payload['path']}")
        if path.stat().st_size != payload["size_bytes"] or hash_file(path) != payload["sha256"]:
            raise ValueError(f"Artifact payload does not match manifest: {payload['path']}")
    return manifest
