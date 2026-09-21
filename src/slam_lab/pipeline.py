"""Content-addressed extraction with frame-level resume."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from slam_lab.cache import CachedFrame, FrameCache
from slam_lab.config import SCHEMA_VERSION, ExtractionConfig, fingerprint
from slam_lab.datasets import hash_source, open_source, source_state


@dataclass
class ProcessResult:
    path: Path
    new_frames: int
    cached_frames: int
    complete: bool


def process_video(
    source: Path,
    cache_dir: Path,
    config: ExtractionConfig,
    extractor,
    *,
    max_frames: int | None = None,
    progress: bool = True,
) -> ProcessResult:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be at least 1")
    source = source.resolve()
    initial_state = source_state(source)
    source_hash = hash_source(source)
    identity = config.identity()
    run_id = fingerprint({"source_sha256": source_hash, "extraction": identity})
    path = cache_dir / run_id / "cache.sqlite3"
    with FrameCache(path, create=True) as cache:
        if cache.get("complete", False):
            return ProcessResult(path, 0, cache.count(), True)
        if cache.get("run_id") is None:
            cache.set(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                source_path=str(source),
                source_sha256=source_hash,
                identity=identity,
                created_at=datetime.now(UTC).isoformat(),
                complete=False,
            )
        new_frames = 0
        cached_frames = 0
        limited = False
        cache.set(selection_complete=False, selection_max_frames=max_frames)
        with open_source(source) as reader:
            cache.set(video=reader.metadata())
            total = reader.total_frames
            total = (total + config.stride - 1) // config.stride if total else None
            if max_frames is not None:
                total = min(total, max_frames) if total is not None else max_frames
            with tqdm(total=total, desc=source.name, unit="frame", disable=not progress) as bar:
                for frame in reader.frames(stride=config.stride, max_side=config.max_side):
                    if max_frames is not None and frame.index // config.stride >= max_frames:
                        limited = True
                        break
                    if cache.contains(frame.index):
                        cached_frames += 1
                    else:
                        features = extractor(frame.image)
                        cache.put(
                            CachedFrame(
                                index=frame.index,
                                timestamp_ns=frame.timestamp_ns,
                                pts=frame.pts,
                                width=frame.image.shape[1],
                                height=frame.image.shape[0],
                                original_width=frame.original_width,
                                original_height=frame.original_height,
                                **features,
                            )
                        )
                        new_frames += 1
                        if new_frames == 1:
                            cache.set(extractor=getattr(extractor, "provenance", {}))
                    bar.update(1)
                    bar.set_postfix(new=new_frames, reused=cached_frames, refresh=False)
        if cache.count() == 0:
            raise ValueError(f"No frames could be decoded from {source}")
        if initial_state != source_state(source):
            # The original content hash no longer identifies these decoded frames.
            with cache.db:
                cache.db.execute("DELETE FROM frames")
            raise RuntimeError(
                f"Input changed during processing; retry after copying finishes: {source}"
            )
        cache.set(
            complete=not limited,
            selection_complete=True,
            selection_max_frames=max_frames,
            updated_at=datetime.now(UTC).isoformat(),
        )
        return ProcessResult(path, new_frames, cached_frames, not limited)
