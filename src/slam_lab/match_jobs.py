"""Resume matching from stored provenance rather than guessing prior settings."""

from pathlib import Path

from slam_lab.cache import resolve_cache
from slam_lab.correspondence import MatchStore


def resume_arguments(path):
    path = Path(path).expanduser().resolve()
    with MatchStore(path / "matches.sqlite3") as store:
        identity = store.get("identity")
        matcher = store.get("active_matcher", identity["matcher"])
        cache = resolve_cache(Path(store.get("feature_cache")))
        args = [
            "match-all",
            str(cache),
            "--output",
            str(path),
            "--max-frames",
            str(identity["max_frames"]),
            "--frame-step",
            str(identity["frame_step"]),
            "--workers",
            str(store.get("workers", 4)),
            "--quiet",
        ]
        if matcher["name"] == "lightglue-superpoint":
            args += [
                "--matcher",
                "lightglue",
                "--threshold",
                str(matcher["filter_threshold"]),
                "--device",
                matcher["device"],
            ]
            if matcher != identity["matcher"]:
                args += ["--allow-runtime-change"]
        elif matcher["name"] == "mutual-nearest-neighbor":
            args += ["--matcher", "nn", "--ratio", str(matcher["ratio"])]
        elif matcher["name"] == "cosine-mutual-nearest-neighbor":
            args += ["--matcher", "cosine", "--cosine-threshold", str(matcher["min_similarity"])]
        else:
            raise ValueError(f"Cannot resume unknown matcher: {matcher['name']}")
    return args


def resume_saved_run(path):
    from slam_lab.cli import main

    print(f"Matching and export finished; resuming {path}", flush=True)
    if main([*resume_arguments(path), "--background"]) != 0:
        raise RuntimeError(f"Could not resume matching run: {path}")
