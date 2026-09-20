"""Commands for processing, listing, and viewing local recordings."""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from slam_lab import __version__
from slam_lab.config import ExtractionConfig


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="slam-lab", description=__doc__)
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    process = commands.add_parser("process", help="Extract and cache video features")
    process.add_argument("input", type=Path, help="Video, TUM sequence, or folder of inputs")
    process.add_argument("--cache-dir", type=Path, default=Path(".slam-cache"))
    process.add_argument(
        "--recursive", action="store_true", help="Include videos in subdirectories"
    )
    process.add_argument(
        "--stride", type=positive_int, default=1, help="Keep every Nth frame (default: 1)"
    )
    process.add_argument(
        "--max-side", type=positive_int, default=1024, help="Longest edge in pixels"
    )
    process.add_argument("--max-keypoints", type=positive_int, default=2048)
    process.add_argument(
        "--threshold", type=float, default=0.0005, help="SuperPoint detection threshold"
    )
    process.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    process.add_argument(
        "--max-frames",
        type=positive_int,
        help="Limit to N selected frames per video; omit later to resume the full video",
    )
    process.add_argument("--quiet", action="store_true", help="Hide progress bars")
    listing = commands.add_parser("list", help="List cached videos and extraction runs")
    listing.add_argument("--cache-dir", type=Path, default=Path(".slam-cache"))
    view = commands.add_parser("view", help="Open a cached run in Rerun, without inference")
    view.add_argument(
        "cache", type=Path, help="Cache directory or cache.sqlite3 (from process/list)"
    )
    view.add_argument(
        "--save", type=Path, help="Write an .rrd recording instead of launching a viewer"
    )
    view.add_argument(
        "--min-score", type=float, default=0.0, help="Filter displayed keypoints only"
    )
    view.add_argument(
        "--point-radius", type=float, default=2.0, help="Point radius in image pixels"
    )
    reconstruct = commands.add_parser(
        "reconstruct", help="Solve monocular camera poses and sparse 3D"
    )
    reconstruct.add_argument("cache", type=Path)
    reconstruct.add_argument("--output", type=Path, required=True, help="New output directory")
    reconstruct.add_argument("--max-frames", type=positive_int, default=300)
    reconstruct.add_argument(
        "--frame-step", type=positive_int, default=1, help="Use every Nth cached frame"
    )
    reconstruct.add_argument(
        "--fov-deg", type=float, default=60.0, help="Assumed horizontal FOV; overridden by --fx"
    )
    for parameter in ("fx", "fy", "cx", "cy"):
        reconstruct.add_argument(
            f"--{parameter}", type=float, help="Fixed intrinsic in CACHED preview pixels"
        )
    reconstruct.add_argument("--ratio", type=float, default=0.8)
    reconstruct.add_argument("--ransac-px", type=float, default=1.5)
    reconstruct.add_argument("--reprojection-px", type=float, default=3.0)
    reconstruct.add_argument(
        "--min-parallax", type=float, default=1.0, help="Triangulation angle in degrees"
    )
    reconstruct.add_argument("--min-track-length", type=positive_int, default=3)
    reconstruct.add_argument(
        "--bundle-evaluations", type=int, default=30, help="0 disables bundle adjustment"
    )
    reconstruct.add_argument("--seed", type=int, default=7)
    reconstruct.add_argument("--quiet", action="store_true")
    reconstruct.add_argument("--no-rerun", action="store_true", help="Skip automatic Rerun export")
    reconstruction_view = commands.add_parser(
        "view-reconstruction", help="View solved poses and points in Rerun"
    )
    reconstruction_view.add_argument("reconstruction", type=Path)
    reconstruction_view.add_argument("--save", type=Path)
    reconstruction_view.add_argument("--cache", type=Path, help="Override feature-cache location")
    matching = commands.add_parser("match-all", help="Match every pair of cached frames; resumable")
    matching.add_argument("cache", type=Path)
    matching.add_argument("--output", type=Path, required=True)
    matching.add_argument("--matcher", choices=["lightglue", "nn", "cosine"], default="lightglue")
    matching.add_argument("--max-frames", type=positive_int, default=300)
    matching.add_argument("--frame-step", type=positive_int, default=1)
    matching.add_argument("--workers", type=positive_int, default=4)
    matching.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    matching.add_argument(
        "--allow-runtime-change",
        action="store_true",
        help="Reuse existing LightGlue pairs across devices/PyTorch builds; record each runtime",
    )
    matching.add_argument("--threshold", type=float, default=0.1, help="LightGlue match threshold")
    matching.add_argument("--ratio", type=float, default=0.8, help="NN distance ratio threshold")
    matching.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.8,
        help="Minimum cosine similarity for mutual best matches",
    )
    matching.add_argument(
        "--resume-after",
        type=Path,
        help="Resume a saved matching run after this run finishes successfully",
    )
    matching.add_argument("--quiet", action="store_true")
    matching.add_argument("--no-rerun", action="store_true", help="Skip automatic Rerun export")
    matching.add_argument(
        "--background", action="store_true", help="Detach and write progress to a log"
    )
    status = commands.add_parser("match-status", help="Read saved matching progress")
    status.add_argument("run", type=Path)
    matches_view = commands.add_parser("view-matches", help="View matches/tracks in Rerun")
    matches_view.add_argument("run", type=Path)
    matches_view.add_argument(
        "--pair",
        type=int,
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="Zero-based selected-frame rows, e.g. 100 101 around 10 s",
    )
    matches_view.add_argument(
        "--max-links",
        type=int,
        help="Limit displayed lines to the strongest N (default: all); 0 hides lines",
    )
    matches_view.add_argument("--min-track-length", type=positive_int, default=2)
    matches_view.add_argument("--save", type=Path)
    verify = commands.add_parser(
        "verify-matches", help="Cache pairwise F/E/H RANSAC and verified tracks"
    )
    verify.add_argument("run", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--frame-step", type=positive_int, default=1)
    verify.add_argument("--max-frames", type=positive_int)
    verify.add_argument("--ransac-px", type=float, default=1.5)
    verify.add_argument("--reprojection-px", type=float, default=3.0)
    verify.add_argument("--min-parallax", type=float, default=2.0)
    verify.add_argument("--min-inliers", type=positive_int, default=30)
    verify.add_argument("--fov-deg", type=float, default=60.0)
    for parameter in ("fx", "fy", "cx", "cy"):
        verify.add_argument(f"--{parameter}", type=float, help="Fixed intrinsic in cached pixels")
    verify.add_argument("--quiet", action="store_true")
    geometry_view = commands.add_parser(
        "view-geometry", help="Inspect saved F/E/H and seed diagnostics"
    )
    geometry_view.add_argument("run", type=Path)
    geometry_view.add_argument("--pair", type=int, nargs=2)
    geometry_view.add_argument("--save", type=Path)
    geometry_status = commands.add_parser("geometry-status", help="Read geometry progress as JSON")
    geometry_status.add_argument("run", type=Path)
    solve = commands.add_parser("solve", help="Reconstruct cameras and points from verified tracks")
    solve.add_argument("geometry", type=Path)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--bundle-evaluations", type=int, default=30)
    solve.add_argument("--min-track-length", type=positive_int, default=3)
    solve.add_argument("--reprojection-px", type=float, default=3.0)
    solve.add_argument("--min-parallax", type=float, default=1.0)
    solve.add_argument("--quiet", action="store_true")
    solve.add_argument("--no-rerun", action="store_true", help="Skip automatic Rerun export")
    return root


def process_command(args) -> int:
    from slam_lab.extractor import SuperPointExtractor
    from slam_lab.pipeline import process_video
    from slam_lab.video import discover_sources

    config = ExtractionConfig(
        stride=args.stride,
        max_side=args.max_side,
        max_keypoints=args.max_keypoints,
        detection_threshold=args.threshold,
    )
    cache_dir = args.cache_dir.expanduser().resolve()
    videos = discover_sources(args.input, recursive=args.recursive)
    extractor = SuperPointExtractor(config, device=args.device, model_dir=cache_dir / "models")
    failures = 0
    for video in videos:
        try:
            result = process_video(
                video,
                cache_dir,
                config,
                extractor,
                max_frames=args.max_frames,
                progress=not args.quiet,
            )
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            failures += 1
            print(f"Error processing {video}: {error}", file=sys.stderr)
            continue
        status = (
            "complete" if result.complete else "partial; rerun without --max-frames to continue"
        )
        print(f"{video.name}: {result.new_frames} new, {result.cached_frames} reused ({status})")
        print(f"  {result.path}")
    return 1 if failures else 0


def list_command(args) -> int:
    from slam_lab.cache import FrameCache

    paths = sorted(args.cache_dir.expanduser().glob("*/cache.sqlite3"))
    if not paths:
        print("No cached videos yet. Run: slam-lab process /path/to/video-or-folder")
    for path in paths:
        try:
            with FrameCache(path) as cache:
                state = "complete" if cache.get("complete") else "partial"
                source = Path(cache.get("source_path", "unknown")).name
                print(f"{source}: {cache.count()} frames ({state})")
                print(f"  {path.resolve()}")
        except (ValueError, sqlite3.Error) as error:
            print(f"Cannot read {path}: {error}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(raw_args)
    try:
        if args.command == "geometry-status":
            from slam_lab.verification import geometry_status

            print(json.dumps(geometry_status(args.run), indent=2))
            return 0
        if args.command == "verify-matches":
            from slam_lab.verification import VerificationConfig, verify_matches

            summary = verify_matches(
                args.run,
                args.output,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                config=VerificationConfig(
                    threshold_px=args.ransac_px,
                    reprojection_px=args.reprojection_px,
                    min_inliers=args.min_inliers,
                    min_parallax_deg=args.min_parallax,
                ),
                camera_options={
                    key: getattr(args, key) for key in ("fx", "fy", "cx", "cy", "fov_deg")
                },
                progress=not args.quiet,
            )
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "view-geometry":
            from slam_lab.geometry_view import view_geometry

            pair = view_geometry(args.run, pair=args.pair, output=args.save)
            print(f"{'Saved' if args.save else 'Displayed'} geometry for match rows {pair}")
            return 0
        if args.command == "solve":
            from slam_lab.reconstruction import ReconstructionConfig
            from slam_lab.reconstruction_io import view_reconstruction
            from slam_lab.solver import solve_geometry

            result = solve_geometry(
                args.geometry,
                args.output,
                progress=not args.quiet,
                config=ReconstructionConfig(
                    bundle_evaluations=args.bundle_evaluations,
                    min_track_length=args.min_track_length,
                    reprojection_threshold=args.reprojection_px,
                    min_parallax=args.min_parallax,
                ),
            )
            if not args.no_rerun:
                recording = result / "reconstruction.rrd"
                view_reconstruction(result, output=recording)
            print(f"Saved reconstruction to {result}")
            return 0
        if args.command == "match-status":
            from slam_lab.correspondence import match_status

            print(json.dumps(match_status(args.run), indent=2))
            return 0
        if args.command == "match-all":
            from slam_lab.correspondence import match_all, match_status

            if args.resume_after is not None:
                from slam_lab.match_jobs import resume_arguments

                if args.resume_after.expanduser().resolve() == args.output.expanduser().resolve():
                    raise ValueError("resume-after must refer to a different matching run")
                resume_arguments(args.resume_after)  # Validate the saved run before starting.
            if args.background:
                output = args.output.expanduser().resolve()
                if match_status(output)["running"]:
                    print(f"Already running: {output}")
                    return 0
                output.parent.mkdir(parents=True, exist_ok=True)
                log_path = output.with_suffix(".log")
                child_args = [arg for arg in raw_args if arg != "--background"]
                if "--quiet" not in child_args:
                    child_args.append("--quiet")
                with log_path.open("ab") as log:
                    child = subprocess.Popen(
                        [sys.executable, "-u", "-m", "slam_lab", *child_args],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                output.with_suffix(".job.json").write_text(
                    json.dumps(
                        {"pid": child.pid, "log": str(log_path), "arguments": child_args}, indent=2
                    )
                    + "\n"
                )
                print(
                    f"Started background job PID {child.pid}\nLog: {log_path}\n"
                    f"Status: slam-lab match-status {output}"
                )
                return 0
            match_all(
                args.cache,
                args.output,
                matcher_name=args.matcher,
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                workers=args.workers,
                device=args.device,
                threshold=args.threshold,
                cosine_threshold=args.cosine_threshold,
                ratio=args.ratio,
                progress=not args.quiet,
                export=not args.no_rerun,
                resume_after=args.resume_after,
                allow_runtime_change=args.allow_runtime_change,
            )
            if args.resume_after is not None:
                from slam_lab.match_jobs import resume_saved_run

                resume_saved_run(args.resume_after)
            return 0
        if args.command == "view-matches":
            from slam_lab.match_view import view_matches

            count = view_matches(
                args.run,
                output=args.save,
                pair=args.pair,
                max_links=args.max_links,
                min_track_length=args.min_track_length,
            )
            print(f"{'Saved' if args.save else 'Displayed'} {count} correspondence frames")
            return 0
        if args.command == "process":
            return process_command(args)
        if args.command == "list":
            return list_command(args)
        if args.command == "reconstruct":
            from slam_lab.reconstruction import ReconstructionConfig, reconstruct
            from slam_lab.reconstruction_io import view_reconstruction

            config = ReconstructionConfig(
                ratio=args.ratio,
                essential_threshold=args.ransac_px,
                reprojection_threshold=args.reprojection_px,
                min_parallax=args.min_parallax,
                min_track_length=args.min_track_length,
                bundle_evaluations=args.bundle_evaluations,
                seed=args.seed,
            )
            result = reconstruct(
                args.cache,
                args.output,
                config=config,
                camera_options={
                    key: getattr(args, key) for key in ("fx", "fy", "cx", "cy", "fov_deg")
                },
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                progress=not args.quiet,
            )
            if not args.no_rerun:
                recording = result / "reconstruction.rrd"
                view_reconstruction(result, output=recording)
            print(f"Saved reconstruction to {result}")
            return 0
        if args.command == "view-reconstruction":
            from slam_lab.reconstruction_io import view_reconstruction

            count = view_reconstruction(
                args.reconstruction, output=args.save, cache_path=args.cache
            )
            print(f"{'Saved' if args.save else 'Displayed'} {count} reconstruction frames")
            return 0
        from slam_lab.viewer import view_cache

        count = view_cache(
            args.cache, output=args.save, min_score=args.min_score, point_radius=args.point_radius
        )
        print(
            f"{'Saved' if args.save else 'Displayed'} {count} frames"
            + (f" to {args.save}" if args.save else " in Rerun")
        )
        return 0
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Features and completed pairs remain cached; "
            "rerun the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError, sqlite3.Error, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
