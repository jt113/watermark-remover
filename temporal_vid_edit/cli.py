from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .deep_inpaint import DeepInpainter, DeepModelConfig
from .frames import extract_frames
from .inpaint import TemporalInpainter, TemporalModelConfig
from .overlays import load_overlays


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal video overlay removal (text or watermark)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Split a video into labeled frames")
    extract_parser.add_argument("video", type=Path, help="Input video path")
    extract_parser.add_argument("output", type=Path, help="Directory to store frames")
    extract_parser.add_argument("--prefix", default="frame", help="Filename prefix for frames")
    extract_parser.add_argument("--image-format", default="png", help="Image format for frames")

    inpaint_parser = subparsers.add_parser(
        "inpaint", help="Remove overlays using temporal, flow-aligned, or watermark-focused models"
    )
    inpaint_parser.add_argument("frames", type=Path, help="Directory containing extracted frames")
    inpaint_parser.add_argument("overlays", type=Path, help="JSON file describing overlay regions")
    inpaint_parser.add_argument(
        "--window", type=int, default=2, help="Number of neighboring frames on each side"
    )
    inpaint_parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable spatial inpaint fallback when neighbors are unavailable",
    )
    inpaint_parser.add_argument(
        "--output-video",
        type=Path,
        help="Write the edited frames into a video file",
    )
    inpaint_parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec to use when writing the output video",
    )
    inpaint_parser.add_argument(
        "--output-frames",
        type=Path,
        help="Directory where cleaned frames should be written",
    )
    inpaint_parser.add_argument(
        "--method",
        choices=["temporal", "flow", "deep", "delogo"],
        default="temporal",
        help="Inpainting backend: temporal median, optical-flow aggregation, deep model, or ffmpeg delogo",
    )
    inpaint_parser.add_argument(
        "--flow-preset",
        default="medium",
        help="DIS optical flow preset to use with --method flow (ultrafast, fast, medium, fine)",
    )
    inpaint_parser.add_argument(
        "--no-seamless-clone",
        action="store_true",
        help="Disable Poisson blending for --method flow",
    )
    inpaint_parser.add_argument(
        "--no-temporal-median",
        action="store_true",
        help="Use weighted averaging instead of temporal median for --method flow",
    )
    inpaint_parser.add_argument(
        "--no-flow-fallback",
        action="store_true",
        help="Disable OpenCV inpaint fallback for --method flow",
    )
    inpaint_parser.add_argument(
        "--flow-max-sources",
        type=int,
        default=6,
        help="Maximum number of clean source frames to aggregate for --method flow",
    )
    inpaint_parser.add_argument(
        "--flow-max-distance",
        type=int,
        default=120,
        help="Maximum frame distance when searching for clean sources in --method flow",
    )
    inpaint_parser.add_argument(
        "--deep-cmd",
        nargs="+",
        help="Command used to invoke the deep inpainting model (e.g. python path/to/infer.py)",
    )
    inpaint_parser.add_argument(
        "--deep-weights",
        type=Path,
        help="Optional weights file passed to the deep model",
    )
    inpaint_parser.add_argument(
        "--deep-extra-arg",
        action="append",
        default=[],
        help="Extra arguments appended to the deep model command",
    )
    inpaint_parser.add_argument(
        "--deep-context",
        type=int,
        default=3,
        help="Number of context frames before/after an overlay run for --method deep",
    )
    inpaint_parser.add_argument(
        "--deep-keep-temp",
        action="store_true",
        help="Keep intermediate clips generated for the deep model",
    )
    inpaint_parser.add_argument(
        "--deep-video-arg",
        default="--video",
        help="Flag name used by the deep model command to specify the input video",
    )
    inpaint_parser.add_argument(
        "--deep-mask-arg",
        default="--mask",
        help="Flag name used by the deep model command to specify the mask video",
    )
    inpaint_parser.add_argument(
        "--deep-output-arg",
        default="--output",
        help="Flag name used by the deep model command to specify the output video",
    )
    inpaint_parser.add_argument(
        "--deep-weights-arg",
        default="--weights",
        help="Flag name used by the deep model command to specify weights",
    )
    inpaint_parser.add_argument(
        "--deep-workdir",
        type=Path,
        help="Working directory to run the deep model command in",
    )

    return parser


def run_extract(args: argparse.Namespace) -> None:
    metadata = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        prefix=args.prefix,
        image_format=args.image_format,
    )
    frame_count = len(metadata.get("frames", []))
    print(f"Extracted {frame_count} frames to {args.output}")


def run_inpaint(args: argparse.Namespace) -> None:
    overlays = load_overlays(args.overlays)
    if not overlays:
        print("No overlay regions provided; nothing to do")
        return

    if args.method == "delogo":
        if args.output_frames:
            raise SystemExit("--output-frames is not supported with --method delogo")
        if not args.output_video:
            raise SystemExit("--method delogo requires --output-video to be set")

        from .ffmpeg_delogo import run_delogo_pipeline
        codec = args.codec
        if codec.lower() == "mp4v":
            codec = "libx264"
        run_delogo_pipeline(args.frames, overlays, args.output_video, codec=codec)
        print(f"Wrote cleaned video to {args.output_video}")
        return

    if args.method == "deep":
        if not args.deep_cmd:
            raise SystemExit("--deep-cmd is required when --method deep is selected")

        deep_config = DeepModelConfig(
            command=args.deep_cmd,
            weights=args.deep_weights,
            extra_args=args.deep_extra_arg or [],
            context=max(0, args.deep_context),
            codec=args.codec,
            keep_temp=bool(args.deep_keep_temp),
            video_arg=args.deep_video_arg,
            mask_arg=args.deep_mask_arg,
            output_arg=args.deep_output_arg,
            weights_arg=args.deep_weights_arg,
            workdir=args.deep_workdir,
        )
        inpainter = DeepInpainter(args.frames, deep_config)
        inpainter.inpaint(overlays)

        wrote_output = False
        if args.output_frames:
            inpainter.save_frames(args.output_frames)
            print(f"Wrote cleaned frames to {args.output_frames}")
            wrote_output = True
        if args.output_video:
            inpainter.write_video(args.output_video, codec=args.codec)
            print(f"Wrote cleaned video to {args.output_video}")
            wrote_output = True
        if not wrote_output:
            raise SystemExit(
                "Specify --output-video and/or --output-frames for the inpaint command"
            )
        return

    config = TemporalModelConfig(window=max(1, args.window), use_inpaint_fallback=not args.no_fallback)
    if args.method == "flow":
        from .flow_inpaint import FlowAlignedInpainter, FlowInpainterConfig

        flow_config = FlowInpainterConfig(
            window=max(1, args.window),
            flow_preset=args.flow_preset,
            use_seamless_clone=not args.no_seamless_clone,
            temporal_median=not args.no_temporal_median,
            fall_back_temporal=not args.no_flow_fallback,
            max_sources=max(1, args.flow_max_sources),
            max_search_distance=max(1, args.flow_max_distance),
        )
        inpainter = FlowAlignedInpainter(args.frames, config=flow_config)
        inpainter.inpaint(overlays)

        wrote_output = False
        if args.output_frames:
            inpainter.save_frames(args.output_frames)
            print(f"Wrote cleaned frames to {args.output_frames}")
            wrote_output = True
        if args.output_video:
            inpainter.write_video(args.output_video, codec=args.codec)
            print(f"Wrote cleaned video to {args.output_video}")
            wrote_output = True
        if not wrote_output:
            raise SystemExit(
                "Specify --output-video and/or --output-frames for the inpaint command"
            )
        return

    inpainter = TemporalInpainter(args.frames, config=config)
    inpainter.inpaint(overlays)

    wrote_output = False

    if args.output_frames:
        inpainter.save_frames(args.output_frames)
        print(f"Wrote cleaned frames to {args.output_frames}")
        wrote_output = True

    if args.output_video:
        inpainter.write_video(args.output_video, codec=args.codec)
        print(f"Wrote cleaned video to {args.output_video}")
        wrote_output = True

    if not wrote_output:
        raise SystemExit(
            "Specify --output-video and/or --output-frames for the inpaint command"
        )


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        run_extract(args)
    elif args.command == "inpaint":
        run_inpaint(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
