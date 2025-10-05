from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .frames import load_metadata
from .overlays import OverlayRegion


def _group_frames(frames: Sequence[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    sorted_frames = sorted(set(frames))
    grouped: List[Tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for frame in sorted_frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        grouped.append((start, prev))
        start = prev = frame
    grouped.append((start, prev))
    return grouped


def _build_filter_entries(overlays: Iterable[OverlayRegion]) -> List[str]:
    by_box: defaultdict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for region in overlays:
        width = region.x2 - region.x1
        height = region.y2 - region.y1
        if width <= 0 or height <= 0:
            continue
        key = (region.x1, region.y1, width, height)
        by_box[key].append(region.frame)

    filters: List[str] = []
    for (x, y, w, h), frames in by_box.items():
        for start, end in _group_frames(frames):
            enable = f"between(n,{start},{end})"
            filters.append(
                f"delogo=x={x}:y={y}:w={w}:h={h}:show=0:enable='{enable}'"
            )
    return filters


def run_delogo_pipeline(
    frames_dir: Path | str,
    overlays: Sequence[OverlayRegion],
    output_path: Path | str,
    *,
    codec: str = "libx264",
    extra_ffmpeg_args: Sequence[str] | None = None,
) -> None:
    frames_dir = Path(frames_dir)
    metadata = load_metadata(frames_dir)
    video_path = metadata.get("video_path")
    if not video_path:
        raise SystemExit("metadata.json must include 'video_path' for delogo pipeline")

    video_source = Path(video_path)
    if not video_source.exists():
        raise SystemExit(f"Original video not found at {video_source}")

    filter_entries = _build_filter_entries(overlays)
    if not filter_entries:
        raise SystemExit("No valid overlay regions to process")

    filter_complex = ",".join(filter_entries)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_source),
        "-vf",
        filter_complex,
        "-c:v",
        codec,
        "-c:a",
        "copy",
        str(output_path),
    ]

    if extra_ffmpeg_args:
        cmd[1:1] = list(extra_ffmpeg_args)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg is required for --method delogo but was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg command failed with exit code {exc.returncode}") from exc
