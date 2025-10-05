from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    prefix: str = "frame",
    image_format: str = "png",
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    frames: List[Dict[str, object]] = []
    index = 0
    success, frame = capture.read()
    while success:
        filename = f"{prefix}_{index:06d}.{image_format}"
        frame_path = output_dir / filename
        cv2.imwrite(str(frame_path), frame)

        frames.append(
            {
                "index": index,
                "path": filename,
                "timestamp_sec": index / fps,
            }
        )

        index += 1
        success, frame = capture.read()

    capture.release()

    metadata = {
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "frame_count": index,
        "image_format": image_format,
        "prefix": prefix,
        "frames": frames,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def load_metadata(frames_dir: Path) -> Dict[str, object]:
    frames_dir = Path(frames_dir)
    metadata_path = frames_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {frames_dir}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return data


def resolve_frame_path(frames_dir: Path, frame_entry: Dict[str, object]) -> Path:
    return Path(frames_dir) / Path(frame_entry["path"])  # type: ignore[index]


def frame_dimensions(frames_dir: Path, metadata: Dict[str, object]) -> Tuple[int, int]:
    frames = metadata.get("frames")
    if not frames:
        raise ValueError("No frames found in metadata")

    first_entry = frames[0]
    frame_path = resolve_frame_path(frames_dir, first_entry)
    image = cv2.imread(str(frame_path))
    if image is None:
        raise RuntimeError(f"Unable to read frame: {frame_path}")

    height, width = image.shape[:2]
    return width, height
