from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .frames import load_metadata, resolve_frame_path
from .overlays import OverlayRegion, group_by_frame


@dataclass
class TemporalModelConfig:
    window: int = 2
    use_inpaint_fallback: bool = True


class TemporalInpainter:
    def __init__(self, frames_dir: Path, config: Optional[TemporalModelConfig] = None) -> None:
        self.frames_dir = Path(frames_dir)
        self.config = config or TemporalModelConfig()
        self.metadata = load_metadata(self.frames_dir)
        self.frames = self._load_frames()
        self.frames_original = [frame.copy() for frame in self.frames]

    def _load_frames(self) -> List[np.ndarray]:
        entries: Sequence[Dict[str, object]] = self.metadata["frames"]  # type: ignore[index]
        loaded: List[np.ndarray] = []
        for entry in entries:
            frame_path = resolve_frame_path(self.frames_dir, entry)
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Unable to read frame {frame_path}")
            loaded.append(frame)
        return loaded

    def inpaint(self, overlays: Sequence[OverlayRegion]) -> List[np.ndarray]:
        grouped = group_by_frame(overlays)
        for frame_index, regions in grouped.items():
            if frame_index < 0 or frame_index >= len(self.frames):
                continue
            for region in regions:
                self._inpaint_region(frame_index, region)
        return self.frames

    def _inpaint_region(self, frame_index: int, region: OverlayRegion) -> None:
        frame = self.frames[frame_index]
        original = self.frames_original
        h, w = frame.shape[:2]

        x1 = max(0, min(region.x1, w - 1))
        y1 = max(0, min(region.y1, h - 1))
        x2 = max(0, min(region.x2, w))
        y2 = max(0, min(region.y2, h))
        if x2 <= x1 or y2 <= y1:
            return

        patches = self._collect_neighbor_patches(frame_index, x1, y1, x2, y2)
        if patches:
            stacked = np.stack(patches, axis=0)
            fill_patch = np.median(stacked, axis=0).astype(np.uint8)
            frame[y1:y2, x1:x2] = fill_patch
        elif self.config.use_inpaint_fallback:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            frame[:, :] = inpainted

    def _collect_neighbor_patches(
        self, frame_index: int, x1: int, y1: int, x2: int, y2: int
    ) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        window = self.config.window
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbor_index = frame_index + offset
            if neighbor_index < 0 or neighbor_index >= len(self.frames_original):
                continue
            neighbor_frame = self.frames_original[neighbor_index]
            patch = neighbor_frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patches.append(patch)
        return patches

    def save_frames(self, output_dir: Path, image_format: Optional[str] = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        entries: Sequence[Dict[str, object]] = self.metadata["frames"]  # type: ignore[index]
        for entry, frame in zip(entries, self.frames):
            frame_path = resolve_frame_path(output_dir, entry)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = image_format or frame_path.suffix.lstrip(".") or "png"
            filename = frame_path.with_suffix(f".{suffix}")
            cv2.imwrite(str(filename), frame)

    def write_video(self, output_path: Path, codec: str = "mp4v") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = self.frames[0].shape[:2]
        fps = float(self.metadata.get("fps", 30.0))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create video writer for {output_path}")

        for frame in self.frames:
            writer.write(frame)
        writer.release()
