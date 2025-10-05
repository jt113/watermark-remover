from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frames import load_metadata, resolve_frame_path
from .overlays import OverlayRegion, group_by_frame


@dataclass
class FlowInpainterConfig:
    window: int = 3
    flow_preset: str = "medium"  # ultrafast | fast | medium | fine
    use_seamless_clone: bool = True
    temporal_median: bool = True
    fall_back_temporal: bool = True
    max_sources: int = 6
    max_search_distance: int = 120


class FlowAlignedInpainter:
    def __init__(self, frames_dir: Path, config: Optional[FlowInpainterConfig] = None) -> None:
        self.frames_dir = Path(frames_dir)
        self.config = config or FlowInpainterConfig()
        self.metadata = load_metadata(self.frames_dir)
        self.frames = self._load_frames()
        self.frames_original = [frame.copy() for frame in self.frames]
        self._flow_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._warp_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._clean_indices: List[int] = []

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
        overlay_frames = {region.frame for region in overlays}
        total = len(self.frames)
        self._clean_indices = [idx for idx in range(total) if idx not in overlay_frames]
        if not self._clean_indices:
            self._clean_indices = list(range(total))

        grouped = group_by_frame(overlays)
        for frame_index, regions in grouped.items():
            if frame_index < 0 or frame_index >= len(self.frames):
                continue
            for region in regions:
                self._inpaint_region(frame_index, region)
        return self.frames

    def _inpaint_region(self, frame_index: int, region: OverlayRegion) -> None:
        frame = self.frames[frame_index]
        h, w = frame.shape[:2]

        x1 = max(0, min(region.x1, w - 1))
        y1 = max(0, min(region.y1, h - 1))
        x2 = max(0, min(region.x2, w))
        y2 = max(0, min(region.y2, h))
        if x2 <= x1 or y2 <= y1:
            return

        patches = self._collect_flow_aligned_patches(frame_index, x1, y1, x2, y2)

        if patches:
            stacked = np.stack(patches, axis=0).astype(np.float32)
            if self.config.temporal_median:
                fill_patch = np.median(stacked, axis=0)
            else:
                weights = np.linspace(1.0, 0.2, num=stacked.shape[0], dtype=np.float32)
                weights = weights / weights.sum()
                fill_patch = np.tensordot(weights, stacked, axes=(0, 0))
            fill_patch = np.clip(fill_patch, 0, 255).astype(np.uint8)
            self._blend_patch(frame_index, (x1, y1, x2, y2), fill_patch)
            return

        if self.config.fall_back_temporal:
            self._opencv_fallback(frame_index, (x1, y1, x2, y2))

    def _opencv_fallback(self, frame_index: int, box: Tuple[int, int, int, int]) -> None:
        frame = self.frames[frame_index]
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        frame[:, :] = inpainted

    def _select_source_indices(self, target_index: int) -> List[int]:
        candidates: List[Tuple[int, int]] = []
        max_distance = max(1, self.config.max_search_distance)
        for index in self._clean_indices:
            if index == target_index:
                continue
            distance = abs(index - target_index)
            if distance > max_distance:
                continue
            candidates.append((distance, index))
        candidates.sort(key=lambda item: item[0])
        return [idx for _, idx in candidates[: self.config.max_sources]]

    def _collect_flow_aligned_patches(
        self, frame_index: int, x1: int, y1: int, x2: int, y2: int
    ) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        sources = self._select_source_indices(frame_index)
        if not sources:
            # fallback to local temporal window when no clean frames found
            window = self.config.window
            sources = [
                frame_index + offset
                for offset in range(-window, window + 1)
                if offset != 0
                and 0 <= frame_index + offset < len(self.frames_original)
            ]

        for neighbor_index in sources:
            warped = self._warp_neighbor(neighbor_index, frame_index)
            patch = warped[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patches.append(patch)
        return patches

    def _warp_neighbor(self, source_index: int, target_index: int) -> np.ndarray:
        cache_key = (source_index, target_index)
        if cache_key in self._warp_cache:
            return self._warp_cache[cache_key]

        flow = self._compute_flow(source_index, target_index)
        src = self.frames_original[source_index]
        h, w = src.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self._warp_cache[cache_key] = warped
        return warped

    def _compute_flow(self, source_index: int, target_index: int) -> np.ndarray:
        cache_key = (source_index, target_index)
        if cache_key in self._flow_cache:
            return self._flow_cache[cache_key]

        src = self.frames_original[source_index]
        dst = self.frames_original[target_index]
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        preset_candidates = [
            ("ultrafast", "DISOPTICAL_FLOW_PRESET_ULTRAFAST"),
            ("fast", "DISOPTICAL_FLOW_PRESET_FAST"),
            ("medium", "DISOPTICAL_FLOW_PRESET_MEDIUM"),
            ("fine", "DISOPTICAL_FLOW_PRESET_FINE"),
        ]
        preset_map: Dict[str, int] = {}
        for name, attr in preset_candidates:
            value = getattr(cv2, attr, None)
            if value is not None:
                preset_map[name] = value

        if not preset_map:
            raise RuntimeError("OpenCV build does not expose DIS optical flow presets")

        requested = self.config.flow_preset.lower()
        preset = preset_map.get(requested)
        if preset is None:
            preset = preset_map.get("medium") or next(iter(preset_map.values()))
        dis = cv2.DISOpticalFlow_create(preset)
        flow = dis.calc(src_gray, dst_gray, None)
        self._flow_cache[cache_key] = flow
        return flow

    def _blend_patch(self, frame_index: int, box: Tuple[int, int, int, int], patch: np.ndarray) -> None:
        frame = self.frames[frame_index]
        x1, y1, x2, y2 = box
        if not self.config.use_seamless_clone:
            frame[y1:y2, x1:x2] = patch
            return

        base = frame.copy()
        base[y1:y2, x1:x2] = patch
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        try:
            blended = cv2.seamlessClone(base, frame, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            frame[y1:y2, x1:x2] = patch
        else:
            frame[:, :] = blended

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
