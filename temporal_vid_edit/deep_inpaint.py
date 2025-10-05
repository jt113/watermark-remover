from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frames import load_metadata, resolve_frame_path
from .overlays import OverlayRegion, group_by_frame


@dataclass
class DeepModelConfig:
    command: Sequence[str]
    weights: Optional[Path] = None
    extra_args: Sequence[str] = ()
    context: int = 3
    codec: str = "mp4v"
    keep_temp: bool = False
    video_arg: str = "--video"
    mask_arg: str = "--mask"
    output_arg: str = "--output"
    weights_arg: str = "--weights"
    workdir: Optional[Path] = None


class DeepInpainter:
    def __init__(self, frames_dir: Path, config: DeepModelConfig) -> None:
        if not config.command:
            raise ValueError("DeepModelConfig.command must contain the model invocation")

        self.frames_dir = Path(frames_dir)
        self.config = config
        self.metadata = load_metadata(self.frames_dir)
        self.frames = self._load_frames()
        # Use copies to keep an original reference around in case segments overlap
        self.frames_original = [frame.copy() for frame in self.frames]
        self.overlay_map: Dict[int, List[OverlayRegion]] = {}

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
        if not overlays:
            return self.frames

        self.overlay_map = group_by_frame(overlays)
        overlay_frames = sorted(self.overlay_map.keys())
        if not overlay_frames:
            return self.frames

        segments = self._build_segments(overlay_frames)
        if not segments:
            return self.frames

        temp_root_ctx = None
        if self.config.keep_temp:
            path = Path(tempfile.mkdtemp(prefix="deep_inpaint_", dir=str(self.frames_dir)))
            temp_root = path
        else:
            temp_root_ctx = tempfile.TemporaryDirectory(prefix="deep_inpaint_")
            temp_root = Path(temp_root_ctx.name)

        try:
            for index, (start, end) in enumerate(segments):
                segment_dir = temp_root / f"segment_{index:03d}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                video_path = segment_dir / "input.mp4"
                mask_path = segment_dir / "mask.mp4"
                output_path = segment_dir / "output.mp4"

                self._export_segment(start, end, video_path, mask_path)
                self._run_model(video_path, mask_path, output_path)
                self._import_segment(start, end, output_path)
        finally:
            if temp_root_ctx is not None:
                temp_root_ctx.cleanup()
        return self.frames

    def _build_segments(self, overlay_frames: Iterable[int]) -> List[Tuple[int, int]]:
        sorted_frames = sorted(set(overlay_frames))
        if not sorted_frames:
            return []

        contiguous: List[Tuple[int, int]] = []
        start = prev = sorted_frames[0]
        for frame in sorted_frames[1:]:
            if frame == prev + 1:
                prev = frame
                continue
            contiguous.append((start, prev))
            start = prev = frame
        contiguous.append((start, prev))

        expanded: List[Tuple[int, int]] = []
        total_frames = len(self.frames)
        context = max(0, self.config.context)
        for start, end in contiguous:
            new_start = max(0, start - context)
            new_end = min(total_frames - 1, end + context)
            if expanded and new_start <= expanded[-1][1] + 1:
                expanded[-1] = (expanded[-1][0], max(expanded[-1][1], new_end))
            else:
                expanded.append((new_start, new_end))
        return expanded

    def _export_segment(self, start: int, end: int, video_path: Path, mask_path: Path) -> None:
        fps = float(self.metadata.get("fps", 30.0))
        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)

        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Unable to write segment video to {video_path}")
        if not mask_writer.isOpened():
            raise RuntimeError(f"Unable to write mask video to {mask_path}")

        for frame_index in range(start, end + 1):
            frame = self.frames_original[frame_index]
            video_writer.write(frame)

            mask = np.zeros((height, width), dtype=np.uint8)
            for region in self.overlay_map.get(frame_index, []):
                x1 = max(0, min(region.x1, width - 1))
                y1 = max(0, min(region.y1, height - 1))
                x2 = max(0, min(region.x2, width))
                y2 = max(0, min(region.y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue
                mask[y1:y2, x1:x2] = 255
            mask_bgr = cv2.merge([mask, mask, mask])
            mask_writer.write(mask_bgr)

        video_writer.release()
        mask_writer.release()

    def _run_model(self, video_path: Path, mask_path: Path, output_path: Path) -> None:
        command = list(self.config.command)
        command.extend([self.config.video_arg, str(video_path)])
        command.extend([self.config.mask_arg, str(mask_path)])
        command.extend([self.config.output_arg, str(output_path)])
        if self.config.weights:
            command.extend([self.config.weights_arg, str(self.config.weights)])
        if self.config.extra_args:
            command.extend(self.config.extra_args)

        workdir = str(self.config.workdir) if self.config.workdir else None

        try:
            subprocess.run(command, check=True, cwd=workdir)
        except FileNotFoundError as exc:
            raise SystemExit(
                "Deep model command failed: executable not found. Check --deep-cmd"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Deep model command failed with exit code {exc.returncode}"
            ) from exc

        if not output_path.exists():
            raise SystemExit(
                f"Deep model did not produce expected output video at {output_path}"
            )

    def _import_segment(self, start: int, end: int, output_path: Path) -> None:
        capture = cv2.VideoCapture(str(output_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to read deep model output {output_path}")

        segment_frames: List[np.ndarray] = []
        success, frame = capture.read()
        while success:
            segment_frames.append(frame)
            success, frame = capture.read()
        capture.release()

        expected = end - start + 1
        if len(segment_frames) != expected:
            raise SystemExit(
                f"Deep model output frame count mismatch ({len(segment_frames)} vs {expected})"
            )

        for offset, frame in enumerate(segment_frames):
            target_index = start + offset
            self.frames[target_index] = frame
            self.frames_original[target_index] = frame.copy()

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
