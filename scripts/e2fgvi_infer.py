#!/usr/bin/env python3
"""Headless wrapper around E2FGVI test inference."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV (cv2) is required for E2FGVI inference. Install opencv-python in your environment."
    ) from exc

import numpy as np

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Pillow is required for E2FGVI inference. Install pillow in your environment."
    ) from exc

from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyTorch is required for E2FGVI inference. Install it in your environment before running."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
E2FGVI_ROOT = PROJECT_ROOT / "third_party" / "E2FGVI"
if str(E2FGVI_ROOT) not in sys.path:
    sys.path.insert(0, str(E2FGVI_ROOT))

try:
    from core.utils import to_tensors  # noqa: E402  pylint: disable=wrong-import-position
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Unable to import E2FGVI modules. Confirm the repository exists under third_party/E2FGVI."
    ) from exc


def read_frames(path: Path, use_mp4: bool) -> list[Image.Image]:
    frames: list[Image.Image] = []
    if use_mp4:
        capture = cv2.VideoCapture(str(path))
        success, frame = capture.read()
        while success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            success, frame = capture.read()
        capture.release()
    else:
        for child in sorted(os.listdir(path)):
            image = cv2.imread(str(path / child))
            if image is None:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not frames:
        raise FileNotFoundError(f"No frames found at {path}")
    return frames


def read_masks(path: Path, size: tuple[int, int]) -> list[Image.Image]:
    masks: list[Image.Image] = []
    if path.suffix.lower() == ".mp4":
        capture = cv2.VideoCapture(str(path))
        success, frame = capture.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).resize(size, resample=Image.NEAREST)
            masks.append(image)
            success, frame = capture.read()
        capture.release()
    else:
        for child in sorted(os.listdir(path)):
            image = cv2.imread(str(path / child))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks.append(Image.fromarray(image).resize(size, resample=Image.NEAREST))
    if not masks:
        raise FileNotFoundError(f"No masks found at {path}")
    return masks


def resize_frames(frames: list[Image.Image], size: tuple[int, int] | None) -> tuple[list[Image.Image], tuple[int, int]]:
    if size is None:
        size = frames[0].size
        return frames, size
    resized = [frame.resize(size) for frame in frames]
    return resized, size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2FGVI headless inference")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="e2fgvi_hq", choices=["e2fgvi", "e2fgvi_hq"])
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=5)
    parser.add_argument("--savefps", type=int, default=24)
    parser.add_argument("--set_size", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = Path(args.video)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not mask_path.exists():
        raise SystemExit(f"Mask not found: {mask_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    default_neighbor_stride = parser.get_default("neighbor_stride")
    device_index: int | None = None
    gpu_props: torch.cuda.DeviceProperties | None = None
    total_mem_mb: int | None = None

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU requested via --device gpu but no CUDA device is available.")
        device_index = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(device_index)
        total_mem_mb = gpu_props.total_memory // (1024 * 1024)
        print(f"Using CUDA device {device_index}: {gpu_props.name} ({total_mem_mb} MiB VRAM)")
        device = torch.device("cuda")
    else:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            gpu_props = torch.cuda.get_device_properties(device_index)
            total_mem_mb = gpu_props.total_memory // (1024 * 1024)
            if total_mem_mb <= 3072:
                print(f"Detected CUDA device {device_index}: {gpu_props.name} ({total_mem_mb} MiB VRAM)")
                print("VRAM below 3 GiB; running inference on CPU for stability.")
                device = torch.device("cpu")
                device_index = None
                gpu_props = None
                total_mem_mb = None
            else:
                print(f"Using CUDA device {device_index}: {gpu_props.name} ({total_mem_mb} MiB VRAM)")
                device = torch.device("cuda")
        else:
            print("No CUDA device detected; running inference on CPU.")
            device = torch.device("cpu")

    if device.type == "cuda" and total_mem_mb is not None:
        if args.num_ref == -1 and total_mem_mb <= 4096:
            args.num_ref = 4
            print("Auto-adjusted num_ref to 4 due to limited GPU memory.")

        if (
            args.neighbor_stride == default_neighbor_stride
            and total_mem_mb <= 4096
            and default_neighbor_stride > 3
        ):
            args.neighbor_stride = 3
            print("Auto-adjusted neighbor_stride to 3 due to limited GPU memory.")

    if args.model == "e2fgvi":
        target_size = (432, 240)
    elif args.set_size:
        target_size = (args.width, args.height)
    else:
        target_size = None

    module = importlib.import_module(f"model.{args.model}")
    model = module.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    model.eval()

    use_mp4 = video_path.suffix.lower() == ".mp4"
    frames = read_frames(video_path, use_mp4)
    frames, target_size = resize_frames(frames, target_size)
    h, w = target_size[1], target_size[0]
    video_length = len(frames)

    to_tensor = to_tensors()
    frame_tensors = torch.stack(
        [(to_tensor([frame]).squeeze(0) * 2 - 1) for frame in frames], dim=0
    )
    frames_np = [np.array(frame).astype(np.uint8) for frame in frames]

    masks = read_masks(mask_path, target_size)
    if len(masks) != video_length:
        raise SystemExit(
            "Mask count does not match frame count. Ensure mask video has the same number of frames."
        )
    binary_masks = []
    for mask in masks:
        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask_array = np.any(mask_array != 0, axis=2, keepdims=True).astype(np.uint8)
        else:
            mask_array = np.expand_dims((mask_array != 0).astype(np.uint8), axis=2)
        binary_masks.append(mask_array)
    mask_tensors = torch.stack([to_tensor([mask]).squeeze(0) for mask in masks], dim=0)
    comp_frames: list[np.ndarray | None] = [None] * video_length

    neighbor_stride = args.neighbor_stride
    default_fps = args.savefps
    ref_length = args.step
    num_ref = args.num_ref

    def get_ref_index(frame_idx: int, neighbor_ids: list[int], length: int) -> list[int]:
        ref_index: list[int] = []
        if num_ref == -1:
            for i in range(0, length, ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, frame_idx - ref_length * (num_ref // 2))
            end_idx = min(length, frame_idx + ref_length * (num_ref // 2) + 1)
            for i in range(start_idx, end_idx, ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        return ref_index

    print("Running E2FGVI inference...")
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        combined_ids = neighbor_ids + ref_ids

        batch_imgs = frame_tensors[combined_ids].unsqueeze(0).to(device, non_blocking=True)
        batch_masks = mask_tensors[combined_ids].unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            masked_imgs = batch_imgs * (1 - batch_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[:, :, :, : h + h_pad, :]
            masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[:, :, :, :, : w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i, neighbor_id in enumerate(neighbor_ids):
                comp = (np.array(pred_imgs[i]).astype(np.uint8) * binary_masks[neighbor_id] +
                        frames_np[neighbor_id] * (1 - binary_masks[neighbor_id]))
                if comp_frames[neighbor_id] is None:
                    comp_frames[neighbor_id] = comp
                else:
                    comp_frames[neighbor_id] = (
                        comp_frames[neighbor_id].astype(np.float32) * 0.5 + comp.astype(np.float32) * 0.5
                    )

        if device.type == "cuda":
            del batch_imgs, batch_masks, masked_imgs
            torch.cuda.empty_cache()

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Unable to open output writer: {output_path}")

    for frame in comp_frames:
        if frame is None:
            raise SystemExit("Model did not generate all frames")
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
