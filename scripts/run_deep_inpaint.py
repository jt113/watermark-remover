#!/usr/bin/env python3
"""Convenience launcher for deep inpainting with E2FGVI-HQ."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - torch required for GPU runs
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES = Path("frames")
DEFAULT_OVERLAYS = Path("overlays.json")
DEFAULT_OUTPUT = Path("cleaned_deep.mp4")
REPO_DIR = Path("third_party/E2FGVI")
WEIGHTS_PATH = REPO_DIR / "weights" / "e2fgvi_hq.pth"
WRAPPER_SCRIPT = Path("scripts/e2fgvi_infer.py")


def _find_forward_flag(args: list[str], flag: str) -> list[str]:
    matches: list[str] = []
    prefix = flag + "="
    i = 0
    while i < len(args):
        token = args[i]
        if token == flag:
            if i + 1 < len(args):
                matches.append(args[i + 1])
            i += 2
            continue
        if token.startswith(prefix):
            matches.append(token[len(prefix) :])
            i += 1
            continue
        i += 1
    return matches


def ensure_model(download: bool, forward_args: list[str]) -> None:
    if not REPO_DIR.exists():
        if not download:
            raise SystemExit(
                "E2FGVI repository not found. Run scripts/download_e2fgvi.py first or pass --download."
            )
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "download_e2fgvi.py")]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    if not WRAPPER_SCRIPT.exists():
        raise SystemExit(f"Missing helper script: {WRAPPER_SCRIPT}")

    weight_overrides = _find_forward_flag(forward_args, "--deep-weights")
    if weight_overrides:
        custom_path = Path(weight_overrides[-1])
        if custom_path.exists():
            return

    if WEIGHTS_PATH.exists():
        return

    if not download:
        raise SystemExit(
            "E2FGVI weights not found. Run scripts/download_e2fgvi.py or provide --extra --deep-weights <path>."
        )

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "download_e2fgvi.py")]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    if not WEIGHTS_PATH.exists():
        raise SystemExit(
            "Download finished but weights still missing. Provide --extra --deep-weights <path>"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep inpainting with E2FGVI-HQ", allow_abbrev=False
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--overlays", type=Path, default=DEFAULT_OVERLAYS)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--download", action="store_true", help="Download the model if missing")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to run both the pipeline and the deep model",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        action="append",
        default=[],
        help="Extra flags forwarded to main.py (repeatable)",
    )
    args, unknown = parser.parse_known_args()

    forward: list[str] = []
    for group in args.extra:
        if group:
            forward.extend(group)
    forward.extend(unknown)
    args.forward_args = forward
    return args


def main() -> None:
    args = parse_args()
    ensure_model(download=args.download, forward_args=args.forward_args)

    if torch is not None:
        cuda_available = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            device_index = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            print(f"Using GPU device {device_index}: {device_name}")
        else:
            print("No CUDA GPU detected; deep inpainting will run on CPU and may be slow.")
    else:
        print("PyTorch not available in this environment; skipping GPU check.")

    cmd = [
        args.python,
        "main.py",
        "inpaint",
        str(args.frames),
        str(args.overlays),
        "--method",
        "deep",
        "--output-video",
        str(args.output_video),
        "--deep-cmd",
        args.python,
        str(WRAPPER_SCRIPT),
    ]

    forward_args = args.forward_args

    if not _find_forward_flag(forward_args, "--deep-weights"):
        cmd.extend(["--deep-weights", str(WEIGHTS_PATH)])

    if not _find_forward_flag(forward_args, "--deep-weights-arg"):
        cmd.append("--deep-weights-arg=--ckpt")

    if "--deep-context" not in forward_args and not any(
        token.startswith("--deep-context=") for token in forward_args
    ):
        cmd.extend(["--deep-context", str(args.context)])

    if "--deep-extra-arg" not in forward_args and not any(
        token.startswith("--deep-extra-arg=") for token in forward_args
    ):
        cmd.extend([
            "--deep-extra-arg=--model",
            "--deep-extra-arg=e2fgvi_hq",
        ])

    if forward_args:
        cmd.extend(forward_args)

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Deep inpainting pipeline failed with exit code {exc.returncode}. Review the logs above."
        ) from None


if __name__ == "__main__":
    main()
