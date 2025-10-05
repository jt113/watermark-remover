#!/usr/bin/env python3
"""Fetch the E2FGVI-HQ video inpainting model and its weights."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_REPO_URL = "https://github.com/MCG-NKU/E2FGVI.git"
DEFAULT_WEIGHTS_URLS = [
    "https://github.com/MCG-NKU/E2FGVI/releases/download/v1.0/e2fgvi_hq.pth",
    "https://huggingface.co/MCG-NKU/E2FGVI/resolve/main/e2fgvi_hq.pth",
    "https://huggingface.co/MCG-NKU/E2FGVI/resolve/main/e2fgvi_hq.pth?download=1",
]
DEFAULT_WEIGHTS_MD5 = "44d346f2f4d51ed245032e0f1b6771e0"


def download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urlopen(url) as response, destination.open("wb") as handle:
        total = response.length or 0
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                print(f"  {percent:3d}%", end="\r", flush=True)
        print("" )


def ensure_repo(repo_url: str, target_dir: Path, branch: Optional[str]) -> None:
    if target_dir.exists():
        print(f"Repository already present at {target_dir}")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", repo_url, str(target_dir)]
    if branch:
        cmd.extend(["-b", branch])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_weights(urls: Iterable[str], destination: Path, md5: Optional[str]) -> None:
    if destination.exists():
        if md5 and verify_md5(destination, md5):
            print(f"Weights already downloaded at {destination}")
            return
        print(f"Weights exist but checksum mismatch, re-downloading {destination}")
        destination.unlink()

    errors = []
    for url in urls:
        try:
            download_file(url, destination)
        except (HTTPError, URLError) as exc:
            errors.append(f"{url} -> {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
            continue

        if md5 and not verify_md5(destination, md5):
            destination.unlink(missing_ok=True)
            errors.append(f"{url} -> MD5 verification failed")
            continue

        print(f"Downloaded weights to {destination}")
        return

    error_text = "\n".join(errors) or "unknown errors"
    raise SystemExit(
        "Unable to download weights. Tried the following sources:\n" + error_text
    )


def verify_md5(path: Path, expected: str) -> bool:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    match = actual == expected.lower()
    if match:
        print(f"Verified MD5 for {path} ({actual})")
    else:
        print(f"MD5 mismatch for {path}: expected {expected}, got {actual}")
    return match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the E2FGVI-HQ model")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-dir", type=Path, default=Path("third_party/E2FGVI"))
    parser.add_argument("--branch", help="Optional git branch to checkout")
    parser.add_argument(
        "--weights-url",
        action="append",
        default=list(DEFAULT_WEIGHTS_URLS),
        help="Candidate URLs for the weights (can repeat)",
    )
    parser.add_argument("--weights", type=Path, default=Path("third_party/E2FGVI/weights/e2fgvi_hq.pth"))
    parser.add_argument("--weights-md5", default=DEFAULT_WEIGHTS_MD5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_repo(args.repo_url, args.repo_dir, args.branch)
    ensure_weights(args.weights_url, args.weights, args.weights_md5)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
