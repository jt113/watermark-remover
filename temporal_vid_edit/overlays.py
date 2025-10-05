from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class OverlayRegion:
    frame: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def _normalize_region(entry: Dict[str, int]) -> OverlayRegion:
    frame = int(entry["frame"])

    if {"x1", "y1", "x2", "y2"} <= entry.keys():
        x1 = int(entry["x1"])
        y1 = int(entry["y1"])
        x2 = int(entry["x2"])
        y2 = int(entry["y2"])
    elif {"x", "y", "width", "height"} <= entry.keys():
        x1 = int(entry["x"])
        y1 = int(entry["y"])
        x2 = x1 + int(entry["width"])
        y2 = y1 + int(entry["height"])
    else:
        raise ValueError(
            "Overlay entries must include either x1/y1/x2/y2 or x/y/width/height"
        )

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Overlay region must have positive width and height")

    return OverlayRegion(frame=frame, x1=x1, y1=y1, x2=x2, y2=y2)


def load_overlays(source: Path) -> List[OverlayRegion]:
    data = Path(source)
    if not data.exists():
        raise FileNotFoundError(f"Overlay file not found: {source}")

    with data.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)

    if isinstance(parsed, dict) and "overlays" in parsed:
        entries = parsed["overlays"]
    else:
        entries = parsed

    if not isinstance(entries, Iterable):
        raise TypeError("Overlay file must contain a list of entries")

    result: List[OverlayRegion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each overlay entry must be a dictionary")
        result.append(_normalize_region(entry))

    return result


def group_by_frame(overlays: Iterable[OverlayRegion]) -> Dict[int, List[OverlayRegion]]:
    grouped: Dict[int, List[OverlayRegion]] = {}
    for region in overlays:
        grouped.setdefault(region.frame, []).append(region)
    return grouped
