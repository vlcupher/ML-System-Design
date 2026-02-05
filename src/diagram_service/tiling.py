"""Tiling utilities: split large images into overlapping crops."""

from dataclasses import dataclass
from typing import List, Tuple, Literal

from PIL import Image

Axis = Literal["width", "height"]


@dataclass
class Tile:
    index: int
    axis: Axis
    bbox: Tuple[int, int, int, int]
    image: Image.Image


def choose_num_tiles(long_dim: int, threshold: int = 800) -> int:
    if long_dim <= threshold:
        return 1
    if long_dim <= 2 * threshold:
        return 2
    if long_dim <= 3 * threshold:
        return 3
    return 4


def make_tiles(img: Image.Image, threshold: int = 800, overlap: float = 0.15) -> List[Tile]:
    w, h = img.size
    long_dim = max(w, h)
    if long_dim <= threshold:
        return [Tile(index=1, axis="width", bbox=(0, 0, w, h), image=img)]

    axis: Axis = "width" if w >= h else "height"
    n_tiles = choose_num_tiles(long_dim=long_dim, threshold=threshold)
    tiles: List[Tile] = []

    if axis == "width":
        step = w / n_tiles
        ov = int(step * overlap)
        for i in range(n_tiles):
            x1 = int(i * step) - (ov if i > 0 else 0)
            x2 = int((i + 1) * step) + (ov if i < n_tiles - 1 else 0)
            x1 = max(0, x1)
            x2 = min(w, x2)
            crop = img.crop((x1, 0, x2, h))
            tiles.append(Tile(index=i + 1, axis="width", bbox=(x1, 0, x2, h), image=crop))
    else:
        step = h / n_tiles
        ov = int(step * overlap)
        for i in range(n_tiles):
            y1 = int(i * step) - (ov if i > 0 else 0)
            y2 = int((i + 1) * step) + (ov if i < n_tiles - 1 else 0)
            y1 = max(0, y1)
            y2 = min(h, y2)
            crop = img.crop((0, y1, w, y2))
            tiles.append(Tile(index=i + 1, axis="height", bbox=(0, y1, w, y2), image=crop))
    return tiles


def allowed_sides(axis: Axis) -> str:
    return "LEFT/RIGHT" if axis == "width" else "TOP/BOTTOM"
