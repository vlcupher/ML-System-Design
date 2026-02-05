"""Preprocessing: crop empty edges before sending image to the model."""

from typing import Optional, Tuple

from PIL import Image


def crop_empty_edges(
    img: Image.Image,
    padding: int = 4,
    background_threshold: int = 250,
) -> Image.Image:
    """
    Crop unused (empty) margins so the image has minimal empty space.
    Uses bounding box of non-background pixels; supports RGB and RGBA.

    Args:
        img: PIL Image (RGB or RGBA).
        padding: Pixels to leave around the content bbox (avoid cropping too tight).
        background_threshold: Pixels with max channel value >= this are considered background (0-255).

    Returns:
        Cropped image, or original if bbox cannot be determined.
    """
    if img.size[0] == 0 or img.size[1] == 0:
        return img

    gray = img.convert("L") if img.mode != "L" else img

    try:
        import numpy as np
    except ImportError:
        bbox = _getbbox_pil_only(gray, background_threshold)
        if bbox is None:
            return img
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.width, x2 + padding)
        y2 = min(img.height, y2 + padding)
        return img.crop((x1, y1, x2, y2))

    if img.mode == "RGBA":
        alpha = img.split()[3]
        try:
            arr = np.array(gray)
            alpha_arr = np.array(alpha)
            non_empty = (arr < background_threshold) & (alpha_arr > 30)
        except Exception:
            non_empty = np.array(gray) < background_threshold
    else:
        non_empty = np.array(gray) < background_threshold

    rows = non_empty.any(axis=1)
    cols = non_empty.any(axis=0)
    if not rows.any() or not cols.any():
        return img

    y1, y2 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax())
    x1, x2 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax())

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.width, x2 + padding)
    y2 = min(img.height, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _getbbox_pil_only(gray: Image.Image, background_threshold: int) -> Optional[Tuple[int, int, int, int]]:
    """Fallback bbox using only PIL (no numpy): find first/last row and column with non-background pixel."""
    w, h = gray.size
    data = list(gray.getdata())
    x1, y1, x2, y2 = w, h, 0, 0
    for y in range(h):
        for x in range(w):
            if data[y * w + x] < background_threshold:
                x1 = min(x1, x)
                y1 = min(y1, y)
                x2 = max(x2, x + 1)
                y2 = max(y2, y + 1)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)
