"""Focal Prompting Mechanism — focal crop of image + mask with a context margin.

Returns the cropped image, cropped mask, and the bounding box. Keeping a margin
around the nodule preserves surrounding anatomy (pleura, vessels) exactly as the
paper's focal-crop-with-context describes.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def focal_crop(image: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.5
               ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image, mask, (0, 0, image.shape[0], image.shape[1])
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    my = int((y1 - y0 + 1) * margin_ratio)
    mx = int((x1 - x0 + 1) * margin_ratio)
    h, w = image.shape
    y0 = max(0, y0 - my); y1 = min(h - 1, y1 + my)
    x0 = max(0, x0 - mx); x1 = min(w - 1, x1 + mx)
    return image[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1], (y0, x0, y1, x1)
