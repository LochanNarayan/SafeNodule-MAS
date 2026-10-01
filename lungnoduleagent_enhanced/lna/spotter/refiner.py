"""Segmentation Refiner agent (NEW).

Cleans the averaged consensus mask with a light morphological closing
(dilate then erode) implemented in pure numpy, then keeps the largest
connected component. Tightens boundaries before measurement.
"""
from __future__ import annotations

import numpy as np


def _dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _erode(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _largest_component(m: np.ndarray) -> np.ndarray:
    """Flood-fill labeling; return the largest 4-connected component."""
    h, w = m.shape
    labels = np.zeros((h, w), dtype=int)
    cur = 0
    best_id, best_size = 0, 0
    for i in range(h):
        for j in range(w):
            if m[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                size = 0
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
                if size > best_size:
                    best_size, best_id = size, cur
    if best_id == 0:
        return m
    return labels == best_id


def refine(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    m = _erode(_dilate(m))          # morphological closing
    m = _largest_component(m)
    return m
