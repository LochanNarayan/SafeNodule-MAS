"""Mixture of Experts detector (mocked).

Each 'expert' foundation model returns a candidate mask. We simulate diverse
detectors by perturbing the ground-truth nodule mask differently per expert
(shift, scale, occasional miss / false positive). This produces exactly the
noisy multi-mask set the clustering + judging panel are designed to clean up.

Replace `run` with real detector inference (nnUNet, MONAI, YOLO, etc.) to run
on true data — it just needs to return a list of binary masks.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..config import Config
from ..types import Case


def _perturb(mask: np.ndarray, rng: np.random.RandomState,
             jitter: float = 1.5) -> np.ndarray:
    h, w = mask.shape
    dy, dx = int(rng.normal(0, jitter)), int(rng.normal(0, jitter))
    shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    ys, xs = np.nonzero(shifted)
    if len(ys) == 0:
        return shifted.astype(bool)
    # under a degraded detector, some experts also erode or dilate the boundary
    if jitter > 2.0:
        if rng.rand() < 0.5:
            shifted[1:, :] |= shifted[:-1, :]; shifted[:, 1:] |= shifted[:, :-1]
        else:
            shifted[1:, :] &= shifted[:-1, :]; shifted[:, 1:] &= shifted[:, :-1]
    return shifted.astype(bool)


def run(case: Case, config: Config) -> List[np.ndarray]:
    masks: List[np.ndarray] = []
    q = max(0.05, float(getattr(config, "detector_quality", 1.0)))
    jitter = 1.5 / q                        # lower quality -> wider mask scatter
    miss_p = 0.15 + 0.35 * (1.0 - q)        # lower quality -> more misses
    for e in range(config.n_experts):
        rng = np.random.RandomState(config.rng_seed_for(case.case_id, "expert", e))
        if rng.rand() < miss_p:
            # this expert misses the nodule -> emits a spurious false-positive blob
            h, w = case.true_mask.shape
            fp = np.zeros((h, w), dtype=bool)
            cy, cx = rng.randint(10, h - 10), rng.randint(10, w - 10)
            fp[cy - 3:cy + 3, cx - 3:cx + 3] = True
            masks.append(fp)
        else:
            masks.append(_perturb(case.true_mask, rng, jitter=jitter))
    return masks
