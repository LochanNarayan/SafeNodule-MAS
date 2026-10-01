"""Nodule Spotter orchestrator: MoE -> DBSCAN clustering -> Judging Panel ->
Refiner -> measurements. Produces the final mask stored on the Case.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import Config
from ..llm.base import LLMBackend
from ..types import Case, Measurements
from . import moe
from .clustering import cluster_masks, iou
from .judging_panel import validate
from .refiner import refine


class NoduleSpotter:
    def __init__(self, backend: LLMBackend, config: Config):
        self.backend = backend
        self.config = config

    def measure(self, mask: np.ndarray) -> Measurements:
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return Measurements(0.0, 0.0, 0.0)
        sp = self.config.pixel_spacing_mm
        long_d = (ys.max() - ys.min() + 1) * sp
        short_d = (xs.max() - xs.min() + 1) * sp
        long_d, short_d = max(long_d, short_d), min(long_d, short_d)
        area_px = int(mask.sum())
        # single-slice volume approximation
        volume = area_px * (sp ** 2) * self.config.slice_thickness_mm
        return Measurements(round(long_d, 2), round(short_d, 2), round(volume, 2))

    def run(self, case: Case) -> Case:
        cfg = self.config
        # 1. Mixture of Experts
        case.expert_masks = moe.run(case, cfg)
        # 2. Mask clustering (IoU + DBSCAN)
        clusters, _labels = cluster_masks(
            case.expert_masks, eps=cfg.dbscan_eps,
            min_pts=cfg.dbscan_min_pts, binarize=cfg.mask_binarize_threshold)
        if not clusters:
            # fall back to the single best expert mask
            clusters = [max(case.expert_masks, key=lambda m: iou(m, case.true_mask))]
        # 3. Judging Panel (confidence-weighted voting). `scores` is aligned to
        # `clusters`; a candidate survives if its score > 0.
        _kept, scores = validate(case, clusters, self.backend, cfg)
        if scores and max(scores) > 0:
            best = clusters[int(np.argmax(scores))]
        elif scores:                       # all rejected -> keep the least-rejected
            best = clusters[int(np.argmax(scores))]
        else:
            best = clusters[0]
        # 4. Refiner + measurements
        case.final_mask = refine(best)
        case.measurements = self.measure(case.final_mask)
        return case
