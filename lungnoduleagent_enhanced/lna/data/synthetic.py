"""Synthetic CT-slice + nodule generator.

Produces a Case with a 2D image, a ground-truth nodule mask (an elliptical blob
with a texture), coherent morphological features, and an oracle malignancy label
derived from those features via the shared clinical heuristic. This lets the full
pipeline run and be scored without any real dataset.

Regimes (set via Config) let the harness *stress* the safety and cost machinery:

  clean       perceived features == oracle; balanced classes
  noisy       perceived features + report corrupted -> the Evidence Verifier and
              Calibration/Abstention agents have real errors to catch
  borderline  only cases near the class decision boundary are kept -> abstention
  imbalanced  class prior skewed -> macro-F1 / minority-class stress
  adversarial noisy + borderline + degraded detector masks

Swap this module for a real LIDC-IDRI / DICOM loader to run on actual data; the
rest of the pipeline consumes `Case` objects and does not care where they came from.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..clinical import argmax_label, feature_class_scores, softmax
from ..config import Config
from ..types import Case, NoduleFeatures

LOBES = ["right upper lobe", "right middle lobe", "right lower lobe",
         "left upper lobe", "left lower lobe"]
DENSITIES = ["ggo", "part_solid", "solid"]
SHAPES = ["round", "oval", "irregular"]
MARGINS = ["smooth", "lobulated", "spiculated"]

_CATEGORICAL_ALT = {
    "density": DENSITIES,
    "shape": SHAPES,
    "margin": MARGINS,
    "lobe": LOBES,
}
_BOOL_KEYS = ("spiculation", "pleural_indentation", "vascular_convergence",
              "air_bronchogram", "cavitation")


def _draw_blob(h: int, w: int, cy: int, cx: int, ry: float, rx: float,
               irregular: bool, rng: np.random.RandomState) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = r <= 1.0
    if irregular:
        for _ in range(rng.randint(2, 5)):
            by = cy + int(rng.normal(0, ry))
            bx = cx + int(rng.normal(0, rx))
            br = max(2.0, rng.uniform(0.3, 0.7) * min(ry, rx))
            bump = ((yy - by) ** 2 + (xx - bx) ** 2) <= br ** 2
            mask = mask | bump
    return mask.astype(np.float32)


def _sample_features(rng: np.random.RandomState,
                     force_features: Optional[dict] = None) -> NoduleFeatures:
    density = rng.choice(DENSITIES)
    shape = rng.choice(SHAPES)
    margin = rng.choice(MARGINS)
    spiculation = bool(margin == "spiculated" or rng.rand() < 0.3)
    long_d = float(np.clip(rng.normal(10, 5), 3, 30))
    short_d = float(np.clip(long_d * rng.uniform(0.6, 0.95), 2, long_d))
    has_prior = rng.rand() < 0.5
    feats = NoduleFeatures(
        lobe=str(rng.choice(LOBES)),
        density=str(density),
        shape=str(shape),
        margin=str(margin),
        spiculation=spiculation,
        pleural_indentation=bool(rng.rand() < 0.35),
        vascular_convergence=bool(rng.rand() < 0.3),
        air_bronchogram=bool(rng.rand() < 0.2),
        cavitation=bool(rng.rand() < 0.15),
        long_diameter_mm=round(long_d, 1),
        short_diameter_mm=round(short_d, 1),
        prior_long_diameter_mm=round(long_d * rng.uniform(0.6, 0.95), 1) if has_prior else None,
    )
    if force_features:
        for k, v in force_features.items():
            setattr(feats, k, v)
    return feats


def _class_margin(feats: NoduleFeatures) -> float:
    """Softmax gap between the top-2 malignancy classes for these features.
    Small gap == genuinely ambiguous case."""
    probs = sorted(softmax(feature_class_scores(feats.as_dict())).values(), reverse=True)
    return probs[0] - probs[1]


def _corrupt(feats: NoduleFeatures, rng: np.random.RandomState,
             p: float) -> NoduleFeatures:
    """Return a copy of `feats` with each categorical/boolean attribute
    independently corrupted with probability `p`. Size is jittered by ~15%."""
    d = feats.as_dict()
    for key, alts in _CATEGORICAL_ALT.items():
        if rng.rand() < p:
            d[key] = str(rng.choice([a for a in alts if a != d[key]]))
    for key in _BOOL_KEYS:
        if rng.rand() < p:
            d[key] = not d[key]
    if rng.rand() < p:
        d["long_diameter_mm"] = round(float(np.clip(
            d["long_diameter_mm"] * rng.uniform(0.7, 1.4), 3, 40)), 1)
        d["short_diameter_mm"] = round(min(d["short_diameter_mm"],
                                           d["long_diameter_mm"]), 1)
    return NoduleFeatures(**d)


def make_case(case_id: str, seed: int = 0, config: Optional[Config] = None,
              force_features: Optional[dict] = None) -> Case:
    cfg = config or Config()
    rng = np.random.RandomState(seed)
    h = w = 128

    # ---- sample oracle features, honouring regime constraints ----
    feats: Optional[NoduleFeatures] = None
    target_label: Optional[str] = None
    if cfg.class_prior is not None and not force_features:
        target_label = ["pre_invasive", "minimally_invasive", "invasive"][
            int(rng.choice(3, p=np.asarray(cfg.class_prior) / np.sum(cfg.class_prior)))]

    for _attempt in range(400):
        cand = _sample_features(rng, force_features)
        if target_label is not None and \
                argmax_label(feature_class_scores(cand.as_dict())) != target_label:
            continue
        if cfg.borderline_margin > 0 and not force_features and \
                _class_margin(cand) > cfg.borderline_margin:
            continue
        feats = cand
        break
    if feats is None:                       # constraints too tight -> last candidate
        feats = cand

    true_label = argmax_label(feature_class_scores(feats.as_dict()))

    # ---- what the VLM stack perceives (corrupted under noisy regimes) ----
    if cfg.feature_noise > 0 and not force_features:
        perc_rng = np.random.RandomState(seed ^ 0x5DEECE66)
        perceived = _corrupt(feats, perc_rng, cfg.feature_noise)
    else:
        perceived = NoduleFeatures(**feats.as_dict())

    # ---- build image + mask (detector_quality widens perturbation later) ----
    ry = max(3.0, feats.long_diameter_mm * 0.6)
    rx = max(3.0, feats.short_diameter_mm * 0.6)
    cy = rng.randint(int(ry) + 5, h - int(ry) - 5)
    cx = rng.randint(int(rx) + 5, w - int(rx) - 5)

    lung = rng.normal(0.15, 0.03, size=(h, w)).astype(np.float32)
    mask = _draw_blob(h, w, cy, cx, ry, rx, feats.shape == "irregular", rng)
    intensity = {"ggo": 0.35, "part_solid": 0.55, "solid": 0.8}[feats.density]
    image = lung + mask * intensity + rng.normal(0, 0.02, size=(h, w)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    return Case(case_id=case_id, image=image, true_mask=(mask > 0.5),
                features=feats, perceived=perceived, true_label=true_label)


def make_dataset(n: int, seed: int = 0,
                 config: Optional[Config] = None) -> List[Case]:
    return [make_case(f"case_{i:03d}", seed=seed + i, config=config)
            for i in range(n)]
