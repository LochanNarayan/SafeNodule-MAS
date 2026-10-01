"""Shared, transparent clinical heuristics.

This is deliberately simple and rule-based. It is used in two places:
  1. the synthetic data generator, to assign a coherent ground-truth label;
  2. the mock LLM backend, so simulated doctors reason *from the same feature
     evidence* (with role bias + noise) and therefore mostly agree with the
     ground truth, letting debate / verification / calibration be demonstrated.

Replace this entirely when you plug in a real VLM — the rest of the system does
not depend on it.
"""
from __future__ import annotations

import math
from typing import Dict

from .types import CLASSES


def feature_class_scores(f: Dict) -> Dict[str, float]:
    """Map morphological features -> unnormalized malignancy-class scores.

    Higher score for `invasive` when the classic aggressive signs are present
    (solid/part-solid, spiculation, lobulation, large size, pleural indentation,
    vascular convergence). `pre_invasive` favored for pure GGO, small, smooth.
    """
    pre = 0.0
    mini = 0.0
    inv = 0.0

    density = f.get("density", "solid")
    if density == "ggo":
        pre += 2.0
        mini += 0.5
    elif density == "part_solid":
        mini += 1.5
        inv += 1.0
    else:  # solid
        inv += 1.5

    margin = f.get("margin", "smooth")
    if margin == "smooth":
        pre += 1.0
    elif margin == "lobulated":
        mini += 1.0
        inv += 0.8
    elif margin == "spiculated":
        inv += 2.0

    if f.get("spiculation"):
        inv += 1.5
    if f.get("pleural_indentation"):
        inv += 1.0
    if f.get("vascular_convergence"):
        inv += 1.0
    if f.get("shape") == "irregular":
        inv += 0.8
        mini += 0.3

    size = float(f.get("long_diameter_mm", 8.0))
    if size < 6:
        pre += 1.2
    elif size < 10:
        mini += 1.0
    else:
        inv += 1.2

    # growth signal (Temporal agent evidence)
    prior = f.get("prior_long_diameter_mm")
    if prior:
        growth = (size - float(prior)) / max(float(prior), 1e-6)
        if growth > 0.25:
            inv += 1.5
        elif growth > 0.05:
            mini += 0.6

    return {"pre_invasive": pre, "minimally_invasive": mini, "invasive": inv}


def softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    t = max(temperature, 1e-6)
    vals = {k: v / t for k, v in scores.items()}
    m = max(vals.values())
    exps = {k: math.exp(v - m) for k, v in vals.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def argmax_label(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)


def normalized_entropy(probs: Dict[str, float]) -> float:
    """Entropy scaled to [0,1] (1 = maximally uncertain across the classes)."""
    h = -sum(p * math.log(p + 1e-12) for p in probs.values())
    return h / math.log(len(CLASSES))
