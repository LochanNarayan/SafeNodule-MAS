"""Calibration + Abstention agent (NEW) — safe uncertainty.

Aggregates the (verified, confidence-weighted) opinions into a single
distribution, applies temperature scaling to soften overconfident LLM outputs,
and decides whether to ABSTAIN (escalate to a human) when the top probability is
too low or the distribution is too uncertain (high entropy).

This is the antidote to the original paper's 'always be definitive' prompt.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..clinical import normalized_entropy, softmax
from ..config import Config
from ..types import CLASSES, AgentOpinion


def aggregate(opinions: List[AgentOpinion],
              role_weights: Dict[str, float]) -> Dict[str, float]:
    """Confidence- and role-weighted average of agent distributions."""
    acc = {c: 0.0 for c in CLASSES}
    total_w = 0.0
    for op in opinions:
        w = op.confidence * role_weights.get(op.role, 1.0)
        total_w += w
        for c in CLASSES:
            acc[c] += w * op.probs.get(c, 0.0)
    if total_w == 0:
        return {c: 1.0 / len(CLASSES) for c in CLASSES}
    return {c: v / total_w for c, v in acc.items()}


def temperature_scale(probs: Dict[str, float], temp: float) -> Dict[str, float]:
    """Apply temperature `temp` to a probability vector via its log-odds."""
    return softmax({c: math.log(p + 1e-9) for c, p in probs.items()},
                   temperature=max(temp, 1e-3))


def fit_temperature(samples: List[Tuple[Dict[str, float], str]],
                    grid: Tuple[float, float, int] = (0.4, 4.0, 73)) -> float:
    """Classic temperature scaling: pick T minimising negative log-likelihood of
    the true class on a held-out set of (raw_probs, true_label) pairs."""
    if not samples:
        return 1.0
    lo, hi, k = grid
    best_t, best_nll = 1.0, float("inf")
    for j in range(k):
        t = lo + (hi - lo) * j / (k - 1)
        nll = 0.0
        for probs, y in samples:
            scaled = temperature_scale(probs, t)
            nll -= math.log(scaled.get(y, 1e-9) + 1e-12)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def calibrate_and_decide(
        opinions: List[AgentOpinion], config: Config,
        role_weights: Dict[str, float], *,
        agreement: float = 1.0,
        devil_probs: Dict[str, float] = None,
        temperature_override: float = None,
) -> Tuple[str, Dict[str, float], Dict[str, float], float, bool, List[str]]:
    """Returns (label, calibrated_probs, raw_probs, confidence, abstain, notes).

    Abstains on any of: low calibrated top probability, high entropy, weak
    inter-specialist agreement, or the Devil's-Advocate false-negative guard
    (board leans benign while a credible malignant counter-case exists)."""
    notes: List[str] = []
    raw = aggregate(opinions, role_weights)

    if not config.use_calibration:
        temp = 1.0
    elif temperature_override is not None:
        temp = temperature_override
    else:
        temp = config.calibration_temperature
    calibrated = softmax({c: math.log(p + 1e-9) for c, p in raw.items()}, temperature=temp)

    ordered = sorted(calibrated.values(), reverse=True)
    top_label = max(calibrated, key=calibrated.get)
    top_p = ordered[0]
    margin = ordered[0] - ordered[1]
    ent = normalized_entropy(calibrated)

    abstain = False
    if margin < config.abstain_margin_below:
        abstain = True
        notes.append(f"ABSTAIN: top-1/top-2 margin {margin:.2f} "
                     f"< {config.abstain_margin_below}")
    if top_p < config.abstain_confidence_below:
        abstain = True
        notes.append(f"ABSTAIN: top probability {top_p:.2f} < {config.abstain_confidence_below}")
    if ent > config.abstain_entropy_above:
        abstain = True
        notes.append(f"ABSTAIN: normalized entropy {ent:.2f} > {config.abstain_entropy_above}")
    if config.abstain_disagreement_below > 0 and agreement < config.abstain_disagreement_below:
        abstain = True
        notes.append(f"ABSTAIN: only {agreement:.2f} of specialists agree "
                     f"(< {config.abstain_disagreement_below})")
    if devil_probs and config.use_devils_advocate and top_label == "pre_invasive":
        mal = devil_probs.get("minimally_invasive", 0.0) + devil_probs.get("invasive", 0.0)
        if mal >= config.devil_fn_guard_prob:
            abstain = True
            notes.append(f"ABSTAIN: board leans benign but Devil's-Advocate puts "
                         f"{mal:.2f} on malignancy (false-negative guard)")

    return top_label, calibrated, raw, top_p, abstain, notes
