"""Central configuration for the enhanced pipeline.

Everything tunable lives here so experiments are reproducible: change a value,
re-run, and every module picks it up. This directly answers the "reproducibility
gaps" critique of the original paper (DBSCAN eps, MinPts, #experts, #agents,
#rounds were all unspecified there).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---- global ----
    seed: int = 7
    backend: str = "mock"          # "mock" | "anthropic" | "openai"
    model: str = "claude-sonnet-5" # used only by real backends
    verbose: bool = True

    # ---- evaluation regime (NEW) ----
    # Controls how synthetic cases are generated so the safety/cost mechanisms
    # can be *stressed*, not just exercised on clean inputs.
    #   clean       : perceived features == oracle features, balanced classes
    #   noisy       : perceived features/report corrupted (verifier has work)
    #   borderline  : cases sampled near the class decision boundary (abstention)
    #   imbalanced  : class prior skewed toward pre_invasive (macro-F1 stress)
    #   adversarial : noisy + borderline + degraded detector masks
    regime: str = "clean"
    feature_noise: float = 0.0     # P(each perceived categorical feature corrupted)
    report_dropout: float = 0.0    # P(a truly-present sign is omitted from the report)
    detector_quality: float = 1.0  # 1.0 = normal MoE; <1 widens mask perturbations
    class_prior: Optional[List[float]] = None  # [pre, mini, invasive]; None = natural
    borderline_margin: float = 0.0  # if >0, keep only cases whose top-2 class
                                    # score gap (softmax) is below this value

    # ---- Nodule Spotter: Mixture of Experts ----
    n_experts: int = 5             # number of foundation detection models (mocked)

    # ---- Nodule Spotter: DBSCAN mask clustering ----
    # distance = 1 - IoU, so eps in [0,1]; eps=0.5 means "group masks with IoU>=0.5".
    dbscan_eps: float = 0.5
    dbscan_min_pts: int = 2
    mask_binarize_threshold: float = 0.5

    # ---- Nodule Spotter: Judging Panel (VLM peer review) ----
    n_judges: int = 3              # independent VLM judges voting on each candidate

    # ---- imaging geometry (for measurements) ----
    pixel_spacing_mm: float = 0.7  # mm per pixel (in-plane)
    slice_thickness_mm: float = 1.0

    # ---- Doctor Board ----
    # Roles that participate on hard cases. Easy cases use only the first `easy_roles`.
    board_roles: List[str] = field(default_factory=lambda: [
        "thoracic_radiologist",
        "pathologist",
        "oncologist",
        "pulmonologist",
    ])
    easy_roles: int = 1            # router uses this many roles for "easy" cases
    max_debate_rounds: int = 4
    convergence_agreement: float = 0.75  # fraction of agents that must agree to stop
    use_devils_advocate: bool = True
    use_verifier: bool = True
    use_guideline_agent: bool = True
    use_router: bool = True        # NEW: disable to force the full board on every case

    # ---- Calibration / Abstention ----
    use_calibration: bool = True           # NEW: disable to skip temperature scaling
    calibration_temperature: float = 1.3   # fallback T when no calibration split is used
    # Fraction of a run's cases held out to FIT the temperature by NLL
    # minimisation before scoring the rest. 0 -> use the fixed fallback T.
    calibration_split_frac: float = 0.30
    # Primary selective-prediction rule: abstain when the calibrated top-1 and
    # top-2 class probabilities are within this margin (the aggregate cannot
    # separate two diagnoses).
    abstain_margin_below: float = 0.10
    abstain_confidence_below: float = 0.40   # hard floor on the calibrated top prob
    abstain_entropy_above: float = 1.01      # normalized entropy gate (>1 => off)
    abstain_disagreement_below: float = 0.0  # inter-specialist agreement gate (0 => off)
    # Devil's-Advocate false-negative guard: if the board leans benign
    # (pre_invasive) but the Devil's-Advocate builds a case putting at least this
    # much probability on a malignant class, escalate to human review.
    devil_fn_guard_prob: float = 0.60

    # ---- Knowledge (GraphRAG) ----
    top_k_chunks: int = 3

    def rng_seed_for(self, *tags: str) -> int:
        """Deterministic per-(config.seed, tags) seed. Keeps runs reproducible
        while letting each agent/round behave distinctly."""
        h = self.seed
        for t in tags:
            for ch in str(t):
                h = (h * 131 + ord(ch)) % (2**31 - 1)
        return h

    # ---- regime presets -------------------------------------------------------
    @classmethod
    def for_regime(cls, regime: str, **overrides) -> "Config":
        """Return a Config with the noise/prior knobs set for a named regime."""
        presets = {
            "clean": dict(feature_noise=0.0, report_dropout=0.0,
                          detector_quality=1.0, class_prior=None, borderline_margin=0.0),
            "noisy": dict(feature_noise=0.25, report_dropout=0.30,
                          detector_quality=0.85, class_prior=None, borderline_margin=0.0),
            "borderline": dict(feature_noise=0.05, report_dropout=0.10,
                               detector_quality=1.0, class_prior=None,
                               borderline_margin=0.20),
            "imbalanced": dict(feature_noise=0.0, report_dropout=0.0,
                               detector_quality=1.0, class_prior=[0.6, 0.25, 0.15],
                               borderline_margin=0.0),
            "adversarial": dict(feature_noise=0.30, report_dropout=0.35,
                                detector_quality=0.7, class_prior=[0.45, 0.30, 0.25],
                                borderline_margin=0.18),
        }
        if regime not in presets:
            raise ValueError(f"unknown regime {regime!r}; choose from {list(presets)}")
        cfg = cls(regime=regime, **presets[regime])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
