"""Shared data structures passed between modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# The three malignancy classes used by the private datasets in the paper.
CLASSES = ["pre_invasive", "minimally_invasive", "invasive"]


@dataclass
class NoduleFeatures:
    """Ground-truth-ish morphological attributes of a nodule.

    In a real deployment these are *perceived* by the vision-language model.
    In the demo they are generated synthetically and also serve as the oracle
    the Evidence Verifier checks agent claims against.
    """
    lobe: str                    # e.g. "right upper lobe"
    density: str                 # "ggo" | "part_solid" | "solid"
    shape: str                   # "round" | "oval" | "irregular"
    margin: str                  # "smooth" | "lobulated" | "spiculated"
    spiculation: bool
    pleural_indentation: bool
    vascular_convergence: bool
    air_bronchogram: bool
    cavitation: bool
    long_diameter_mm: float
    short_diameter_mm: float
    prior_long_diameter_mm: Optional[float] = None  # for the Temporal agent

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class Measurements:
    long_diameter_mm: float
    short_diameter_mm: float
    volume_mm3: float


@dataclass
class Case:
    case_id: str
    image: np.ndarray                     # 2D CT slice (H, W), float in [0,1]
    true_mask: np.ndarray                 # ground-truth binary mask (H, W)
    features: NoduleFeatures              # ORACLE morphology (evaluation only)
    true_label: str                       # oracle malignancy class

    # What the vision-language stack *perceives* (report + specialists reason from
    # this). In the clean regime `perceived == features`; under noise it is a
    # corrupted copy, so the Evidence Verifier and Calibration/Abstention agents
    # have something real to catch. Never read this for scoring.
    perceived: Optional[NoduleFeatures] = None

    # filled in as the pipeline runs
    expert_masks: List[np.ndarray] = field(default_factory=list)
    final_mask: Optional[np.ndarray] = None
    measurements: Optional[Measurements] = None
    report: Optional[str] = None

    def __post_init__(self):
        if self.perceived is None:
            # default: perception is faithful (clean regime)
            self.perceived = NoduleFeatures(**self.features.as_dict())

    @property
    def true_malignant(self) -> bool:
        """Clinically, minimally-invasive and invasive adenocarcinoma are both
        malignant; pre-invasive (AAH/AIS) is the 'safe' class. A false negative
        is calling a truly-malignant nodule pre-invasive."""
        return self.true_label in ("minimally_invasive", "invasive")


@dataclass
class AgentOpinion:
    role: str
    probs: Dict[str, float]               # distribution over CLASSES
    rationale: str
    cited_features: List[str]             # feature keys the agent claims to have used
    confidence: float                     # the agent's self-reported confidence

    @property
    def label(self) -> str:
        return max(self.probs, key=self.probs.get)


@dataclass
class BoardOutcome:
    final_label: str
    calibrated_probs: Dict[str, float]
    confidence: float
    abstained: bool
    rounds_used: int
    guideline: str
    evidence_trace: List[str]
    per_agent: List[AgentOpinion]
    notes: List[str] = field(default_factory=list)
    # NEW: aggregate distribution BEFORE temperature scaling (for calibration eval)
    raw_probs: Dict[str, float] = field(default_factory=dict)
    # NEW: the specialists' opinions in the final round *before* the verifier
    # cleaned them, kept so the verifier's flags can be scored against the oracle
    raw_per_agent: List[AgentOpinion] = field(default_factory=list)
    n_hard_flags: int = 0
    n_soft_flags: int = 0
    guideline_detail: Dict[str, str] = field(default_factory=dict)
    temperature_used: float = 1.0
