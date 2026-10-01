"""End-to-end LungNoduleAgent-Enhanced pipeline.

    detect (Nodule Spotter) -> describe (Simulated Radiologist)
        -> route (Triage) -> diagnose (Doctor Board + safety agents)

Returns a structured PipelineResult per case, including the measurements needed
for the safety / cost / calibration evaluation (LLM-call counts, pre-calibration
probabilities, verifier flag counts, under-triage flag).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import Config
from .doctors import DoctorBoard
from .doctors.verifier import verifier_scorecard
from .knowledge import MedicalGraphRAG
from .llm import get_backend
from .llm.instrumented import CountingBackend
from .memory import HierarchicalMemory
from .radiologist import SimulatedRadiologist
from .router import oracle_difficulty, route
from .spotter import NoduleSpotter
from .spotter.clustering import iou
from .types import Case


@dataclass
class PipelineResult:
    case_id: str
    detection_iou: float
    report: str
    routing: str
    predicted_label: str
    true_label: str
    correct: bool
    confidence: float
    abstained: bool
    rounds_used: int
    guideline: str
    calibrated_probs: Dict[str, float]
    notes: List[str]
    evidence_trace: List[str]
    # ---- NEW: measurements for the safety / cost / calibration study ----
    raw_probs: Dict[str, float] = field(default_factory=dict)
    n_llm_calls: int = 0
    llm_call_breakdown: Dict[str, int] = field(default_factory=dict)
    n_hard_flags: int = 0
    n_soft_flags: int = 0
    verifier_counts: Dict[str, int] = field(default_factory=dict)
    under_triaged: bool = False
    true_malignant: bool = False
    guideline_detail: Dict[str, str] = field(default_factory=dict)
    temperature_used: float = 1.0

    def summary(self) -> str:
        flag = "ABSTAIN->human" if self.abstained else ("OK" if self.correct else "WRONG")
        return (f"[{self.case_id}] det_IoU={self.detection_iou:.2f} "
                f"route={self.routing} pred={self.predicted_label} "
                f"true={self.true_label} conf={self.confidence:.2f} "
                f"rounds={self.rounds_used} calls={self.n_llm_calls} -> {flag}")


class LungNoduleAgent:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.backend = CountingBackend(get_backend(self.config))
        self.rag = MedicalGraphRAG(self.config)
        self.memory = HierarchicalMemory()
        self.spotter = NoduleSpotter(self.backend, self.config)
        self.radiologist = SimulatedRadiologist(self.backend, self.config)
        self.board = DoctorBoard(self.backend, self.rag, self.config, self.memory)

    def diagnose(self, case: Case) -> PipelineResult:
        self.backend.reset()

        # 1. detect
        case = self.spotter.run(case)
        det_iou = iou(case.final_mask, case.true_mask)
        self.memory.set("final_mask", case.final_mask)
        self.memory.set("measurements", case.measurements)

        # 2. describe
        case = self.radiologist.run(case)
        self.memory.set("report", case.report)

        # 3. route
        decision = route(case, self.config)

        # 4. diagnose
        outcome = self.board.run(case, roles=decision.roles,
                                 max_rounds=decision.max_rounds)

        correct = (outcome.final_label == case.true_label) and not outcome.abstained
        vcounts = verifier_scorecard(case, outcome.raw_per_agent)

        return PipelineResult(
            case_id=case.case_id,
            detection_iou=det_iou,
            report=case.report,
            routing=decision.difficulty,
            predicted_label=outcome.final_label,
            true_label=case.true_label,
            correct=correct,
            confidence=outcome.confidence,
            abstained=outcome.abstained,
            rounds_used=outcome.rounds_used,
            guideline=outcome.guideline,
            calibrated_probs=outcome.calibrated_probs,
            notes=outcome.notes,
            evidence_trace=outcome.evidence_trace,
            raw_probs=outcome.raw_probs,
            n_llm_calls=self.backend.total,
            llm_call_breakdown=self.backend.snapshot(),
            n_hard_flags=outcome.n_hard_flags,
            n_soft_flags=outcome.n_soft_flags,
            verifier_counts=vcounts,
            under_triaged=(decision.difficulty == "easy"
                           and oracle_difficulty(case) == "hard"),
            true_malignant=case.true_malignant,
            guideline_detail=outcome.guideline_detail,
            temperature_used=outcome.temperature_used,
        )

    def diagnose_many(self, cases: List[Case]) -> List[PipelineResult]:
        return [self.diagnose(c) for c in cases]

    def fit_calibration(self, cases: List[Case]) -> float:
        """Run a held-out split, fit the temperature by NLL minimisation on the
        board's raw aggregate distributions, and install it for later cases.
        Returns the fitted temperature (1.0 if calibration is disabled)."""
        from .doctors.calibration import fit_temperature
        if not self.config.use_calibration or not cases:
            return 1.0
        self.board.temperature_override = None      # collect raw, T-independent
        samples = [(r.raw_probs, r.true_label)
                   for r in self.diagnose_many([c for c in cases]) if r.raw_probs]
        t = fit_temperature(samples)
        self.board.temperature_override = t
        return t
