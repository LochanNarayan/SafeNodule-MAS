"""Doctor Board — orchestrates the roundtable debate and all safety agents.

Round structure:
  1. each specialist opines independently (from the CT report, not the oracle)
  2. Evidence Verifier grounds each opinion against the report (flags/soft-flags
     claims, penalises confidence); the raw opinions are kept for scoring
  3. Devil's-Advocate challenges the leading label
  4. Chair checks convergence -> stop, else specialists revise toward the
     summarized peer consensus and repeat
Finally the Calibration/Abstention agent produces the calibrated diagnosis (or
escalates), and the Guideline agent attaches a Lung-RADS/Fleischner assessment.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import Config
from ..knowledge import MedicalGraphRAG
from ..llm.base import LLMBackend
from ..memory import HierarchicalMemory
from ..types import AgentOpinion, BoardOutcome, Case
from . import guideline
from .calibration import calibrate_and_decide
from .chair import Chair
from .devils_advocate import DevilsAdvocate
from .specialists import SpecialistAgent
from .verifier import verify_opinion


class DoctorBoard:
    def __init__(self, backend: LLMBackend, rag: MedicalGraphRAG,
                 config: Config, memory: Optional[HierarchicalMemory] = None):
        self.backend = backend
        self.rag = rag
        self.config = config
        self.memory = memory or HierarchicalMemory()
        self.role_weights: Dict[str, float] = {
            "thoracic_radiologist": 1.0, "pathologist": 1.0,
            "oncologist": 1.2, "pulmonologist": 0.9,
            "devils_advocate": 0.6,
        }
        # set by LungNoduleAgent.fit_calibration(); overrides the fixed T
        self.temperature_override: Optional[float] = None

    def run(self, case: Case, roles: List[str],
            max_rounds: Optional[int] = None) -> BoardOutcome:
        cfg = self.config
        max_rounds = cfg.max_debate_rounds if max_rounds is None else max_rounds
        chair = Chair(cfg)
        devil = DevilsAdvocate(self.backend, cfg) if cfg.use_devils_advocate else None
        specialists = [SpecialistAgent(r, self.backend, self.rag, cfg) for r in roles]

        evidence_trace: List[str] = []
        notes: List[str] = []
        opinions: List[AgentOpinion] = []
        raw_opinions: List[AgentOpinion] = []
        peer_label: Optional[str] = None
        rounds_used = 0
        n_hard = n_soft = 0
        converged = False

        for round_i in range(1, max_rounds + 1):
            rounds_used = round_i
            raw_opinions = [s.opine(case, round_i, peer_label=peer_label)
                            for s in specialists]

            if cfg.use_verifier:
                verified = []
                for op in raw_opinions:
                    cleaned, trace = verify_opinion(case, op)
                    verified.append(cleaned)
                    evidence_trace.extend(trace)
                    n_hard += sum(1 for t in trace if t.startswith("FLAG"))
                    n_soft += sum(1 for t in trace if t.startswith("soft"))
                opinions = verified
            else:
                opinions = list(raw_opinions)

            self.memory.log_turn(f"round_{round_i}", [o.__dict__ for o in opinions])

            leading = chair.leading_label(opinions)
            peer_label = leading

            if devil is not None:
                challenge = devil.challenge(case, leading, round_i)
                notes.append(f"round {round_i}: devil's advocate argued for "
                             f"{challenge.cited_features[0] or 'an alternative'}")

            if chair.has_converged(opinions, round_i):
                converged = True
                notes.append(f"converged at round {round_i} "
                             f"(agreement={chair.agreement(opinions):.2f})")
                break

        if not converged:
            agree = chair.agreement(opinions)
            if agree >= cfg.convergence_agreement:
                converged = True
                notes.append(f"stopped at round limit {max_rounds}; "
                             f"super-majority reached (agreement={agree:.2f})")
            else:
                notes.append(f"no consensus after {max_rounds} rounds "
                             f"(agreement={agree:.2f}) -> calibration layer decides")

        agg_inputs = list(opinions)
        devil_probs = None
        if devil is not None:
            final_challenge = devil.challenge(case, chair.leading_label(opinions),
                                              rounds_used)
            agg_inputs.append(final_challenge)
            devil_probs = final_challenge.probs

        label, probs, raw_probs, conf, abstain, cal_notes = calibrate_and_decide(
            agg_inputs, cfg, self.role_weights,
            agreement=chair.agreement(opinions), devil_probs=devil_probs,
            temperature_override=self.temperature_override)
        notes.extend(cal_notes)

        temp_used = (1.0 if not cfg.use_calibration else
                     (self.temperature_override if self.temperature_override is not None
                      else cfg.calibration_temperature))

        guide_detail = guideline.assess_detail(case) if cfg.use_guideline_agent else {}
        guide = guideline.assess(case) if cfg.use_guideline_agent else ""

        self.memory.remember_case(case.case_id, case.features.as_dict(), label)

        return BoardOutcome(
            final_label=label, calibrated_probs=probs, confidence=conf,
            abstained=abstain, rounds_used=rounds_used, guideline=guide,
            evidence_trace=evidence_trace, per_agent=opinions, notes=notes,
            raw_probs=raw_probs, raw_per_agent=raw_opinions,
            n_hard_flags=n_hard, n_soft_flags=n_soft, guideline_detail=guide_detail,
            temperature_used=temp_used)
