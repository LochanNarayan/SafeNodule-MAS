"""Role-differentiated specialist doctor agents.

Each agent has its own system persona and queries its own slice of the knowledge
graph, then emits a probability distribution over the malignancy classes with a
rationale and the features it claims to rely on.
"""
from __future__ import annotations

from typing import Dict, Optional

from ..config import Config
from ..knowledge import MedicalGraphRAG
from ..llm.base import LLMBackend
from ..types import AgentOpinion, Case

ROLE_PERSONA = {
    "thoracic_radiologist": "You are a thoracic radiologist focused on nodule morphology on CT.",
    "pathologist": "You are a pulmonary pathologist reasoning about histologic subtype.",
    "oncologist": "You are a thoracic oncologist assessing malignancy risk and aggressiveness.",
    "pulmonologist": "You are a pulmonologist weighing clinical context and differentials.",
}

ROLE_QUERY_HINT = {
    "thoracic_radiologist": "margin density shape spiculation imaging features",
    "pathologist": "adenocarcinoma subtype invasive minimally invasive pre-invasive histology",
    "oncologist": "malignancy risk aggressiveness growth staging",
    "pulmonologist": "differential follow-up Fleischner Lung-RADS benign",
}


class SpecialistAgent:
    def __init__(self, role: str, backend: LLMBackend, rag: MedicalGraphRAG, config: Config):
        self.role = role
        self.backend = backend
        self.rag = rag
        self.config = config

    def opine(self, case: Case, round_i: int,
              peer_label: Optional[str] = None) -> AgentOpinion:
        # retrieve role-specific knowledge
        query = f"{ROLE_QUERY_HINT[self.role]} {case.report}"
        knowledge = self.rag.summarize_for(query)

        # perceived features: from the CT report the agent was given, NOT the
        # oracle. Under noisy regimes these differ from case.features, which is
        # exactly what the Evidence Verifier and calibration layer must handle.
        perceived = (case.perceived or case.features).as_dict()
        perceived["long_diameter_mm"] = case.measurements.long_diameter_mm
        perceived["short_diameter_mm"] = case.measurements.short_diameter_mm

        out = self.backend.structured(
            system=ROLE_PERSONA[self.role] + " Ground every claim in the CT report and knowledge.",
            prompt=(f"CT REPORT:\n{case.report}\n\nRELEVANT KNOWLEDGE:\n{knowledge}\n\n"
                    "Give probabilities over {pre_invasive, minimally_invasive, invasive}."),
            task="specialist_opinion",
            context={"case_id": case.case_id, "role": self.role, "round": round_i,
                     "features_perceived": perceived, "peer_label": peer_label,
                     "knowledge": knowledge},
        )
        return AgentOpinion(
            role=self.role,
            probs={k: float(v) for k, v in out["probs"].items()},
            rationale=out["rationale"],
            cited_features=list(out["cited_features"]),
            confidence=float(out["confidence"]),
        )
