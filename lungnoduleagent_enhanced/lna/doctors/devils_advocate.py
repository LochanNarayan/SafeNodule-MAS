"""Devil's-Advocate agent (NEW) — anti-groupthink / anti-false-negative.

Given the board's current leading label, it argues for the strongest alternative.
Its objection must be considered before consensus locks, which combats the
"confidently wrong consensus" failure mode of naive multi-agent debate.
"""
from __future__ import annotations

from typing import Dict

from ..config import Config
from ..llm.base import LLMBackend
from ..types import AgentOpinion, Case


class DevilsAdvocate:
    role = "devils_advocate"

    def __init__(self, backend: LLMBackend, config: Config):
        self.backend = backend
        self.config = config

    def challenge(self, case: Case, leading_label: str, round_i: int) -> AgentOpinion:
        perceived = (case.perceived or case.features).as_dict()
        perceived["long_diameter_mm"] = case.measurements.long_diameter_mm
        out = self.backend.structured(
            system=("You are the devil's advocate on a tumor board. Argue against the "
                    "current leading diagnosis to stress-test it."),
            prompt=f"Leading diagnosis: {leading_label}. Build the strongest counter-case.",
            task="devils_advocate",
            context={"case_id": case.case_id, "role": self.role, "round": round_i,
                     "leading_label": leading_label, "features_perceived": perceived},
        )
        return AgentOpinion(
            role=self.role,
            probs={k: float(v) for k, v in out["probs"].items()},
            rationale=out["rationale"],
            cited_features=[out.get("alternative", "")],
            confidence=0.5,
        )
