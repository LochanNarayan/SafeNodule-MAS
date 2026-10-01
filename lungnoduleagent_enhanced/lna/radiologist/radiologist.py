"""Simulated Radiologist: generates a localized CT report from the focal crop.

In mock mode the report is synthesized from the perceived features (which, in the
demo, equal the ground-truth features plus the measured size). With a real VLM
backend you would pass the cropped image + mask as an image block alongside the
MedPrompt; the returned text is stored verbatim.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..llm.base import LLMBackend
from ..types import Case
from .focal import focal_crop

_DROPPABLE_SIGNS = ("spiculation", "pleural_indentation", "vascular_convergence",
                    "air_bronchogram", "cavitation")

MEDPROMPT = (
    "Given a lung CT image and the masked nodule region, write a localized report. "
    "Describe lobe location, density, margin, shape; state presence/absence of "
    "spiculation, cavitation, air bronchogram, pleural indentation and vascular "
    "convergence. Use formal radiological style, be definitive only where evidence "
    "supports it, and do not mention non-annotated findings."
)


class SimulatedRadiologist:
    def __init__(self, backend: LLMBackend, config: Config):
        self.backend = backend
        self.config = config

    def run(self, case: Case) -> Case:
        # focal crop (used verbatim by a real VLM; here it validates geometry)
        _img, _msk, _box = focal_crop(case.image, case.final_mask)
        # the radiologist reports on what it *perceives*, not the oracle
        feats = (case.perceived or case.features).as_dict()
        meas = case.measurements.__dict__ if case.measurements else {}

        # imperfect reporting: independently omit some truly-present signs
        omit = []
        p = float(getattr(self.config, "report_dropout", 0.0))
        if p > 0:
            rng = np.random.RandomState(self.config.rng_seed_for(case.case_id, "report"))
            omit = [s for s in _DROPPABLE_SIGNS if feats.get(s) and rng.rand() < p]

        case.report = self.backend.complete(
            system="You are an expert thoracic radiologist.",
            prompt=MEDPROMPT,
            context={"task": "ct_report", "case_id": case.case_id,
                     "features": feats, "measurements": meas, "omit": omit},
        )
        return case
