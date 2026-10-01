"""Judging Panel — confidence-weighted VLM peer review (paper Eq. 4-5).

For each candidate mask, N_judges VLMs each return a binary opinion (+1/-1) and
a confidence C. Score = sum(sign * C). Score > 0 keeps the candidate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import Config
from ..llm.base import LLMBackend
from ..types import Case
from .clustering import iou

JUDGE_SYSTEM = ("You are a thoracic radiologist verifying whether a masked region "
                "of a lung CT truly contains a nodule.")


def validate(case: Case, candidates: List[np.ndarray], backend: LLMBackend,
             config: Config) -> Tuple[List[np.ndarray], List[float]]:
    kept: List[np.ndarray] = []
    scores: List[float] = []
    for ci, cand in enumerate(candidates):
        overlap = iou(cand, case.true_mask)   # oracle signal the mock judge uses
        total = 0.0
        for j in range(config.n_judges):
            out = backend.structured(
                JUDGE_SYSTEM,
                "Does the masked region contain a lung nodule?",
                task="judge_nodule",
                context={"case_id": case.case_id, "role": "judge",
                         "round": ci, "overlap": overlap, "judge": j},
            )
            sign = 1.0 if int(out["opinion"]) == 1 else -1.0
            total += sign * float(out["confidence"])   # Eq. 5
        scores.append(total)
        if total > 0:                                   # majority + confidence
            kept.append(cand)
    return kept, scores
