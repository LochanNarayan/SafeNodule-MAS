"""Chair / Moderator agent (NEW) — explicit debate control.

Decides when the board has converged (a super-majority agrees on a label and it
is stable across rounds) versus when it is deadlocked and should stop and let the
calibration agent decide / escalate.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ..config import Config
from ..types import AgentOpinion


class Chair:
    def __init__(self, config: Config):
        self.config = config
        self._last_leading: Optional[str] = None

    def leading_label(self, opinions: List[AgentOpinion]) -> str:
        c = Counter(op.label for op in opinions)
        return c.most_common(1)[0][0]

    def agreement(self, opinions: List[AgentOpinion]) -> float:
        if not opinions:
            return 0.0
        c = Counter(op.label for op in opinions)
        return c.most_common(1)[0][1] / len(opinions)

    def has_converged(self, opinions: List[AgentOpinion], round_i: int) -> bool:
        lead = self.leading_label(opinions)
        agree = self.agreement(opinions)
        stable = (lead == self._last_leading)
        self._last_leading = lead
        thr = self.config.convergence_agreement
        # stop early if the leader is stable across rounds AND a super-majority
        # agree, OR the very first round is already unanimous
        if round_i == 1:
            return agree >= 1.0 - 1e-9
        return agree >= thr and stable
