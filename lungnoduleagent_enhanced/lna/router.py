"""Triage / Router agent (NEW) — adaptive cost control.

Estimates case difficulty from the *report / perceived* features. Easy cases
(clear benign GGO, or unambiguous large spiculated solid mass) use a single
specialist and a single round; ambiguous cases (part-solid, mid-size,
conflicting signs) get the full board and multiple rounds. This is what keeps
average inference cost low.

``oracle_difficulty`` recomputes the same predicate on the ground-truth
features so the evaluator can measure the *under-triage rate* — genuinely hard
cases that perception made look easy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Config
from .types import Case, NoduleFeatures


@dataclass
class RoutingDecision:
    difficulty: str          # "easy" | "hard"
    roles: List[str]
    max_rounds: int
    reason: str


def _is_ambiguous(f: NoduleFeatures, size: float) -> bool:
    signs = sum([f.spiculation, f.pleural_indentation, f.vascular_convergence])
    return (
        f.density == "part_solid" or
        (6 <= size <= 12) or
        f.margin == "lobulated" or
        (0 < signs < 2)
    )


def oracle_difficulty(case: Case) -> str:
    f = case.features
    size = f.long_diameter_mm
    return "hard" if _is_ambiguous(f, size) else "easy"


def route(case: Case, config: Config) -> RoutingDecision:
    f = case.perceived or case.features
    size = case.measurements.long_diameter_mm if case.measurements else f.long_diameter_mm

    if not config.use_router:
        return RoutingDecision("hard", list(config.board_roles),
                               config.max_debate_rounds,
                               "router disabled -> full board on every case")

    if not _is_ambiguous(f, size):
        return RoutingDecision(
            difficulty="easy",
            roles=config.board_roles[:config.easy_roles],
            max_rounds=1,
            reason="unambiguous morphology -> single specialist, 1 round",
        )
    return RoutingDecision(
        difficulty="hard",
        roles=list(config.board_roles),
        max_rounds=config.max_debate_rounds,
        reason="ambiguous morphology -> full board, multi-round debate",
    )
