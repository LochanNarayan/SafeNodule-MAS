"""Guideline agent (NEW) — maps findings to a named, versioned standard
(Lung-RADS v2022 / Fleischner 2017), grounding 'evidence-based' output in
something auditable rather than an unspecified knowledge base.

Fix vs. the naive version: high-risk morphology (spiculation, pleural
indentation) *escalates* the category instead of being appended as a
contradictory modifier ("Lung-RADS 2 ... 4X highly suspicious").
"""
from __future__ import annotations

from typing import Dict

from ..types import Case

_CAT_ORDER = ["2", "3", "4A", "4B", "4X"]


def _bump(cat: str, to: str) -> str:
    return to if _CAT_ORDER.index(to) > _CAT_ORDER.index(cat) else cat


def assess_detail(case: Case) -> Dict[str, str]:
    f = case.features
    size = case.measurements.long_diameter_mm if case.measurements else f.long_diameter_mm
    density = f.density

    if density == "ggo":
        cat, rec = "2", "conservative follow-up (Fleischner: CT at 6-12 months if >6 mm)"
    elif density == "part_solid":
        if size >= 6:
            cat, rec = "4A", "PET-CT and/or tissue sampling"
        else:
            cat, rec = "3", "short-interval follow-up CT at 6 months"
    else:  # solid
        if size > 8:
            cat, rec = "4A", "PET-CT and/or biopsy"
        elif size >= 6:
            cat, rec = "3", "follow-up CT at 3-6 months"
        else:
            cat, rec = "2", "benign appearance; 12-month follow-up if screening"

    # high-risk morphology escalates the category (Lung-RADS 4X logic)
    escalators = []
    if f.spiculation:
        escalators.append("spiculation")
    if f.pleural_indentation:
        escalators.append("pleural indentation")
    if f.vascular_convergence:
        escalators.append("vascular convergence")
    if escalators:
        cat = _bump(cat, "4X")
        rec = "PET-CT and tissue sampling (high-risk morphology)"

    # interval growth also escalates
    prior = f.prior_long_diameter_mm
    if prior and size > prior * 1.25:
        cat = _bump(cat, "4B")
        escalators.append(f"interval growth {prior:.1f}->{size:.1f} mm")
        rec = "PET-CT and tissue sampling (documented growth)"

    return {
        "system": "Lung-RADS v2022",
        "category": f"Lung-RADS {cat}",
        "recommendation": rec,
        "drivers": ", ".join(escalators) if escalators
        else f"{density.replace('_', '-')} density, {size:.1f} mm",
    }


def assess(case: Case) -> str:
    d = assess_detail(case)
    return f"{d['category']} ({d['system']}): {d['recommendation']} | drivers: {d['drivers']}"
