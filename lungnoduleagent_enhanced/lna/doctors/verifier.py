"""Evidence Verifier agent (NEW) — report-grounded anti-hallucination check.

The verifier keeps every diagnostic claim traceable to the shared evidence. It
parses the *CT report* into the morphological facts the report asserts, then
grades each feature a specialist cites:

  CONTRADICTED  report asserts the opposite (``density:solid`` vs "ggo density",
                or ``spiculation`` cited while the report says it is absent)
                -> hard flag, claim dropped, confidence penalised 0.20
  UNMENTIONED   report never mentions the cited sign
                -> soft flag, claim dropped, confidence penalised 0.05
  SUPPORTED     -> kept

Grounding is on ``case.report`` (what the pipeline produced), never on
``case.features`` (the oracle). ``verifier_scorecard`` — evaluation only — uses
the oracle to measure whether the hard flags are the *right* ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..types import AgentOpinion, Case

_SIGN_PHRASES = {
    "spiculation": "spiculation",
    "pleural_indentation": "pleural indentation",
    "vascular_convergence": "vascular convergence",
    "air_bronchogram": "air bronchogram",
    "cavitation": "cavitation",
}
_MARGINS = ("smooth", "lobulated", "spiculated")
_SHAPES = ("round", "oval", "irregular")

_HARD_PENALTY = 0.20
_SOFT_PENALTY = 0.05


@dataclass
class ReportClaims:
    density: str = ""
    margin: str = ""
    shape: str = ""
    signs_present: Tuple[str, ...] = ()
    signs_absent: Tuple[str, ...] = ()


def parse_report(report: str) -> ReportClaims:
    """Extract the facts a free-text CT report asserts. Deliberately simple and
    transparent; a production verifier would consume the VLM's structured output."""
    r = (report or "").lower()
    density = next((d.replace("part-solid", "part_solid")
                    for d in ("ggo", "part-solid", "solid") if f"{d} density" in r), "")
    margin = next((m for m in _MARGINS if f"{m} margin" in r), "")
    shape = next((s for s in _SHAPES if f"{s} shape" in r), "")

    present, absent = [], []
    for key, phrase in _SIGN_PHRASES.items():
        idx = r.find(phrase)
        if idx == -1:
            continue
        sent_start = max(r.rfind(".", 0, idx), r.rfind(";", 0, idx)) + 1
        sent = r[sent_start: r.find(".", idx) + 1 or len(r)]
        if sent.lstrip().startswith("no ") or ("is present" in sent and "there is" not in sent):
            absent.append(key)
        else:
            present.append(key)
    return ReportClaims(density, margin, shape, tuple(present), tuple(absent))


def grade_claim(claim: str, claims: ReportClaims) -> Tuple[str, str]:
    """Return (verdict, reason) where verdict in {SUPPORTED, UNMENTIONED, CONTRADICTED}."""
    if ":" in claim:
        key, val = claim.split(":", 1)
        val = val.strip().lower().replace("part-solid", "part_solid")
        have = {"density": claims.density, "margin": claims.margin,
                "shape": claims.shape}.get(key, "")
        if not have:
            return "UNMENTIONED", f"report does not characterise {key}"
        return ("SUPPORTED" if have == val else "CONTRADICTED"), f"report says {key}={have}"
    if claim in _SIGN_PHRASES:
        if claim in claims.signs_present:
            return "SUPPORTED", "report asserts present"
        if claim in claims.signs_absent:
            return "CONTRADICTED", "report asserts absent"
        return "UNMENTIONED", "report does not mention this sign"
    return "SUPPORTED", "non-morphological claim; not checkable"


def verify_opinion(case: Case, op: AgentOpinion) -> Tuple[AgentOpinion, List[str]]:
    claims = parse_report(case.report or "")
    trace: List[str] = []
    kept: List[str] = []
    penalty = 0.0
    for claim in op.cited_features:
        if not claim:
            continue
        verdict, reason = grade_claim(claim, claims)
        if verdict == "SUPPORTED":
            kept.append(claim)
            trace.append(f"OK   {op.role}: {claim} ({reason})")
        elif verdict == "UNMENTIONED":
            penalty += _SOFT_PENALTY
            trace.append(f"soft {op.role}: {claim} - {reason}")
        else:
            penalty += _HARD_PENALTY
            trace.append(f"FLAG {op.role}: {claim} - {reason}")

    cleaned = AgentOpinion(
        role=op.role, probs=op.probs, rationale=op.rationale,
        cited_features=kept, confidence=max(0.05, op.confidence - penalty),
    )
    return cleaned, trace


# ---------------------------------------------------------------------------
# Evaluation only. Two distinct error sources are separated:
#   * agent fabrication : claim disagrees with BOTH the oracle and the perceived
#     features the agent was handed -> the agent invented it. The verifier
#     SHOULD catch these (recall denominator).
#   * perception error  : the report itself is wrong. Out of the verifier's
#     scope by construction; excluded from the recall denominator.
# Precision is judged against the oracle over all hard flags.
# ---------------------------------------------------------------------------
def _disagrees(claim: str, feats: Dict) -> bool:
    if ":" in claim:
        key, val = claim.split(":", 1)
        return str(feats.get(key, "")).lower() != \
            val.strip().lower().replace("part-solid", "part_solid")
    if claim in _SIGN_PHRASES:
        return not bool(feats.get(claim))
    return False


def verifier_scorecard(case: Case, raw_opinions: List[AgentOpinion]) -> Dict[str, int]:
    """Confusion counts for the verifier's HARD flags. Aggregate across a run:
        precision = tp / (tp + fp)          (flags that are genuinely wrong)
        recall    = tp / (tp + fn)          (fabrications that got flagged)
    `n_unmentioned` (soft flags) reported separately."""
    claims = parse_report(case.report or "")
    truth = case.features.as_dict()
    perceived = (case.perceived or case.features).as_dict()
    tp = fp = fn = tn = unmentioned = 0
    for op in raw_opinions:
        for claim in op.cited_features:
            if not claim:
                continue
            verdict = grade_claim(claim, claims)[0]
            if verdict == "UNMENTIONED":
                unmentioned += 1
                continue
            flagged = verdict == "CONTRADICTED"
            wrong = _disagrees(claim, truth)
            fabricated = wrong and _disagrees(claim, perceived)
            tp += int(flagged and fabricated)
            fp += int(flagged and not wrong)
            fn += int(not flagged and fabricated)
            tn += int(not flagged and not wrong)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_unmentioned": unmentioned}
