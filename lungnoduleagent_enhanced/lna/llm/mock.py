"""Deterministic mock LLM backend.

Lets the *entire* pipeline run with no API key and no network. It produces
coherent, feature-grounded outputs so that debate, devil's-advocate rebuttal,
evidence verification, calibration and abstention all exercise real code paths.

Determinism: every call seeds a numpy RandomState from the `context` (case id,
role, round). Same input -> same output, so experiments are reproducible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..clinical import feature_class_scores, softmax
from ..types import CLASSES
from .base import LLMBackend


ROLE_BIAS = {
    # each role nudges the distribution toward what that specialist over-weights
    "thoracic_radiologist": {"invasive": 0.15, "minimally_invasive": 0.05, "pre_invasive": 0.0},
    "pathologist":          {"invasive": 0.10, "minimally_invasive": 0.10, "pre_invasive": 0.0},
    "oncologist":           {"invasive": 0.25, "minimally_invasive": 0.0,  "pre_invasive": -0.05},
    "pulmonologist":        {"invasive": 0.0,  "minimally_invasive": 0.10, "pre_invasive": 0.10},
}

FEATURE_PHRASES = {
    "spiculation": "spiculation",
    "pleural_indentation": "pleural indentation",
    "vascular_convergence": "vascular convergence",
    "air_bronchogram": "an air bronchogram",
    "cavitation": "cavitation",
}


class MockBackend(LLMBackend):
    def __init__(self, seed_base: int = 0):
        self.seed_base = seed_base

    def _rng(self, context: Optional[Dict[str, Any]]) -> np.random.RandomState:
        tag = "|".join(str((context or {}).get(k, "")) for k in
                       ("case_id", "role", "round", "task"))
        h = self.seed_base
        for ch in tag:
            h = (h * 131 + ord(ch)) % (2**31 - 1)
        return np.random.RandomState(h)

    # -- free text ------------------------------------------------------------
    def complete(self, system: str, prompt: str,
                 context: Optional[Dict[str, Any]] = None) -> str:
        ctx = context or {}
        task = ctx.get("task", "")
        if task == "ct_report":
            return self._ct_report(ctx)
        if task == "summarize":
            return self._summarize(ctx)
        if task == "graph_answer":
            return self._graph_answer(ctx)
        return "[mock] " + prompt[:80]

    # -- structured -----------------------------------------------------------
    def structured(self, system: str, prompt: str, task: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx["task"] = task
        if task == "judge_nodule":
            return self._judge_nodule(ctx)
        if task == "specialist_opinion":
            return self._specialist_opinion(ctx)
        if task == "devils_advocate":
            return self._devils_advocate(ctx)
        raise ValueError(f"MockBackend has no handler for task={task!r}")

    # ---- handlers -----------------------------------------------------------
    def _judge_nodule(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Vote whether a mask candidate is a real nodule. The context carries
        `overlap` = IoU of the candidate with the true nodule; judges approve
        high-overlap candidates with high confidence."""
        rng = self._rng(ctx)
        overlap = float(ctx.get("overlap", 0.0))
        noise = rng.normal(0, 0.06)
        score = overlap + noise
        opinion = 1 if score >= 0.35 else 0
        confidence = float(np.clip(abs(score - 0.35) * 1.8 + 0.5, 0.5, 0.99))
        return {"opinion": opinion, "confidence": round(confidence, 3)}

    def _ct_report(self, ctx: Dict[str, Any]) -> str:
        f = ctx["features"]
        m = ctx.get("measurements", {})
        omit = set(ctx.get("omit", []))          # signs the report fails to mention
        all_signs = ("spiculation", "pleural_indentation", "vascular_convergence",
                     "air_bronchogram", "cavitation")
        present = [k for k in all_signs if f.get(k) and k not in omit]
        # absent signs are only asserted absent if we did not "forget" them
        absent = [k for k in all_signs if not f.get(k) and k not in omit]
        present_txt = ("There is " + ", ".join(FEATURE_PHRASES[k] for k in present) + "."
                       ) if present else ""
        absent_txt = ("No " + ", ".join(FEATURE_PHRASES[k].replace("an ", "")
                      for k in absent) + " is present."
                      ) if absent else ""
        size_txt = ""
        if m:
            size_txt = (f" The nodule measures {m['long_diameter_mm']:.1f} mm in long "
                        f"axis and {m['short_diameter_mm']:.1f} mm in short axis.")
        return (
            f"A {f['density'].replace('_', '-')} density nodule with a {f['shape']} shape "
            f"and {f['margin']} margin is located in the {f['lobe']}."
            f"{size_txt} {present_txt} {absent_txt}"
        ).strip()

    def _retrieved_facts(self, f: Dict) -> List[str]:
        facts = []
        if f.get("margin") == "spiculated" or f.get("spiculation"):
            facts.append("Spiculated margins strongly correlate with invasive adenocarcinoma.")
        if f.get("density") == "ggo":
            facts.append("Pure ground-glass nodules are frequently pre-invasive lesions (AIS/AAH).")
        if f.get("density") == "part_solid":
            facts.append("Part-solid nodules carry the highest malignancy risk per Fleischner guidance.")
        if f.get("pleural_indentation"):
            facts.append("Pleural indentation is an established sign of invasive growth.")
        if not facts:
            facts.append("Smooth-margined small solid nodules are commonly benign or pre-invasive.")
        return facts

    def _specialist_opinion(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        rng = self._rng(ctx)
        f: Dict = ctx["features_perceived"]        # what the agent 'sees' (report-derived)
        role: str = ctx["role"]
        round_i: int = int(ctx.get("round", 1))

        base = feature_class_scores(f)
        bias = ROLE_BIAS.get(role, {})
        for k in base:
            base[k] += bias.get(k, 0.0)
        # noise shrinks with rounds (agents converge as they see peers)
        noise_scale = 0.6 / round_i
        for k in base:
            base[k] += rng.normal(0, noise_scale)

        # In later rounds, pull toward the peer summary label if provided.
        peer_label = ctx.get("peer_label")
        if peer_label and round_i > 1:
            base[peer_label] += 0.8

        probs = softmax(base, temperature=1.0)

        # cite the features actually supporting the top class
        cited = self._cite(f)
        # deterministic occasional hallucination so the Verifier has work to do
        if rng.rand() < 0.18:
            fake = rng.choice(["cavitation", "air_bronchogram", "vascular_convergence"])
            if fake not in cited:
                cited.append(fake)  # a claim not supported by the true features

        top = max(probs, key=probs.get)
        rationale = self._rationale(role, top, cited)
        confidence = float(np.clip(max(probs.values()) + rng.normal(0, 0.03), 0.3, 0.99))
        return {
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "rationale": rationale,
            "cited_features": cited,
            "confidence": round(confidence, 3),
        }

    def _devils_advocate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Argue against the board's current leading label."""
        rng = self._rng(ctx)
        leading: str = ctx["leading_label"]
        f: Dict = ctx["features_perceived"]
        base = feature_class_scores(f)
        # invert: boost the strongest non-leading class
        alt = max((c for c in CLASSES if c != leading), key=lambda c: base[c])
        boosted = {c: base[c] for c in CLASSES}
        boosted[alt] += 1.5 + abs(rng.normal(0, 0.2))
        probs = softmax(boosted, temperature=1.0)
        rationale = (f"Counter-argument: the evidence for {leading.replace('_',' ')} is not "
                     f"decisive. One can construct a case for {alt.replace('_',' ')} - "
                     f"consider whether the reported signs are truly present and reproducible.")
        return {"probs": {k: round(v, 4) for k, v in probs.items()},
                "challenged_label": leading, "alternative": alt, "rationale": rationale}

    def _summarize(self, ctx: Dict[str, Any]) -> str:
        opinions = ctx.get("opinions", [])
        labels = [o["label"] for o in opinions]
        if not labels:
            return "No opinions to summarize."
        from collections import Counter
        c = Counter(labels)
        lead, n = c.most_common(1)[0]
        return (f"{n}/{len(labels)} agents favor {lead.replace('_',' ')}. "
                f"Distribution: {dict(c)}.")

    def _graph_answer(self, ctx: Dict[str, Any]) -> str:
        facts = self._retrieved_facts(ctx.get("features", {}))
        return " ".join(facts)

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _cite(f: Dict) -> List[str]:
        keys = ["spiculation", "pleural_indentation", "vascular_convergence",
                "air_bronchogram", "cavitation"]
        cited = [k for k in keys if f.get(k)]
        cited.append(f"density:{f.get('density')}")
        cited.append(f"margin:{f.get('margin')}")
        return cited

    @staticmethod
    def _rationale(role: str, label: str, cited: List[str]) -> str:
        pretty = label.replace("_", " ")
        feats = ", ".join(c for c in cited if ":" not in c) or "the overall morphology"
        return (f"As the {role.replace('_', ' ')}, I favor {pretty} based on {feats}.")
