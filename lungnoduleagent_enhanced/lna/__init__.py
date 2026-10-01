"""
LungNoduleAgent-Enhanced
========================

A runnable reference implementation of the *enhanced* collaborative multi-agent
architecture for lung-nodule diagnosis (detect -> describe -> diagnose), adding:

  * Triage / Router agent          (adaptive cost)
  * Role-differentiated doctor board (radiologist / pathologist / oncologist / pulmonologist)
  * Devil's-Advocate agent          (anti-groupthink, anti-false-negative)
  * Evidence Verifier agent         (anti-hallucination grounding)
  * Guideline agent                 (Lung-RADS / Fleischner mapping)
  * Calibration + Abstention agent  (safe uncertainty, human-in-the-loop)
  * Chair / Moderator agent         (explicit debate convergence + escalation)
  * Hierarchical memory             (working / episodic / semantic)

The truly algorithmic pieces (IoU mask clustering via DBSCAN, confidence- and
role-weighted voting, debate convergence detection, calibration, abstention)
are implemented for real in numpy. The language pieces run through a pluggable
LLM backend that defaults to a deterministic mock so the whole pipeline executes
with no API keys and no datasets.
"""

__version__ = "0.1.0"
