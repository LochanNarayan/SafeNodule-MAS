"""Minimal programmatic example.

Run from the project root:
    python examples/run_demo.py
"""
import os
import sys

# make `lna` importable when run as a script from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lna.config import Config
from lna.data import make_dataset
from lna.pipeline import LungNoduleAgent


def main():
    # a harder regime so the safety machinery actually does something
    cfg = Config.for_regime("borderline", seed=7)
    agent = LungNoduleAgent(cfg)
    cases = make_dataset(30, seed=3, config=cfg)

    # hold out 10 cases to fit the calibration temperature
    t = agent.fit_calibration(cases[:10])
    print(f"fitted temperature: {t:.2f}")

    res = agent.diagnose(cases[10])
    print(res.summary())
    print("report      :", res.report)
    print("guideline   :", res.guideline)
    print("calibrated  :", {k: round(v, 3) for k, v in res.calibrated_probs.items()})
    print("raw (pre-T) :", {k: round(v, 3) for k, v in res.raw_probs.items()})
    print("LLM calls   :", res.n_llm_calls, res.llm_call_breakdown)
    print("verifier    :", res.verifier_counts, "hard flags:", res.n_hard_flags)
    print("abstained   :", res.abstained, "| under-triaged:", res.under_triaged)


if __name__ == "__main__":
    main()
