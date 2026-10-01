# SafeNodule-MAS

**A Reproducible, Cost-Aware, and Safety-Calibrated Multi-Agent Architecture for Lung Nodule Diagnosis**

Reference implementation, evaluation harness, and paper sources for SafeNodule-MAS
(LungNoduleAgent-Enhanced), a multi-agent clinical decision-support architecture that
extends the three-stage pipeline of *LungNoduleAgent* ([arXiv:2511.21042](https://arxiv.org/abs/2511.21042))
with seven instrumented safety and cost mechanisms.

The point of this repository is not a higher accuracy number. It is that **every added
mechanism is paired with an explicit, measurable endpoint**, evaluated under controlled
stress with confidence intervals and paired significance tests, including the regimes
where a mechanism *fails*.

> Runs **out of the box with no API keys and no datasets.** A deterministic mock language
> backend and a synthetic CT generator make the whole pipeline executable and scorable.
> Swap in a real VLM and a DICOM loader at the documented interfaces; nothing else changes.

---

## What is in here

| Path | Contents |
|---|---|
| `lungnoduleagent_enhanced/` | The implementation: agents, evaluation harness, tests. See its own [README](lungnoduleagent_enhanced/README.md) for the full API and architecture walkthrough. |
| `lungnoduleagent_enhanced/results/` | `experiments.json`, CSV tables, and figures produced by the full run reported in the paper. |
| `paper/` | LaTeX source, `references.bib`, the figure generators, and the rendered figures. |
| `build_paper_tex.py` | Builds `paper/enhanced_lungnoduleagent.tex` from `results/experiments.json`. Every number in the paper is read from the results file, so the text can never drift from the data. |
| `build_article.py` | Converts that LaTeX into the Word manuscript. |

---

## Quick start

Python 3.9+. The only hard dependency is numpy.

```bash
git clone https://github.com/LochanNarayan/Lungnodule-agent
cd Lungnodule-agent/lungnoduleagent_enhanced
python -m pip install -r requirements.txt
```

Run a single case end to end, with the full reasoning trace:

```bash
python -m lna.cli demo
```

Run the test suite (41 tests):

```bash
python -m pytest -q
```

---

## Reproducing the paper

From `lungnoduleagent_enhanced/`:

```bash
python -m lna.cli experiments --n 64 --seeds 8 --boot 1000 --out results
```

This runs all five stress regimes x all ablation variants, with multi-seed
mean +/- standard deviation and paired bootstrap significance tests, and writes
`results/experiments.json`, `results/*.csv`, and `results/figures/*.png`.

Then, from the repository root, regenerate the manuscript from those results:

```bash
python build_paper_tex.py     # -> paper/enhanced_lungnoduleagent.tex
python build_article.py       # -> output/Enhanced_LungNoduleAgent_Article.docx
```

---

## The seven mechanisms and their endpoints

| Mechanism | Endpoint it is judged on |
|---|---|
| Adaptive triage router | mean language-model calls per case; under-triage rate |
| Role-differentiated board | accuracy and macro-F1 vs. a single agent (paired test) |
| Report-grounded evidence verifier | hard-flag precision and recall vs. the oracle |
| Devil's-Advocate false-negative guard | false-negative rate vs. the board without it |
| Lung-RADS guideline agent | escalation correctness (deterministic unit tests) |
| Fitted temperature scaling | ECE at T=1 vs. fitted T; the fitted T per regime |
| Margin abstention gate | risk-coverage AURC; abstention value |

## Measured results

Seed means over 8 seeds, 64 cases per seed, with paired bootstrap tests:

| Finding | Value |
|---|---|
| Board vs. single agent (borderline) | accuracy +0.161 |
| Adaptive routing | 2 to 4% fewer calls per case, significant (p < 0.001) in every regime |
| Verifier fabrication recall | 1.00 across all regimes |
| Calibration under adversarial shift | ECE 0.540 -> 0.217 (-60%) |
| Devil's-Advocate guard (borderline) | false-negative rate 0.082 -> 0.019 |

Two results are reported as **negative findings** rather than hidden: under a
benign-heavy class prior a single unbiased agent beats the board, and the
confidence-based abstention gate has negative value under class imbalance and
under the adversarial regime, where the confidence signal stops being informative.

---

## Scope

This is a transparent reference architecture, not a clinical device. No patient data
are used. Absolute accuracy on synthetic data says nothing about clinical performance;
the transferable contributions are the architecture, the per-component endpoints, and
the evaluation protocol. See the production checklist in the
[package README](lungnoduleagent_enhanced/README.md) for the real-data and real-model
swap points.

## License

MIT. See [LICENSE](LICENSE).
