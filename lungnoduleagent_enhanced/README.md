# LungNoduleAgent-Enhanced

A runnable reference implementation of an **enhanced** collaborative multi-agent
system for lung-nodule diagnosis, extending the architecture in *LungNoduleAgent*
(arXiv:2511.21042). It keeps the paper's three-stage clinical pipeline
(**detect → describe → diagnose**) and adds the agents and safety mechanisms the
original lacked.

> It runs **out of the box with no API keys and no datasets** — a deterministic
> mock LLM and a synthetic CT generator let the whole pipeline execute and be
> scored. Swap in a real Anthropic/OpenAI backend and a real DICOM loader for
> production; nothing else changes.

---

## What's new vs. the paper

| Added component | Type | Problem it fixes |
|---|---|---|
| **Triage / Router agent** | orchestration | cost — easy cases skip the full board |
| **Role-differentiated board** (radiologist / pathologist / oncologist / pulmonologist) | diagnosis | generic "Doctor 1/2" can't disagree usefully |
| **Devil's-Advocate agent** | safety | groupthink & false negatives |
| **Evidence Verifier agent** | safety | hallucinated findings |
| **Guideline agent** (Lung-RADS / Fleischner) | grounding | "evidence-based" on a black-box KB |
| **Calibration + Abstention agent** | safety | overconfidence; adds human-in-the-loop |
| **Chair / Moderator agent** | orchestration | vague "iterate until consensus" |
| **Hierarchical Memory** (working/episodic/semantic) | infra | flat memory, no case recall |
| **Role- & confidence-weighted voting** | aggregation | equal votes ignore expertise |

Every added mechanism is paired with an **explicit success metric** (see
*Evaluation* below) rather than an accuracy claim:

| Mechanism | Success metric |
|---|---|
| Triage router | mean LLM calls/case; under-triage rate |
| Role-differentiated board | accuracy / macro-F1 vs. single agent (paired bootstrap) |
| Evidence verifier | hard-flag precision / recall vs. oracle |
| Devil's-Advocate FN-guard | false-negative rate vs. board without it |
| Guideline agent | Lung-RADS escalation correctness (unit tests) |
| Fitted temperature scaling | ECE before/after; fitted T per regime |
| Margin abstention gate | risk–coverage AURC; abstention value |

The algorithmic pieces are implemented **for real** in numpy: IoU mask distance +
DBSCAN clustering (paper Eq. 1–3), confidence-weighted judging-panel voting
(Eq. 4–5), debate convergence detection, NLL-fitted temperature scaling, and a
margin-based abstention gate with a structural false-negative guard.

---

## Architecture / data flow

```
3D CT slice
   │
   ▼
[1] NODULE SPOTTER
     MoE experts ─▶ IoU+DBSCAN clustering ─▶ Judging Panel (weighted vote) ─▶ Refiner ─▶ measurements
   │  (lna/spotter/*)
   ▼
[2] SIMULATED RADIOLOGIST
     focal crop ─▶ MedPrompt ─▶ localized CT report      (lna/radiologist/*)
   │
   ▼
[0] TRIAGE ROUTER   easy → 1 specialist / 1 round | hard → full board / N rounds   (lna/router.py)
   │
   ▼
[3] DOCTOR BOARD    (lna/doctors/*)
     specialists ─▶ Evidence Verifier ─▶ Devil's Advocate ─▶ Chair(convergence?)
                        └── loop ──┘
     ─▶ Calibration + Abstention ─▶ Guideline agent
   │
   ▼
Final: label + calibrated probs + confidence + guideline + evidence trace
        (or ABSTAIN → escalate to human)

Hierarchical Memory (lna/memory) underlies every stage.
Medical Graph RAG (lna/knowledge) feeds each specialist role-specific knowledge.
```

---

## Install

Requires **Python 3.9+**. The only hard dependency is numpy.

```bash
cd lungnoduleagent_enhanced
python -m pip install -r requirements.txt
```

(Optional, for the test suite: `pip install pytest`. For real LLMs:
`pip install anthropic` or `pip install openai`.)

---

## Run it

All commands are run from the **project root** (the folder containing `lna/`).

### 1. Single case, full trace
```bash
python -m lna.cli demo
```
Shows detection IoU, the generated CT report, the routing decision, the debate
rounds, the verifier trace, the guideline assessment, and the calibrated final
diagnosis.

### 2. Batch run with metrics
```bash
python -m lna.cli run --n 20
```
Prints a per-case line and overall metrics: accuracy over answered cases,
accuracy counting abstentions as wrong, macro-F1, abstention coverage, mean
detection IoU, and mean debate rounds.

### 3. Leave-one-component-out ablation (multi-seed)
```bash
python -m lna.cli ablation --n 40 --seeds 5 --regime borderline
```
Disables one component at a time, averaged over seeds, and prints the full metric
block (acc, macro-F1, coverage, FNR, AURC, ECE, verifier P/R, calls) per variant.

### 3b. Evaluation regimes

`--regime {clean,noisy,borderline,imbalanced,adversarial}` (on `demo`, `run`,
`ablation`) controls how the synthetic cases are generated so the safety/cost
machinery is actually *stressed*:

| regime | what it does |
|---|---|
| `clean` | perceived morphology == oracle; balanced classes |
| `noisy` | perceived features + report corrupted (verifier / calibration have work) |
| `borderline` | only cases near the class decision boundary (abstention) |
| `imbalanced` | class prior skewed toward pre-invasive (macro-F1 stress) |
| `adversarial` | noise + borderline + degraded detector combined |

### 3c. Full paper matrix
```bash
python -m lna.cli experiments --n 64 --seeds 8 --boot 1000 --out results
```
Runs every regime × ablation with multi-seed mean±sd, paired bootstrap
significance tests, and writes `results/experiments.json`, `results/*.csv`, and
`results/figures/*.png`. `pip install matplotlib` for the figures.

### 4. Programmatic use
```bash
python examples/run_demo.py
```
```python
from lna.config import Config
from lna.data import make_case
from lna.pipeline import LungNoduleAgent

agent = LungNoduleAgent(Config(backend="mock", seed=7))
result = agent.diagnose(make_case("my_case", seed=3))
print(result.summary())
```

### 5. Run the tests
```bash
python -m pytest -q
```
Covers clustering / voting, the five data regimes, report-grounded verification
of injected hallucinations, NLL temperature fitting, the abstention gate and
false-negative guard, LLM-call accounting, guideline escalation, and the paired
bootstrap.

---

## Evaluation metrics

`lna/evaluate.py` reports, per run:

* **accuracy** (answered), **accuracy_all** (abstentions = wrong), **macro-F1**, **coverage**
* **false_negative_rate** — truly-malignant nodule answered as `pre_invasive`
* **aurc** — area under the risk–coverage curve over answered cases (lower better)
* **abstention_precision / abstention_value** — is the gate withholding the harder cases?
* **ece / ece_raw / fitted_temperature** — calibration before/after NLL-fitted temperature scaling
* **verifier_precision / verifier_recall** — hard-flag quality vs. the oracle (recall over agent *fabrications*)
* **mean_llm_calls** — every detector/judge/specialist/critique/summary call, via `CountingBackend`
* **under_triage_rate** — genuinely hard cases the router sent down the easy path

`run_multiseed`, `run_ablation_multiseed`, and `paired_bootstrap` add seed
mean±sd, 95% CIs, and two-sided p-values.

---

## Using a real LLM backend

```bash
# Anthropic
export ANTHROPIC_API_KEY=...          # PowerShell: $env:ANTHROPIC_API_KEY="..."
python -m lna.cli run --n 5 --backend anthropic --model claude-sonnet-5

# OpenAI
export OPENAI_API_KEY=...
python -m lna.cli run --n 5 --backend openai --model gpt-4o
```
The agents call `backend.structured(...)` / `backend.complete(...)`; the real
backends ask the model for strict JSON and parse it. To make doctors reason over
the **actual nodule image**, attach the focal-cropped image as an image content
block in `lna/llm/anthropic_backend.py` (marked in the file).

---

## Repository layout

```
lna/
  config.py            all tunable hyperparameters (DBSCAN eps, #agents, #rounds, thresholds)
  types.py             Case / features / opinions / outcome dataclasses
  clinical.py          transparent feature→class heuristic (shared by data gen + mock LLM)
  pipeline.py          end-to-end orchestrator  (LungNoduleAgent)
  router.py            triage / router agent
  evaluate.py          metrics + leave-one-agent-out ablation
  cli.py               `python -m lna.cli {demo,run,ablation}`
  llm/                 backend interface + mock / anthropic / openai
  data/                synthetic CT + nodule generator  (swap for LIDC/DICOM loader)
  spotter/             MoE, IoU+DBSCAN clustering, judging panel, refiner
  radiologist/         focal crop + MedPrompt report generation
  knowledge/           Medical Graph RAG + kb/ markdown knowledge base
  doctors/             specialists, devil's advocate, verifier, guideline, calibration, chair, board
  memory/              hierarchical memory
tests/                 pytest suite (clustering, voting, pipeline determinism/accuracy)
examples/              run_demo.py
```

---

## Swapping in real data & models (production checklist)

1. **Data:** replace `lna/data/synthetic.py` with a LIDC-IDRI / DICOM loader that
   yields `Case` objects (image slice + true mask + features + label). Everything
   downstream is unchanged.
2. **Detectors:** replace `lna/spotter/moe.py:run` with real detector inference
   (nnUNet, MONAI, YOLO). It only needs to return a list of binary masks.
3. **VLM:** set `--backend anthropic|openai` and attach the cropped image to the
   report + doctor prompts.
4. **Knowledge base:** drop real pathology docs into `lna/knowledge/kb/` (split by
   `##` headings) or point `MedicalGraphRAG` at a real vector/graph store.
5. **Verifier oracle:** in production the verifier grounds claims against the CT
   report + measurements + knowledge graph rather than the synthetic oracle.

---

## Note on scope

This is a faithful, transparent **reference architecture**, not a clinical device.
The synthetic data exists so the control flow, agent interactions, and evaluation
harness are fully runnable and reproducible. Accuracy numbers on synthetic data
only demonstrate that the machinery works — they say nothing about real clinical
performance, which requires the real data/model swaps above plus the validation
described in the paper critique (radiologist review, patient-level splits,
significance testing).
