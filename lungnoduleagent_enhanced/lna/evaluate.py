"""Evaluation: accuracy / macro-F1 / coverage plus the safety, calibration and
cost metrics that make each enhanced component measurable, and the multi-seed +
leave-one-agent-out machinery used for the paper's tables.
"""
from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .data import make_dataset
from .doctors.calibration import fit_temperature, temperature_scale
from .pipeline import LungNoduleAgent, PipelineResult
from .types import CLASSES, Case

_MALIGNANT = ("minimally_invasive", "invasive")


# ===========================================================================
# core + safety metrics
# ===========================================================================
@dataclass
class Metrics:
    n: int
    accuracy: float            # over answered (non-abstained) cases
    accuracy_all: float        # abstentions counted wrong
    macro_f1: float
    coverage: float            # fraction answered
    mean_detection_iou: float
    mean_rounds: float
    # safety
    false_negative_rate: float # truly-malignant nodules answered as pre_invasive
    abstention_precision: float # fraction of abstained cases that WOULD have been wrong
    abstention_value: float     # err(abstained) - err(answered); >0 = gate is useful
    aurc: float                # area under risk-coverage curve (lower = better)
    ece: float                 # ECE of the deployed fixed-temperature probs
    ece_raw: float             # ECE of the pre-temperature-scaling aggregate (T=1)
    ece_fitted: float          # ECE after in-test 5-fold fitted temperature scaling
    fitted_temperature: float  # temperature actually deployed (fitted on the split)
    brier: float
    brier_raw: float
    over_confidence: float     # mean(conf) - accuracy on answered  (>0 = overconfident)
    # verifier
    verifier_precision: float
    verifier_recall: float
    mean_hard_flags: float
    # cost
    mean_llm_calls: float
    mean_prompt_tokens: float
    mean_total_tokens: float
    frac_routed_easy: float
    under_triage_rate: float

    def as_row(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def __str__(self) -> str:
        return (f"n={self.n} acc(ans)={self.accuracy:.3f} acc(all)={self.accuracy_all:.3f} "
                f"macroF1={self.macro_f1:.3f} cov={self.coverage:.3f} "
                f"FNR={self.false_negative_rate:.3f} AURC={self.aurc:.3f} "
                f"ECE={self.ece:.3f}(raw {self.ece_raw:.3f}) "
                f"vP/R={self.verifier_precision:.2f}/{self.verifier_recall:.2f} "
                f"calls={self.mean_llm_calls:.1f} underTri={self.under_triage_rate:.3f}")


def _macro_f1(results: List[PipelineResult]) -> float:
    f1s = []
    for c in CLASSES:
        tp = sum(1 for r in results if r.predicted_label == c and r.true_label == c)
        fp = sum(1 for r in results if r.predicted_label == c and r.true_label != c)
        fn = sum(1 for r in results if r.predicted_label != c and r.true_label == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def _ece(pairs: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    """pairs = [(top_prob, correct), ...] over answered cases."""
    if not pairs:
        return 0.0
    tot = len(pairs)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(p, c) for p, c in pairs if (p > lo and p <= hi) or (b == 0 and p <= hi)]
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(1 for _, c in bucket if c) / len(bucket)
        e += (len(bucket) / tot) * abs(conf - acc)
    return e


def _brier(results: List[PipelineResult], key: str) -> float:
    """Multiclass Brier score over answered cases using probs at `key`."""
    ans = [r for r in results if not r.abstained and getattr(r, key)]
    if not ans:
        return 0.0
    s = 0.0
    for r in ans:
        probs = getattr(r, key)
        for c in CLASSES:
            y = 1.0 if r.true_label == c else 0.0
            s += (probs.get(c, 0.0) - y) ** 2
    return s / len(ans)


def _kfold_fitted_ece(results: List[PipelineResult], k: int = 5
                      ) -> Tuple[float, float]:
    """5-fold temperature scaling on the raw aggregate distributions of answered
    cases: fit T on k-1 folds, apply to the held-out fold, pool, then ECE.
    Returns (ece_fitted, mean_fitted_T)."""
    ans = [r for r in results if not r.abstained and r.raw_probs]
    if len(ans) < k:
        return _ece([(max(r.raw_probs.values()), r.raw_probs and
                      max(r.raw_probs, key=r.raw_probs.get) == r.true_label)
                     for r in ans]), 1.0
    folds = [ans[i::k] for i in range(k)]
    pooled: List[Tuple[float, bool]] = []
    temps: List[float] = []
    for i in range(k):
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        samples = [(r.raw_probs, r.true_label) for r in train]
        t = fit_temperature(samples)
        temps.append(t)
        for r in folds[i]:
            sc = temperature_scale(r.raw_probs, t)
            lab = max(sc, key=sc.get)
            pooled.append((sc[lab], lab == r.true_label))
    return _ece(pooled), statistics.fmean(temps)


def _aurc(results: List[PipelineResult]) -> float:
    """Area under the risk-coverage curve over the ANSWERED cases, ranked by
    confidence (most confident first). This is the standard selective-classifier
    AURC: it measures how well the confidence signal orders correctness. Lower is
    better. Coverage (fraction answered) is reported separately."""
    answered = sorted((r for r in results if not r.abstained),
                      key=lambda r: r.confidence, reverse=True)
    n = len(answered)
    if n == 0:
        return 1.0
    risks, cum_err = [], 0
    for i, r in enumerate(answered, 1):
        cum_err += 0 if r.correct else 1
        risks.append(cum_err / i)
    return sum(risks) / n


def compute_metrics(results: List[PipelineResult]) -> Metrics:
    n = len(results)
    answered = [r for r in results if not r.abstained]
    acc = sum(r.correct for r in answered) / len(answered) if answered else 0.0
    acc_all = sum(r.correct for r in results) / n if n else 0.0

    mal = [r for r in results if r.true_malignant]
    fn = sum(1 for r in mal if (not r.abstained) and r.predicted_label == "pre_invasive")
    fnr = fn / len(mal) if mal else 0.0

    abst = [r for r in results if r.abstained]
    would_wrong = sum(1 for r in abst if r.predicted_label != r.true_label)
    abst_prec = would_wrong / len(abst) if abst else 0.0
    err_answered = 1.0 - acc
    abst_value = (would_wrong / len(abst) - err_answered) if abst else 0.0

    cal_pairs = [(r.confidence, r.correct) for r in answered]
    raw_pairs = [(max(r.raw_probs.values()) if r.raw_probs else r.confidence,
                  (max(r.raw_probs, key=r.raw_probs.get) == r.true_label)
                  if r.raw_probs else r.correct)
                 for r in answered]

    vc = {k: sum(r.verifier_counts.get(k, 0) for r in results)
          for k in ("tp", "fp", "fn", "tn", "n_unmentioned")}
    v_prec = vc["tp"] / (vc["tp"] + vc["fp"]) if (vc["tp"] + vc["fp"]) else 0.0
    v_rec = vc["tp"] / (vc["tp"] + vc["fn"]) if (vc["tp"] + vc["fn"]) else 0.0

    mean_conf = statistics.fmean(r.confidence for r in answered) if answered else 0.0
    ece_fitted, _ = _kfold_fitted_ece(results)
    deployed_t = statistics.fmean(r.temperature_used for r in results) if n else 1.0

    return Metrics(
        n=n, accuracy=acc, accuracy_all=acc_all,
        macro_f1=_macro_f1(answered), coverage=len(answered) / n if n else 0.0,
        mean_detection_iou=statistics.fmean(r.detection_iou for r in results) if n else 0.0,
        mean_rounds=statistics.fmean(r.rounds_used for r in results) if n else 0.0,
        false_negative_rate=fnr,
        abstention_precision=abst_prec,
        abstention_value=abst_value,
        aurc=_aurc(results),
        ece=_ece(cal_pairs), ece_raw=_ece(raw_pairs),
        ece_fitted=ece_fitted, fitted_temperature=deployed_t,
        brier=_brier(results, "calibrated_probs"), brier_raw=_brier(results, "raw_probs"),
        over_confidence=mean_conf - acc,
        verifier_precision=v_prec, verifier_recall=v_rec,
        mean_hard_flags=statistics.fmean(r.n_hard_flags for r in results) if n else 0.0,
        mean_llm_calls=statistics.fmean(r.n_llm_calls for r in results) if n else 0.0,
        mean_prompt_tokens=statistics.fmean(r.n_prompt_tokens for r in results) if n else 0.0,
        mean_total_tokens=statistics.fmean(r.n_total_tokens for r in results) if n else 0.0,
        frac_routed_easy=sum(1 for r in results if r.routing == "easy") / n if n else 0.0,
        under_triage_rate=sum(r.under_triaged for r in results) / n if n else 0.0,
    )


# ===========================================================================
# multi-seed aggregation
# ===========================================================================
@dataclass
class Aggregate:
    seeds: List[int]
    mean: Dict[str, float]
    std: Dict[str, float]
    per_seed: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def ci95(self, key: str) -> Tuple[float, float]:
        m, s, k = self.mean[key], self.std[key], len(self.seeds)
        half = 1.96 * s / math.sqrt(k) if k > 1 else 0.0
        return m - half, m + half

    def __str__(self) -> str:
        return " ".join(f"{k}={self.mean[k]:.3f}±{self.std[k]:.3f}"
                        for k in ("accuracy_all", "macro_f1", "coverage",
                                  "false_negative_rate", "aurc", "ece",
                                  "mean_llm_calls", "mean_prompt_tokens",
                                  "mean_total_tokens"))


def _split_fit_eval(agent: LungNoduleAgent, cases: List[Case],
                    frac: float) -> List[PipelineResult]:
    """Hold out the first `frac` of cases as a calibration split (used to fit the
    temperature when calibration is on), evaluate on the remainder. The split is
    taken for every variant so all are scored on the same test cases."""
    if not frac or frac <= 0:
        return agent.diagnose_many([copy.deepcopy(c) for c in cases])
    k = max(2, int(round(len(cases) * frac)))
    if agent.config.use_calibration:
        agent.fit_calibration([copy.deepcopy(c) for c in cases[:k]])
    return agent.diagnose_many([copy.deepcopy(c) for c in cases[k:]])


def run_multiseed(base_config: Config, n: int, seeds: List[int],
                  data_seed_offset: int = 1000) -> Aggregate:
    rows: Dict[int, Dict[str, float]] = {}
    for s in seeds:
        cfg = copy.deepcopy(base_config)
        cfg.seed = s
        agent = LungNoduleAgent(cfg)
        cases = make_dataset(n, seed=data_seed_offset + s * 97, config=cfg)
        rows[s] = compute_metrics(
            _split_fit_eval(agent, cases, cfg.calibration_split_frac)).as_row()
    keys = list(next(iter(rows.values())).keys())
    mean = {k: statistics.fmean(rows[s][k] for s in seeds) for k in keys}
    std = {k: (statistics.pstdev(rows[s][k] for s in seeds) if len(seeds) > 1 else 0.0)
           for k in keys}
    return Aggregate(seeds=list(seeds), mean=mean, std=std, per_seed=rows)


# ===========================================================================
# leave-one-component-out ablation (multi-seed, any regime)
# ===========================================================================
def ablation_variants(base: Config) -> Dict[str, Config]:
    v: Dict[str, Config] = {}

    def mk(**kw) -> Config:
        c = copy.deepcopy(base)
        for k, val in kw.items():
            setattr(c, k, val)
        return c

    v["full"] = copy.deepcopy(base)
    v["- verifier"] = mk(use_verifier=False)
    v["- devils_advocate"] = mk(use_devils_advocate=False)
    v["- guideline"] = mk(use_guideline_agent=False)
    v["- calibration"] = mk(use_calibration=False)
    v["- abstention"] = mk(abstain_margin_below=0.0, abstain_confidence_below=0.0,
                           abstain_entropy_above=9.9, abstain_disagreement_below=0.0,
                           devil_fn_guard_prob=9.9)
    v["- router"] = mk(use_router=False)
    v["- board (1 agent)"] = mk(board_roles=base.board_roles[:1])
    v["- debate (1 round)"] = mk(max_debate_rounds=1)
    return v


def run_ablation_multiseed(base_config: Config, n: int, seeds: List[int]
                           ) -> Dict[str, Aggregate]:
    out: Dict[str, Aggregate] = {}
    for name, cfg in ablation_variants(base_config).items():
        out[name] = run_multiseed(cfg, n, seeds)
    return out


# ===========================================================================
# paired significance: full system vs a variant, bootstrap over cases x seeds
# ===========================================================================
def paired_bootstrap(config_a: Config, config_b: Config, n: int, seeds: List[int],
                     metric: Callable[[List[PipelineResult]], float],
                     n_boot: int = 2000, data_seed_offset: int = 1000
                     ) -> Dict[str, float]:
    """Returns {delta, p_value, lo, hi} for metric(A) - metric(B), where the two
    systems are run on the *same* cases per seed (a genuine paired comparison)."""
    diffs: List[float] = []
    per_case: List[Tuple[PipelineResult, PipelineResult]] = []
    for s in seeds:
        ca, cb = copy.deepcopy(config_a), copy.deepcopy(config_b)
        ca.seed = cb.seed = s
        cases = make_dataset(n, seed=data_seed_offset + s * 97, config=ca)
        ra = _split_fit_eval(LungNoduleAgent(ca), cases, ca.calibration_split_frac)
        rb = _split_fit_eval(LungNoduleAgent(cb), cases, cb.calibration_split_frac)
        diffs.append(metric(ra) - metric(rb))
        per_case.extend(zip(ra, rb))

    obs = statistics.fmean(diffs)
    rng = np.random.RandomState(12345)
    a_res = [p[0] for p in per_case]
    b_res = [p[1] for p in per_case]
    m = len(per_case)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, m, m)
        boot[i] = metric([a_res[j] for j in idx]) - metric([b_res[j] for j in idx])
    # two-sided p-value against H0: delta == 0, via the bootstrap distribution
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": obs, "p_value": float(min(1.0, p)),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


# handy metric closures for the bootstrap
def m_acc_all(rs: List[PipelineResult]) -> float:
    return sum(r.correct for r in rs) / len(rs) if rs else 0.0


def m_fnr(rs: List[PipelineResult]) -> float:
    mal = [r for r in rs if r.true_malignant]
    return sum(1 for r in mal if not r.abstained and r.predicted_label == "pre_invasive") \
        / len(mal) if mal else 0.0


def m_aurc(rs: List[PipelineResult]) -> float:
    return _aurc(rs)


def m_calls(rs: List[PipelineResult]) -> float:
    return statistics.fmean(r.n_llm_calls for r in rs) if rs else 0.0


def m_ece(rs: List[PipelineResult]) -> float:
    ans = [r for r in rs if not r.abstained]
    return _ece([(r.confidence, r.correct) for r in ans])


# ---- kept for backwards compatibility with the original CLI/tests ----
def run_ablation(cases: List[Case], base_config: Config) -> Dict[str, Metrics]:
    out: Dict[str, Metrics] = {}
    for name, cfg in ablation_variants(base_config).items():
        agent = LungNoduleAgent(cfg)
        res = agent.diagnose_many([copy.deepcopy(c) for c in cases])
        out[name] = compute_metrics(res)
    return out
