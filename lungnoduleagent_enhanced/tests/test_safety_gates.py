"""Abstention gate and Devil's-Advocate false-negative guard must measurably act."""
import copy

from lna.config import Config
from lna.data import make_dataset
from lna.evaluate import compute_metrics, run_multiseed
from lna.pipeline import LungNoduleAgent


def test_abstention_gate_fires_under_borderline_but_not_clean():
    clean = compute_metrics(LungNoduleAgent(Config()).diagnose_many(
        make_dataset(40, seed=1, config=Config())))
    bcfg = Config.for_regime("borderline")
    bord = compute_metrics(LungNoduleAgent(bcfg).diagnose_many(
        make_dataset(40, seed=1, config=bcfg)))
    assert bord.coverage < clean.coverage
    assert bord.coverage < 1.0


def test_abstention_has_positive_value_when_confidence_is_informative():
    """On clean/borderline data the abstained cases should be more error-prone
    than the answered ones (positive abstention value)."""
    for regime in ("clean", "borderline"):
        cfg = Config() if regime == "clean" else Config.for_regime(regime)
        agg = run_multiseed(cfg, 60, list(range(1, 6)))
        assert agg.mean["abstention_value"] > 0.0


def test_devils_advocate_guard_reduces_false_negatives():
    seeds = list(range(1, 7))
    base = Config.for_regime("borderline")
    no_devil = copy.deepcopy(base); no_devil.use_devils_advocate = False
    with_guard = run_multiseed(base, 60, seeds).mean["false_negative_rate"]
    without = run_multiseed(no_devil, 60, seeds).mean["false_negative_rate"]
    assert with_guard < without


def test_removing_abstention_raises_coverage_to_one():
    cfg = Config.for_regime("borderline")
    cfg.abstain_margin_below = 0.0
    cfg.abstain_confidence_below = 0.0
    cfg.abstain_entropy_above = 9.9
    cfg.devil_fn_guard_prob = 9.9
    m = compute_metrics(LungNoduleAgent(cfg).diagnose_many(
        make_dataset(40, seed=2, config=cfg)))
    assert m.coverage == 1.0
