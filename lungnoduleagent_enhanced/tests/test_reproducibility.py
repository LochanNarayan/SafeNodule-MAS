"""Determinism, multi-seed aggregation shape, and the paired bootstrap."""
import copy

from lna.config import Config
from lna.data import make_dataset
from lna.evaluate import (Aggregate, m_acc_all, paired_bootstrap, run_multiseed)
from lna.pipeline import LungNoduleAgent


def test_same_seed_same_result():
    cfg = Config.for_regime("noisy", seed=7)
    a = LungNoduleAgent(copy.deepcopy(cfg)).diagnose_many(
        make_dataset(20, seed=42, config=cfg))
    b = LungNoduleAgent(copy.deepcopy(cfg)).diagnose_many(
        make_dataset(20, seed=42, config=cfg))
    assert [r.predicted_label for r in a] == [r.predicted_label for r in b]
    assert [round(r.confidence, 9) for r in a] == [round(r.confidence, 9) for r in b]
    assert [r.n_llm_calls for r in a] == [r.n_llm_calls for r in b]


def test_multiseed_aggregate_shape_and_ci():
    agg = run_multiseed(Config(), 30, [1, 2, 3, 4])
    assert isinstance(agg, Aggregate) and len(agg.seeds) == 4
    for k in ("accuracy_all", "coverage", "ece", "mean_llm_calls"):
        assert k in agg.mean and k in agg.std
    lo, hi = agg.ci95("accuracy_all")
    assert lo <= agg.mean["accuracy_all"] <= hi


def test_paired_bootstrap_detects_board_effect():
    base = Config.for_regime("borderline")
    single = copy.deepcopy(base); single.board_roles = base.board_roles[:1]
    out = paired_bootstrap(base, single, 40, [1, 2, 3, 4], m_acc_all, n_boot=400)
    assert out["delta"] > 0                     # full board beats single agent
    assert out["p_value"] < 0.10
    assert out["lo"] <= out["delta"] <= out["hi"]


def test_paired_bootstrap_null_for_identical_configs():
    base = Config()
    out = paired_bootstrap(base, copy.deepcopy(base), 30, [1, 2, 3],
                           m_acc_all, n_boot=300)
    assert abs(out["delta"]) < 1e-9
    assert out["p_value"] > 0.5
