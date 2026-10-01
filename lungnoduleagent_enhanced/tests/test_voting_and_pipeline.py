import copy

from lna.config import Config
from lna.clinical import feature_class_scores, softmax, normalized_entropy
from lna.data import make_case, make_dataset
from lna.doctors.calibration import aggregate
from lna.pipeline import LungNoduleAgent
from lna.types import AgentOpinion, CLASSES


def test_softmax_sums_to_one():
    p = softmax({"a": 1.0, "b": 2.0, "c": 0.5})
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_entropy_bounds():
    uniform = {c: 1.0 / 3 for c in CLASSES}
    peaked = {"pre_invasive": 0.98, "minimally_invasive": 0.01, "invasive": 0.01}
    assert normalized_entropy(uniform) > 0.99
    assert normalized_entropy(peaked) < 0.3


def test_aggregate_weighted():
    ops = [
        AgentOpinion("oncologist", {"pre_invasive": 0.1, "minimally_invasive": 0.1,
                                    "invasive": 0.8}, "", [], 0.9),
        AgentOpinion("pulmonologist", {"pre_invasive": 0.6, "minimally_invasive": 0.3,
                                       "invasive": 0.1}, "", [], 0.5),
    ]
    agg = aggregate(ops, {"oncologist": 1.2, "pulmonologist": 0.9})
    assert abs(sum(agg.values()) - 1.0) < 1e-9
    # oncologist has higher weight+confidence -> invasive should lead
    assert max(agg, key=agg.get) == "invasive"


def test_pipeline_runs_and_is_deterministic():
    cfg = Config(seed=7, verbose=False)
    a1 = LungNoduleAgent(copy.deepcopy(cfg))
    a2 = LungNoduleAgent(copy.deepcopy(cfg))
    c1 = make_case("t", seed=1)
    c2 = make_case("t", seed=1)
    r1 = a1.diagnose(c1)
    r2 = a2.diagnose(c2)
    assert r1.predicted_label == r2.predicted_label
    assert abs(r1.confidence - r2.confidence) < 1e-9
    assert r1.predicted_label in CLASSES


def test_pipeline_batch_accuracy_reasonable():
    cfg = Config(seed=7, verbose=False)
    agent = LungNoduleAgent(cfg)
    cases = make_dataset(30, seed=100)
    results = agent.diagnose_many(cases)
    answered = [r for r in results if not r.abstained]
    # sanity: on synthetic data where agents reason from true features, answered
    # accuracy should be clearly better than the 1/3 random baseline.
    acc = sum(r.correct for r in answered) / max(len(answered), 1)
    assert acc > 0.5
    # every case that was answered has a valid label
    assert all(r.predicted_label in CLASSES for r in results)
