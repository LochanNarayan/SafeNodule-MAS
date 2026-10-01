"""LLM-call accounting and adaptive-routing cost claims."""
import copy

from lna.config import Config
from lna.data import make_dataset
from lna.evaluate import compute_metrics
from lna.llm.instrumented import CountingBackend
from lna.llm.mock import MockBackend
from lna.pipeline import LungNoduleAgent
from lna.router import oracle_difficulty, route


def test_counting_backend_tallies_every_call():
    b = CountingBackend(MockBackend(0))
    b.complete("s", "p", {"task": "summarize", "opinions": []})
    b.structured("s", "p", "judge_nodule", {"overlap": 0.5})
    b.structured("s", "p", "judge_nodule", {"overlap": 0.5})
    assert b.total == 3
    assert b.calls["judge_nodule"] == 2 and b.calls["summarize"] == 1
    b.reset()
    assert b.total == 0


def test_pipeline_records_call_counts():
    r = LungNoduleAgent(Config()).diagnose(make_case_one())
    assert r.n_llm_calls > 0
    assert sum(r.llm_call_breakdown.values()) == r.n_llm_calls


def make_case_one():
    return make_dataset(1, seed=11, config=Config())[0]


def test_adaptive_router_is_cheaper_than_always_full_board():
    for regime in ("clean", "noisy"):
        cfg = Config() if regime == "clean" else Config.for_regime(regime)
        no_router = copy.deepcopy(cfg); no_router.use_router = False
        cases = make_dataset(40, seed=3, config=cfg)
        adaptive = compute_metrics(LungNoduleAgent(cfg).diagnose_many(
            [c for c in cases])).mean_llm_calls
        full = compute_metrics(LungNoduleAgent(no_router).diagnose_many(
            [c for c in cases])).mean_llm_calls
        assert adaptive < full


def test_router_easy_path_for_unambiguous_benign_ggo():
    cfg = Config()
    case = make_dataset(1, seed=0, config=cfg)[0]
    case.perceived.density = "ggo"
    case.perceived.margin = "smooth"
    case.perceived.spiculation = False
    case.perceived.pleural_indentation = False
    case.perceived.vascular_convergence = False
    case.perceived.long_diameter_mm = 4.0
    case.measurements = None
    d = route(case, cfg)
    assert d.difficulty == "easy" and d.max_rounds == 1


def test_under_triage_is_measured():
    cfg = Config.for_regime("adversarial")
    results = LungNoduleAgent(cfg).diagnose_many(make_dataset(50, seed=4, config=cfg))
    rate = sum(r.under_triaged for r in results) / len(results)
    assert 0.0 <= rate <= 1.0
    # at least the mechanism triggers somewhere under heavy perception noise
    assert any(r.under_triaged for r in results)
