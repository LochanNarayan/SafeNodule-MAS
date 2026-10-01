"""Guideline escalation logic and the evaluation-metric helpers."""
from lna.config import Config
from lna.data import make_case
from lna.doctors.guideline import assess, assess_detail
from lna.evaluate import _aurc, _ece, _macro_f1
from lna.pipeline import PipelineResult
from lna.types import CLASSES


def _case(**over):
    c = make_case("g", seed=1, config=Config())
    for k, v in over.items():
        setattr(c.features, k, v)
    return c


def test_ggo_with_spiculation_escalates_not_contradicts():
    d = assess_detail(_case(density="ggo", spiculation=True, long_diameter_mm=5.0))
    assert d["category"] == "Lung-RADS 4X"
    assert "conservative" not in assess(_case(density="ggo", spiculation=True)).lower()


def test_small_smooth_ggo_is_category_2():
    d = assess_detail(_case(density="ggo", spiculation=False, pleural_indentation=False,
                            vascular_convergence=False, long_diameter_mm=4.0,
                            prior_long_diameter_mm=None))
    assert d["category"] == "Lung-RADS 2"


def test_large_solid_is_suspicious():
    d = assess_detail(_case(density="solid", long_diameter_mm=15.0, spiculation=False,
                            pleural_indentation=False, vascular_convergence=False,
                            prior_long_diameter_mm=None))
    assert d["category"].startswith("Lung-RADS 4")


def test_interval_growth_escalates():
    d = assess_detail(_case(density="solid", long_diameter_mm=12.0,
                            prior_long_diameter_mm=7.0, spiculation=False,
                            pleural_indentation=False, vascular_convergence=False))
    assert d["category"] in ("Lung-RADS 4B", "Lung-RADS 4X")
    assert "growth" in d["drivers"]


# ---- metric helpers -------------------------------------------------------
def _mk(pred, true, conf, abst=False, raw=None):
    return PipelineResult(
        case_id="x", detection_iou=1.0, report="", routing="easy",
        predicted_label=pred, true_label=true, correct=(pred == true and not abst),
        confidence=conf, abstained=abst, rounds_used=1, guideline="",
        calibrated_probs=raw or {c: 1/3 for c in CLASSES}, notes=[], evidence_trace=[],
        raw_probs=raw or {c: 1/3 for c in CLASSES})


def test_ece_zero_when_perfectly_calibrated():
    # confidence 1.0 and always correct -> ECE 0
    pairs = [(1.0, True)] * 20
    assert _ece(pairs) == 0.0


def test_aurc_lower_when_confidence_ranks_correctness():
    good = [_mk("invasive", "invasive", 0.9), _mk("invasive", "invasive", 0.8),
            _mk("invasive", "pre_invasive", 0.2)]
    bad = [_mk("invasive", "pre_invasive", 0.9), _mk("invasive", "invasive", 0.2)]
    assert _aurc(good) < _aurc(bad)


def test_macro_f1_bounds():
    rs = [_mk("invasive", "invasive", 0.9), _mk("pre_invasive", "pre_invasive", 0.9),
          _mk("minimally_invasive", "minimally_invasive", 0.9)]
    assert abs(_macro_f1(rs) - 1.0) < 1e-9
