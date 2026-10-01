"""The Evidence Verifier must ground on the CT report, not the oracle, and must
flag agent-fabricated citations."""
from lna.config import Config
from lna.data import make_case
from lna.doctors.verifier import (grade_claim, parse_report, verifier_scorecard,
                                  verify_opinion)
from lna.types import AgentOpinion, Case


REPORT = ("A part-solid density nodule with a irregular shape and spiculated margin "
          "is located in the right upper lobe. The nodule measures 12.0 mm in long "
          "axis and 9.0 mm in short axis. There is spiculation, pleural indentation. "
          "No vascular convergence, air bronchogram, cavitation is present.")


def test_parse_report_extracts_claims():
    c = parse_report(REPORT)
    assert c.density == "part_solid"
    assert c.margin == "spiculated"
    assert c.shape == "irregular"
    assert "spiculation" in c.signs_present
    assert "pleural_indentation" in c.signs_present
    assert "cavitation" in c.signs_absent
    assert "vascular_convergence" in c.signs_absent


def test_grade_claim_verdicts():
    c = parse_report(REPORT)
    assert grade_claim("spiculation", c)[0] == "SUPPORTED"
    assert grade_claim("cavitation", c)[0] == "CONTRADICTED"       # report says absent
    assert grade_claim("density:solid", c)[0] == "CONTRADICTED"    # report says part_solid
    assert grade_claim("density:part_solid", c)[0] == "SUPPORTED"


def _case_with_report(report: str) -> Case:
    c = make_case("v", seed=1, config=Config())
    c.report = report
    return c


def test_verify_opinion_drops_and_penalises_fabrications():
    case = _case_with_report(REPORT)
    op = AgentOpinion(
        role="oncologist",
        probs={"pre_invasive": 0.1, "minimally_invasive": 0.2, "invasive": 0.7},
        rationale="", cited_features=["spiculation", "cavitation", "density:solid"],
        confidence=0.9)
    cleaned, trace = verify_opinion(case, op)
    assert "spiculation" in cleaned.cited_features           # supported -> kept
    assert "cavitation" not in cleaned.cited_features        # contradicted -> dropped
    assert "density:solid" not in cleaned.cited_features
    assert cleaned.confidence < op.confidence                # penalised
    assert sum(t.startswith("FLAG") for t in trace) == 2


def test_scorecard_recall_is_one_for_pure_fabrication():
    """A sign that is absent in BOTH the report and the oracle, but cited by the
    agent, is a fabrication the verifier must catch."""
    case = make_case("sc", seed=3, config=Config())
    # force a clean oracle with no cavitation, and a report that says so
    case.features.cavitation = False
    case.perceived.cavitation = False
    case.report = REPORT  # asserts cavitation absent
    op = AgentOpinion("pathologist", {"pre_invasive": .2, "minimally_invasive": .3,
                                      "invasive": .5}, "", ["cavitation"], 0.8)
    sc = verifier_scorecard(case, [op])
    assert sc["tp"] == 1 and sc["fn"] == 0
