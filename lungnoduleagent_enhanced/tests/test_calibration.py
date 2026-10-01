"""Temperature fitting must reduce ECE on an overconfident synthetic set, and the
pipeline must actually install a fitted temperature."""
import numpy as np

from lna.config import Config
from lna.data import make_dataset
from lna.doctors.calibration import fit_temperature, temperature_scale
from lna.evaluate import _ece, compute_metrics, run_multiseed
from lna.pipeline import LungNoduleAgent
from lna.types import CLASSES


def _overconfident_samples(n=300, seed=0):
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n):
        y = CLASSES[rng.randint(3)]
        # model is right 65% of the time but always ~0.95 confident -> overconfident
        right = rng.rand() < 0.65
        top = y if right else CLASSES[(CLASSES.index(y) + 1) % 3]
        p = {c: 0.03 for c in CLASSES}
        p[top] = 0.94
        s = sum(p.values())
        samples.append(({c: v / s for c, v in p.items()}, y))
    return samples


def test_fit_temperature_reduces_ece():
    samples = _overconfident_samples()
    def ece_at(t):
        pairs = []
        for probs, y in samples:
            sc = temperature_scale(probs, t)
            lab = max(sc, key=sc.get)
            pairs.append((sc[lab], lab == y))
        return _ece(pairs)
    t = fit_temperature(samples)
    assert t > 1.0                       # needs softening
    assert ece_at(t) < ece_at(1.0) - 0.05


def test_pipeline_installs_fitted_temperature():
    cfg = Config.for_regime("adversarial")
    agent = LungNoduleAgent(cfg)
    cases = make_dataset(40, seed=7, config=cfg)
    t = agent.fit_calibration(cases[:20])
    assert agent.board.temperature_override == t
    r = agent.diagnose(cases[20])
    assert abs(r.temperature_used - t) < 1e-9


def test_calibration_split_lowers_ece_vs_fixed_T_under_shift():
    base = Config.for_regime("adversarial")
    seeds = list(range(1, 6))
    with_fit = run_multiseed(base, 60, seeds).mean["ece"]
    import copy
    no_cal = copy.deepcopy(base); no_cal.use_calibration = False
    raw = run_multiseed(no_cal, 60, seeds).mean["ece"]   # ece at T=1
    assert with_fit < raw
