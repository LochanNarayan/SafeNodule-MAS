"""The stress regimes must actually produce the distributions they claim."""
import numpy as np

from lna.clinical import feature_class_scores, softmax
from lna.config import Config
from lna.data import make_case, make_dataset


def _margin(feats):
    p = sorted(softmax(feature_class_scores(feats.as_dict())).values(), reverse=True)
    return p[0] - p[1]


def test_clean_regime_perception_is_faithful():
    for i in range(20):
        c = make_case(f"c{i}", seed=i, config=Config())
        assert c.perceived.as_dict() == c.features.as_dict()


def test_noisy_regime_corrupts_perception():
    cfg = Config.for_regime("noisy")
    diffs = 0
    for i in range(40):
        c = make_case(f"c{i}", seed=i, config=cfg)
        if c.perceived.as_dict() != c.features.as_dict():
            diffs += 1
    # with feature_noise=0.25 across ~8 attributes, almost every case differs
    assert diffs >= 30


def test_borderline_regime_is_near_the_boundary():
    cfg = Config.for_regime("borderline")
    clean = [_margin(make_case(f"a{i}", seed=i, config=Config()).features)
             for i in range(60)]
    bord = [_margin(make_case(f"b{i}", seed=i, config=cfg).features)
            for i in range(60)]
    assert np.mean(bord) < np.mean(clean)
    assert np.mean(bord) < cfg.borderline_margin + 0.05


def test_imbalanced_regime_respects_prior():
    cfg = Config.for_regime("imbalanced")            # prior favours pre_invasive
    labels = [c.true_label for c in make_dataset(120, seed=5, config=cfg)]
    frac_pre = labels.count("pre_invasive") / len(labels)
    base = [c.true_label for c in make_dataset(120, seed=5, config=Config())]
    assert frac_pre > base.count("pre_invasive") / len(base)


def test_adversarial_regime_degrades_detection():
    from lna.pipeline import LungNoduleAgent
    clean_iou = np.mean([LungNoduleAgent(Config()).diagnose(
        make_case(f"c{i}", seed=i, config=Config())).detection_iou for i in range(15)])
    adv = Config.for_regime("adversarial")
    adv_iou = np.mean([LungNoduleAgent(adv).diagnose(
        make_case(f"c{i}", seed=i, config=adv)).detection_iou for i in range(15)])
    assert adv_iou < clean_iou
