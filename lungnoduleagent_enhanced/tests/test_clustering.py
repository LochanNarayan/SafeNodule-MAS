import numpy as np

from lna.spotter.clustering import iou, mask_distance, cluster_masks, dbscan_precomputed


def _blob(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def test_iou_identical():
    m = _blob(40, 40, 20, 20, 5)
    assert iou(m, m) == 1.0
    assert mask_distance(m, m) == 0.0


def test_iou_disjoint():
    a = _blob(40, 40, 10, 10, 4)
    b = _blob(40, 40, 30, 30, 4)
    assert iou(a, b) == 0.0
    assert mask_distance(a, b) == 1.0


def test_dbscan_groups_and_noise():
    # 4 overlapping masks near (20,20) + 1 outlier far away
    masks = [_blob(50, 50, 20 + d, 20, 6) for d in (-1, 0, 1, 2)]
    masks.append(_blob(50, 50, 45, 45, 3))
    refined, labels = cluster_masks(masks, eps=0.5, min_pts=2)
    # the outlier must be labeled noise (-1)
    assert labels[-1] == -1
    # the 4 overlapping masks form (at least) one cluster
    assert len([l for l in labels if l >= 0]) >= 4
    assert len(refined) >= 1
    assert refined[0].dtype == bool


def test_dbscan_all_noise_when_min_pts_high():
    masks = [_blob(30, 30, 15, 15, 5), _blob(30, 30, 15, 15, 5)]
    D = np.array([[0.0, 0.0], [0.0, 0.0]])
    labels = dbscan_precomputed(D, eps=0.1, min_pts=5)
    assert all(l == -1 for l in labels)
