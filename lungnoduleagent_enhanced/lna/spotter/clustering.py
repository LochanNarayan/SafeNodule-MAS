"""IoU-based mask clustering — a real numpy implementation of the paper's
Eq. (1)-(3): distance d = 1 - IoU, grouped with DBSCAN, then each cluster is
averaged and binarized at 0.5.

Self-contained DBSCAN on a precomputed distance matrix (no sklearn dependency).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two binary masks (paper Eq. 2)."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_distance(a: np.ndarray, b: np.ndarray) -> float:
    """d(mi, mj) = 1 - IoU  (paper Eq. 1). Lives in [0, 1]."""
    return 1.0 - iou(a, b)


def _pairwise_distances(masks: List[np.ndarray]) -> np.ndarray:
    n = len(masks)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = mask_distance(masks[i], masks[j])
            D[i, j] = D[j, i] = d
    return D


def dbscan_precomputed(D: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Minimal DBSCAN over a precomputed distance matrix.

    Returns a label array: cluster ids >= 0, or -1 for noise.
    Mirrors the paper's density-reachability expansion (Eq. 3).
    """
    n = D.shape[0]
    labels = np.full(n, -2, dtype=int)  # -2 = unvisited
    cluster_id = -1

    def neighbors(i: int) -> List[int]:
        return [j for j in range(n) if D[i, j] <= eps]

    for i in range(n):
        if labels[i] != -2:
            continue
        neigh = neighbors(i)
        if len(neigh) < min_pts:          # includes self; density too low
            labels[i] = -1                 # tentatively noise
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds = [j for j in neigh if j != i]
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if labels[j] == -1:            # was noise -> border point
                labels[j] = cluster_id
            if labels[j] == -2:
                labels[j] = cluster_id
                jneigh = neighbors(j)
                if len(jneigh) >= min_pts:
                    seeds.extend(x for x in jneigh if x not in seeds)
            k += 1
    return labels


def cluster_masks(masks: List[np.ndarray], eps: float, min_pts: int,
                  binarize: float = 0.5) -> Tuple[List[np.ndarray], np.ndarray]:
    """Cluster masks, return (averaged binarized masks per cluster, labels)."""
    if not masks:
        return [], np.array([], dtype=int)
    D = _pairwise_distances(masks)
    labels = dbscan_precomputed(D, eps=eps, min_pts=min_pts)
    refined: List[np.ndarray] = []
    for cid in sorted(set(labels)):
        if cid < 0:
            continue
        members = [masks[i] for i in range(len(masks)) if labels[i] == cid]
        avg = np.mean([m.astype(np.float32) for m in members], axis=0)
        refined.append(avg >= binarize)
    return refined, labels
