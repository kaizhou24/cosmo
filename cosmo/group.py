"""Stage 4: group similar items.

Agglomerative clustering starts with every image as its own group and
repeatedly merges the two closest groups. "Average linkage" means the
distance between two groups is the average distance between their members.
It stops merging once the closest pair is farther apart than
distance_threshold, so we never have to guess the number of groups.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from . import config


def distance_matrix(vectors: np.ndarray, color_probs: np.ndarray | None = None,
                    color_weight: float = config.COLOR_WEIGHT) -> np.ndarray:
    """Pairwise distance between pieces.

    Base: cosine distance between embeddings (0 = identical direction).
    The embedding mostly captures shape and garment type, so two pairs of
    trousers look close even when one is grey and one is black. To fix that
    we add a color term: each piece has a probability distribution over color
    names, and the "total variation" between two distributions (half the sum
    of absolute differences) is 0 when they agree and 1 when they share nothing.
    """
    d = 1.0 - vectors @ vectors.T
    if color_probs is not None and color_weight:
        tv = 0.5 * np.abs(color_probs[:, None, :] - color_probs[None, :, :]).sum(axis=2)
        d = d + color_weight * tv
    d = np.clip((d + d.T) / 2, 0, None)
    np.fill_diagonal(d, 0)
    return d


def cluster(vectors: np.ndarray, threshold: float = config.DEFAULT_DISTANCE_THRESHOLD,
            color_probs: np.ndarray | None = None) -> np.ndarray:
    """Return a cluster id per row. Rows must be unit vectors."""
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",     # we hand it our own distance matrix
        linkage="average",
        distance_threshold=threshold,
    )
    return model.fit_predict(distance_matrix(vectors, color_probs))


def groups_from_labels(labels: np.ndarray, indices: list[int] | None = None) -> list[list[int]]:
    """Turn per-row cluster ids into lists of item indices, largest group first."""
    if indices is None:
        indices = list(range(len(labels)))
    buckets: dict[int, list[int]] = {}
    for idx, lab in zip(indices, labels):
        buckets.setdefault(int(lab), []).append(idx)
    return sorted(buckets.values(), key=lambda g: (-len(g), g[0]))


def group_all(vectors: np.ndarray, threshold: float = config.DEFAULT_DISTANCE_THRESHOLD) -> list[list[int]]:
    """Milestone 1: cluster the whole set, no category split."""
    return groups_from_labels(cluster(vectors, threshold))


def group_by_category(vectors: np.ndarray, categories: list[str],
                      threshold: float = config.DEFAULT_DISTANCE_THRESHOLD,
                      color_probs: np.ndarray | None = None) -> list[list[int]]:
    """Split by predicted category first, then cluster within each category.

    A jacket can never land in the same group as a shoe, however similar
    their vectors happen to be.
    """
    groups: list[list[int]] = []
    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cp = color_probs[idx] if color_probs is not None else None
        groups += groups_from_labels(cluster(vectors[idx], threshold, cp), idx)
    return sorted(groups, key=lambda g: (-len(g), g[0]))
