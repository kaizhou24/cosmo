import numpy as np

from cosmo import group


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_two_obvious_clusters():
    rng = np.random.default_rng(0)
    a = _unit(np.array([1, 0, 0]) + 0.02 * rng.standard_normal((5, 3)))
    b = _unit(np.array([0, 1, 0]) + 0.02 * rng.standard_normal((3, 3)))
    groups = group.group_all(np.vstack([a, b]), threshold=0.25)
    assert [len(g) for g in groups] == [5, 3]  # largest first
    assert sorted(groups[0]) == [0, 1, 2, 3, 4]


def test_lower_threshold_splits_more():
    v = _unit(np.array([[1, 0], [1, 0.3], [1, 0.6], [1, 0.9]]))
    loose = group.group_all(v, threshold=0.5)
    tight = group.group_all(v, threshold=0.01)
    assert len(tight) > len(loose)
    assert len(tight) == 4


def test_edge_sizes():
    assert group.group_all(np.zeros((0, 3))) == []
    assert group.group_all(_unit(np.array([[1.0, 0, 0]]))) == [[0]]


def test_color_term_splits_same_shape_different_color():
    # Two pieces with identical embeddings but opposite color distributions.
    v = _unit(np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]))
    colors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert len(set(group.cluster(v, threshold=0.1))) == 1                      # no color info: one group
    labels = group.cluster(v, threshold=0.1, color_probs=colors)               # color penalty 0.15 > 0.1
    assert labels[0] == labels[1] != labels[2]
