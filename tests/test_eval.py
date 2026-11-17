import eval as cosmo_eval


def test_purity():
    hand = {0: "a", 1: "a", 2: "b", 3: "b", 4: "b"}
    p, n = cosmo_eval.purity([[0, 1, 2], [3, 4]], hand)   # first group: 2 of 3 agree
    assert n == 5 and abs(p - 4 / 5) < 1e-9
    assert cosmo_eval.purity([[9]], hand) == (None, 0)    # unlabeled pieces are ignored
