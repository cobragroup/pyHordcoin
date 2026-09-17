import pyHordcoin as hc
import numpy as np


def test_distribution_entropy():
    A = np.full([2, 2, 2], 1 / 8)
    assert np.isclose(hc.distribution_entropy(A), 3)


def test_precompute_entropies():
    A = np.full([2, 2, 2], 1 / 8)
    ent = hc.precompute_entropies(A)
    assert isinstance(ent, dict)
    assert ent[(1, 2, 3)] == 3
