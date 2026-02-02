import pyHordcoin as hc
import numpy as np
from itertools import combinations
import pytest


def test_MaximiseEntropy_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.MaximiseEntropy(A, 2)[0], float)


def test_MaximiseEntropy_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.MaximiseEntropy(A, 2)[1], np.ndarray)


def test_MaximiseEntropy_discrete_explicit():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.MaximiseEntropy(A, 2, hc.GPolymatroid())[0], float)


def test_MaximiseEntropy_continuous_explicit():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.MaximiseEntropy(A, 2, hc.Cone())[1], np.ndarray)


def test_MaximiseEntropy_discrete_precalculated():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    marginal_entropies = {}
    for i in range(3):
        for a in combinations(range(3), i):
            m = tuple(set(range(3)) - set(a))
            tmp = B.sum(m)
            k = tuple(b + 1 for b in m)
            marginal_entropies[k] = -(tmp * np.log2(tmp)).sum()
    assert np.isclose(
        hc.MaximiseEntropy(A, 2, hc.RawPolymatroid())[0],
        hc.MaximiseEntropy(
            A, 2, hc.RawPolymatroid(), precalculated_entropies=marginal_entropies
        )[0],
    )


def test_MaximiseEntropy_imaginary():
    A = np.random.randint(1000, size=[2, 2, 2])
    A = A + 1j
    with pytest.raises(ValueError):
        hc.MaximiseEntropy(A, 2)


def test_MaximiseEntropy_bad_method():
    A = np.random.randint(1000, size=[2, 2, 2])
    with pytest.raises(ValueError):
        hc.MaximiseEntropy(A, 2, hc.RawPolymatroid().method)
