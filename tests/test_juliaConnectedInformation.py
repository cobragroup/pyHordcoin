import pyHordcoin as hc
import numpy as np
from itertools import combinations
import pytest
from juliacall import JuliaError


def test_ConnectedInformation_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.ConnectedInformation(A, 2), dict)


def test_ConnectedInformation_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.ConnectedInformation(A, 2)[2], float)


def test_ConnectedInformation_discrete_explicit():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.ConnectedInformation(A, 2, hc.RawPolymatroid()), dict)


def test_ConnectedInformation_continuous_explicit():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.ConnectedInformation(A, 2, hc.Ipfp())[2], float)


def test_ConnectedInformation_imaginary():
    A = np.random.randint(1000, size=[2, 2, 2])
    A = A + 1j
    with pytest.raises(ValueError):
        hc.ConnectedInformation(A, 2)


def test_ConnectedInformation_bad_method():
    A = np.random.randint(1000, size=[2, 2, 2])
    with pytest.raises(ValueError):
        hc.ConnectedInformation(A, 2, hc.RawPolymatroid().method)


def test_ConnectedInformation_continuous_GPolymatroid():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    with pytest.raises(JuliaError):
        hc.ConnectedInformation(A, 2, hc.GPolymatroid())


def test_ConnectedInformation_continuous_multiple_orders():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.ConnectedInformation(A, [2, 3]), dict)


def test_ConnectedInformation_discrete_precalculated():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    marginal_entropies = {}
    for i in range(3):
        for a in combinations(range(3), i):
            m = tuple(set(range(3)) - set(a))
            tmp = A.sum(a)
            k = tuple(b + 1 for b in m)
            marginal_entropies[k] = -(tmp * np.log2(tmp)).sum()
    assert np.isclose(
        hc.ConnectedInformation(A, 2, hc.RawPolymatroid())[2],
        hc.ConnectedInformation(
            A, 2, hc.RawPolymatroid(), precalculated_entropies=marginal_entropies
        )[2],
    )


def test_ConnectedInformation_discrete_precalculated_zero_indexed():
    A = np.random.randint(1000, size=[2, 2, 2])
    marginal_entropies = {}
    for i in range(3):
        for a in combinations(range(3), i):
            marginal_entropies[a] = 0.1
    with pytest.raises(AssertionError):
        hc.ConnectedInformation(A, 2, precalculated_entropies=marginal_entropies)
