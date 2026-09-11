import pyHordcoin as hc
import numpy as np
from itertools import combinations
import pytest
from juliacall import JuliaError


def test_connected_information_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    ci = hc.connected_information(A, 2)
    assert isinstance(ci, tuple)
    assert isinstance(ci[0], dict)
    assert ci[1] is None


def test_connected_information_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.connected_information(A, 2)[0][2], float)


def test_connected_information_discrete_explicit():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(
        hc.connected_information(A, 2, hc.RawPolymatroid(), full_output=True)[1], dict
    )


def test_connected_information_continuous_explicit():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.connected_information(A, 2, hc.Ipfp())[0][2], float)


def test_connected_information_imaginary():
    A = np.random.randint(1000, size=[2, 2, 2])
    A = A + 1j
    with pytest.raises(ValueError):
        hc.connected_information(A, 2)


def test_connected_information_bad_method():
    A = np.random.randint(1000, size=[2, 2, 2])
    with pytest.raises(ValueError):
        hc.connected_information(A, 2, hc.RawPolymatroid().method)


def test_connected_information_continuous_GPolymatroid():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    with pytest.raises(ValueError):
        hc.connected_information(A, 2, hc.GPolymatroid())


def test_connected_information_continuous_multiple_orders():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.connected_information(A, [2, 3])[0], dict)


def test_connected_information_discrete_precalculated():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    marginal_entropies = {}
    for i in range(4):
        for a in combinations(range(3), i):
            m = tuple(set(range(3)) - set(a))
            tmp = A.sum(m)
            k = tuple(b + 1 for b in a)
            marginal_entropies[k] = -(tmp * np.log2(tmp)).sum()
    assert np.isclose(
        hc.connected_information(A, 2, hc.RawPolymatroid())[0][2],
        hc.connected_information(
            A, 2, hc.RawPolymatroid(), precalculated_entropies=marginal_entropies
        )[0][2],
    )


def test_connected_information_discrete_precalculated_zero_indexed():
    A = np.random.randint(1000, size=[2, 2, 2])
    marginal_entropies = {}
    for i in range(3):
        for a in combinations(range(3), i):
            marginal_entropies[a] = 0.1
    with pytest.raises(AssertionError):
        hc.connected_information(A, 2, precalculated_entropies=marginal_entropies)


def test_default_order():
    A = np.random.randint(1000, size=[2, 2, 2])
    ci = hc.connected_information(A)
    assert isinstance(ci, tuple)
    assert isinstance(ci[0], dict)
    assert len(ci[0]) == 2
    assert ci[1] is None
