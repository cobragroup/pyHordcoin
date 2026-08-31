import pyHordcoin as hc
import numpy as np
from itertools import combinations
import pytest
from typing import cast

def test_maximise_entropy_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.maximise_entropy(A, 2).entropy, float)


def test_maximise_entropy_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.maximise_entropy(A, 2).joint_probability, np.ndarray)


def test_maximise_entropy_discrete_explicit():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.maximise_entropy(A, 2, hc.GPolymatroid()).entropy, float)


def test_maximise_entropy_continuous_explicit():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(
        hc.maximise_entropy(A, 2, hc.Cone()).joint_probability, np.ndarray
    )


def test_maximise_entropy_discrete_precalculated():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    marginal_entropies = {}
    for i in range(4):
        for a in combinations(range(3), i):
            m = tuple(set(range(3)) - set(a))
            tmp = B.sum(m)
            k = tuple(b + 1 for b in a)
            marginal_entropies[k] = hc.distribution_entropy(tmp)
    EMFMERes_entropy = hc.maximise_entropy(A, 3, hc.RawPolymatroid())
    EMFMERes_entropy = cast(hc.EMFMEResult, EMFMERes_entropy)
    maximise_result = hc.maximise_entropy(A, 2, hc.RawPolymatroid())
    assert np.isclose(
        maximise_result.entropy,
        hc.maximise_entropy(
            A, 2, hc.RawPolymatroid(), precalculated_entropies=EMFMERes_entropy
        ).entropy,
    )
    assert np.isclose(
        maximise_result.entropy,
        hc.maximise_entropy(
            A, 2, hc.RawPolymatroid(), precalculated_entropies=marginal_entropies
        ).entropy,
    )


def test_maximise_entropy_imaginary():
    A = np.random.randint(1000, size=[2, 2, 2])
    A = A + 1j
    with pytest.raises(ValueError):
        hc.maximise_entropy(A, 2)


def test_maximise_entropy_bad_method():
    A = np.random.randint(1000, size=[2, 2, 2])
    with pytest.raises(ValueError):
        hc.maximise_entropy(A, 2, hc.RawPolymatroid().method)
