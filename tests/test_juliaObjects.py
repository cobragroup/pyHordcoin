import pyHordcoin as hc
import numpy as np
import pytest
from itertools import combinations

def test_juliaRawPolymatroid():
    assert hc.RawPolymatroid()


def test_juliaGPolymatroid():
    assert hc.GPolymatroid()


def test_juliaOptimizers():
    assert hc.Cone()
    assert hc.Ipfp()
    assert hc.Gradient()


def test_EMResult():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    entropy = hc.distribution_entropy(B)
    assert hc.EMResult(entropy, B)
    EMResult = hc.EMResult(entropy, B)
    assert EMResult.entropy == entropy
    assert np.isclose(EMResult.joint_probability, B).all()
    with pytest.raises(NotImplementedError):
        EMResult.marginal_entropies
    assert hc.EMResult(EMResult.julia_obj)


def test_EResult():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    entropy = hc.distribution_entropy(B)
    EMResult = hc.EMResult(entropy, B)
    assert hc.EResult(EMResult.julia_obj)
    EResult = hc.EResult(EMResult.julia_obj)
    assert EResult.entropy == entropy
    assert EResult.joint_probability is None
    assert EResult.marginal_entropies is None


def test_EMFMEResult():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    marginal_entropies = {}
    for i in range(4):
        for a in combinations(range(3), i):
            m = tuple(set(range(3)) - set(a))
            tmp = B.sum(m)
            k = tuple(b + 1 for b in a)
            marginal_entropies[k] = hc.distribution_entropy(tmp)
    entropy = hc.distribution_entropy(B)
    assert hc.EMFMEResult(entropy, marginal_entropies)
    EMFMEResult = hc.EMFMEResult(entropy, marginal_entropies)
    assert EMFMEResult.entropy == entropy
    assert EMFMEResult.marginal_entropies == marginal_entropies
    with pytest.raises(NotImplementedError):
        EMFMEResult.joint_probability
