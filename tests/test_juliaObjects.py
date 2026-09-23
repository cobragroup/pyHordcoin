import pyHordcoin as hc
import numpy as np
import pytest
from itertools import combinations

def test_juliaRawPolymatroid():
    assert hc.RawPolymatroid()
    assert hc.RawPolymatroid(True)
    assert hc.RawPolymatroid(True, mle_correction=True)


def test_juliaGPolymatroid():
    assert hc.GPolymatroid()
    assert hc.GPolymatroid(True)
    assert hc.GPolymatroid(tolerance=0.01)


def test_juliaDistributionConstrained():
    assert hc.Cone()
    assert hc.Ipfp()
    assert hc.Ipfp(10)
    assert hc.Gradient()
    assert hc.Gradient(100)


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
    with pytest.raises(NotImplementedError):
        EResult.joint_probability
    with pytest.raises(NotImplementedError):
        EResult.marginal_entropies


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


def test_EResult_repr():
    A = np.random.randint(1000, size=[2, 2, 2])
    B = A.astype(np.float64) / A.sum()
    from numpy import array

    EMResult = hc.EMResult
    EMFMEResult = hc.EMFMEResult
    myEMResult = hc.maximise_entropy(B, 2)
    myEMFMEResult = hc.maximise_entropy(A, 2)
    assert isinstance(eval(repr(myEMResult)), EMResult)
    assert isinstance(eval(repr(myEMFMEResult)), EMFMEResult)


def test_repr():
    from pyHordcoin import RawPolymatroid, GPolymatroid, Ipfp, Cone, Gradient, SCS

    assert isinstance(eval(repr(RawPolymatroid(True))), RawPolymatroid)
    assert isinstance(eval(repr(GPolymatroid(tolerance=0.01))), GPolymatroid)
    assert isinstance(eval(repr(Ipfp(42))), Ipfp)
    assert isinstance(eval(repr(Cone())), Cone)
    assert isinstance(eval(repr(Gradient(42))), Gradient)
    assert isinstance(eval(repr(SCS())), SCS)
