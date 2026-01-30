import pyhordcoin as hc
import numpy as np


def test_ConnectedInformation_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.ConnectedInformation(A, 2), dict)


def test_ConnectedInformation_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.ConnectedInformation(A, 2), float)


def test_MaximiseEntropy_discrete():
    A = np.random.randint(1000, size=[2, 2, 2])
    assert isinstance(hc.MaximiseEntropy(A, 2)[0], float)


def test_MaximiseEntropy_continuous():
    A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
    A /= A.sum()
    assert isinstance(hc.MaximiseEntropy(A, 2)[1], np.ndarray)
