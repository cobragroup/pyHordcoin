import pyHordcoin as hc
import numpy as np


def test_DistributionEntropy():
    A = np.full([2, 2, 2], 1 / 8)
    assert np.isclose(hc.distribution_entropy(A), 3)
