import pyHordcoin as hc


def test_juliaRawPolymatroid():
    assert hc.RawPolymatroid()


def test_juliaGPolymatroid():
    assert hc.GPolymatroid()


def test_juliaOptimizers():
    assert hc.Cone()
    assert hc.Ipfp()
    assert hc.Gradient()
