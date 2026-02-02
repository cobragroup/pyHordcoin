import pyHordcoin as hc


def test_version():
    assert isinstance(hc.__version__, str)
    assert hc.__version__ != ""


def test_import_submodules():
    assert hc.SCS()
    assert hc.Mosek()
