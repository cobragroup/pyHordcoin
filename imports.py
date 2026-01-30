import pytest


def test_import_module():
    import pyhordcoin

    assert pyhordcoin


def test_version():
    import pyhordcoin as hc

    assert isinstance(hc.__version__, str)
    assert hc.__version__ != ""


def test_import_submodules():
    import pyhordcoin as hc

    assert hc.SCS()
    assert hc.Cone()
