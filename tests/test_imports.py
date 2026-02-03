import os
from pathlib import Path

juliaPath = Path(__file__).parent.parent / "src" / "pyHordcoin" / "julia"
if os.path.isdir(juliaPath):
    for file in juliaPath.iterdir():
        os.unlink(file)
    os.rmdir(juliaPath)

import pyHordcoin as hc


def test_version():
    assert isinstance(hc.__version__, str)
    assert hc.__version__ != ""


def test_import_submodules():
    assert hc.SCS()
    assert hc.Mosek()
