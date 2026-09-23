import os
import shutil, platformdirs

juliaPath = platformdirs.user_cache_path("pyHordcoin") / "julia"
if os.path.isdir(juliaPath):
    shutil.rmtree(juliaPath)

import pyHordcoin as hc


def test_version():
    assert isinstance(hc.__version__, str)
    assert hc.__version__ != ""


def test_import_submodules():
    assert hc.SCS()
    assert hc.Mosek()
