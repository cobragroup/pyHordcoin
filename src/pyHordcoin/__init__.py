#!/usr/bin/env python3
# Copyright 2026 Giulio Tani Raffaelli
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
pyHordcoin

Python interface to the Hordcoin.jl Julia package for Connected Information.

This package provides a Python interface to the Hordcoin.jl Julia package.
The package allows the computation of the Connected Information, and the
maximisation of the entropy of discrete distributions given a fixed marginal
distribution or entropy up to a chosen order.

The following functions are available:

- `connected_information`: computes the Connected Information.
- `maximise_entropy`: maximises the entropy of a discrete distribution given a
  fixed marginal distribution or entropy up to a chosen order.
- `distribution_entropy`: computes the entropy of a discrete distribution.
- `RawPolymatroid`: implements the raw polymatroid approximation.
- `GPolymatroid`: implements the Grassberger-corrected polymatroid approximation.
- `Ipfp`: implements the iterative proportional fitting procedure.
- `Cone`: implements the cone programming optimiser.
- `Gradient`: implements the gradient descent optimiser.
- `SCS`: implements the sequential coordinate-wise search optimiser.
- `Mosek`: implements the Mosek optimiser.
- `EResult`: represents the result of a Connected Information computation.
- `EMResult`: represents the result of an entropy maximisation with marginal distribution constraints computation.
- `EMFMEResult`: represents the result of an entropy maximisation with fixed marginal entropies computation.
"""

from .pyHordcoin import (
    connected_information,
    maximise_entropy,
    distribution_entropy,
    RawPolymatroid,
    GPolymatroid,
    Ipfp,
    Cone,
    Gradient,
    SCS,
    Mosek,
    EResult,
    EMResult,
    EMFMEResult,
)

# from ._version import __version__
import importlib.metadata

__version__ = importlib.metadata.version("pyhordcoin")
