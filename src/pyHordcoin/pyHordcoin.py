# Copyright (c) 2026 Giulio Tani Raffaelli (Institute of Computer Science, Czech Academy of Sciences)
# Copyright (c) 2026 Jakub Kislinger
# Copyright (c) 2026 Jaroslav Hlinka (Institute of Computer Science, Czech Academy of Sciences)
# Copyright (c) 2026 Tomáš Kroupa (Czech Technical University)
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

from juliacall import Main as jl, convert, JuliaError, Pkg
import numpy as np
from pathlib import Path
import os


def _init_julia_env():
    env = Path(__file__).parent / "julia"
    if not os.path.isdir(env):
        os.mkdir(env)
        with open(env / "Project.toml", "w") as f:
            f.write('[deps]\nHordcoin = "5495aede-444c-4b33-a3d8-b01a3ffd757a"\n')

    Pkg.activate(str(env))
    Pkg.instantiate()
    jl.seval("using Hordcoin")
    print("Julia environment initialized.")


_init_julia_env()


class AbstractOptimizer:
    """Abstract class for all the optimisers."""

    Optimizer = None
    initd = False
    init_string = ""

    def __init__(self) -> None:
        self.load_lib()

    @classmethod
    def load_lib(cls):
        """
        Load the Julia library.

        This function is called automatically when the object is created.
        It takes the init_string from the class and executes it in the Julia environment.
        After it has been called, the initd flag of the class is set to True.
        """
        if not cls.initd:
            try:
                jl.seval(f"using {cls.init_string}")
            except JuliaError:
                jl.seval(
                    f"""
                    import Pkg
                    Pkg.add("{cls.init_string}")
                    using {cls.init_string}
                    """
                )
            cls.initd = True


class SCS(AbstractOptimizer):
    init_string = "SCS"

    def __init__(self) -> None:
        """
        Initialises the SCS optimiser.

        Calls the parent class's __init__ to load the Julia library, and then sets the Optimizer attribute to the SCS optimiser.
        """
        super().__init__()
        self.Optimizer = jl.SCS.Optimizer()


class Mosek(AbstractOptimizer):
    init_string = "MosekTools"

    def __init__(self) -> None:
        """
        Initialises the Mosek optimiser.

        Calls the parent class's __init__ to load the Julia library, and then sets the Optimizer attribute to the Mosek optimiser.
        """
        super().__init__()
        self.Optimizer = jl.MosekTools.Optimizer()


class OptimisationMethod:
    method = None


class EntropyMethod(OptimisationMethod):
    pass


class RawPolymatroid(EntropyMethod):
    def __init__(
        self,
        zhang_yeung: bool = False,
        optimiser: AbstractOptimizer = SCS(),
        mle_correction: float = 0,
    ):
        """
        Initialises the RawPolymatroid optimisation method for fixed marginal entropy optimisation.

        Parameters
        ----------
        zhang_yeung : bool, optional
            Whether to include Zhang-Yeung inequalities (default False).
        optimiser : AbstractOptimizer, optional
            Optimiser to use (default SCS).
        mle_correction : float, optional
            Amount of MLE bias correction to apply (default 0).
        """
        super().__init__()
        self.method = jl.RawPolymatroid(
            convert(jl.Float64, mle_correction), zhang_yeung, optimiser.Optimizer
        )


class GPolymatroid(EntropyMethod):
    def __init__(
        self,
        zhang_yeung: bool = False,
        optimiser: AbstractOptimizer = SCS(),
        tolerance: float = 0,
    ):
        """
        Initialises the Grassberger-corrected Polymatroid optimisation method for fixed marginal entropy optimisation.

        Parameters
        ----------
        zhang_yeung : bool, optional
            Whether to include Zhang-Yeung inequalities (default False).
        optimiser : AbstractOptimizer, optional
            Optimiser to use (default SCS).
        tolerance : float, optional
            Relative tolerance for constraints (default 0).
        """
        super().__init__()
        self.method = jl.GPolymatroid(
            zhang_yeung, optimiser.Optimizer, convert(jl.Float64, tolerance)
        )


class MarginalMethod(OptimisationMethod):
    pass


class Cone(MarginalMethod):
    def __init__(
        self,
        optimiser: AbstractOptimizer = SCS(),
    ) -> None:
        """
        Initialises the Cone optimisation method for fixed marginal distribution optimisation.

        Parameters
        ----------
        optimiser : AbstractOptimizer, optional
            Optimiser to use (default SCS).
        """
        super().__init__()
        self.method = jl.Cone(optimiser.Optimizer)


class Gradient(MarginalMethod):
    def __init__(
        self,
        iterations: int = 10,
        optimiser: AbstractOptimizer = SCS(),
    ) -> None:
        """
        Initialises the Gradient optimisation method for fixed marginal distribution optimisation.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to run (default 10).
        optimiser : AbstractOptimizer, optional
            Optimiser to use (default SCS).
        """
        super().__init__()
        self.method = jl.Gradient(convert(jl.Int64, iterations), optimiser.Optimizer)


class Ipfp(MarginalMethod):
    def __init__(
        self,
        iterations: int = 10,
    ) -> None:
        """
        Initialises the Ipfp optimisation method for fixed marginal distribution optimisation.

        Parameters
        ----------
        iterations : int, optional
            Number of iterations to run (default 10).
        """
        super().__init__()
        self.method = jl.Ipfp(convert(jl.Int64, iterations))


def _format_precalculated_entropies(precalculated_entropies, dimension):
    _precalculated_entropies = {}
    for k, v in precalculated_entropies.items():
        assert len(k) == len(set(k)), f"Repeated dimension index in key ({k})."
        for k1 in k:
            assert 0 < k1 <= dimension, f"Invalid dimension index {k1} in key ({k})."
        assert isinstance(v, float) or np.issubdtype(v, np.floating)
        _precalculated_entropies[convert(jl.Array[jl.Int64], k)] = v

    return convert(jl.Dict, _precalculated_entropies)


def ConnectedInformation(
    distribution: np.ndarray,
    orders: np.ndarray | list[int] | int,
    method: OptimisationMethod | None = None,
    precalculated_entropies: None | dict[tuple[int, ...], float] = None,
) -> dict[int, float]:
    """
    Computes connected information for given joined probability and multiple `orders`. Optional argument `method`
    specifies which method to use for optimisation. Default is `Cone()`. Preferred when computing multiple connected
    informations - more efficient.

    Parameters
    ----------
    distribution : np.ndarray
        Joined probability distribution.
    orders : np.ndarray | int
        Orders of connected information to compute.
    method : OptimisationMethod | None, optional
        Method to use for optimisation (default None).

    Returns
    -------
    Dict{int, float}
        Computed connected informations.
    """
    dimension = len(distribution.shape)
    if isinstance(method, MarginalMethod) or (
        isinstance(method, RawPolymatroid)
        and np.issubdtype(distribution.dtype, np.floating)
    ):
        _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
    elif isinstance(method, EntropyMethod):
        _distribution = convert(jl.Array[jl.Int64, dimension], distribution)
    elif method is None:
        if np.issubdtype(distribution.dtype, np.integer):
            _distribution = convert(jl.Array[jl.Int64, dimension], distribution)
            method = RawPolymatroid()
        elif (
            np.issubdtype(distribution.dtype, np.floating)
            and not np.iscomplex(distribution).any()
        ):
            _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
            method = Ipfp()
        else:
            raise ValueError(
                f"Cannot infer type of optimisation from distribution dtype ('{distribution.dtype}')"
            )
    else:
        raise ValueError(f"Unrecognise method of type '{type(method)}'")

    if precalculated_entropies is not None and isinstance(method, EntropyMethod):
        extras = {
            "precalculated_entropies": _format_precalculated_entropies(
                precalculated_entropies, dimension
            )
        }
    else:
        extras = {}

    if isinstance(orders, (np.ndarray, list)):
        _orders = convert(jl.Vector, np.array(orders).astype(int))
    else:
        _orders = convert(jl.Int64, orders)

    CI = jl.connected_information(_distribution, _orders, method.method, **extras)

    if isinstance(method, EntropyMethod):
        return dict(CI[0])
    elif isinstance(orders, int):
        return {orders: float(CI)}
    else:
        return dict(CI)


def MaximiseEntropy(
    distribution: np.ndarray,
    order: int,
    method: OptimisationMethod | None = None,
    precalculated_entropies: None | dict[tuple[int, ...], float] = None,
) -> tuple[float, np.ndarray | None]:
    """
    Computes the maximum entropy of a distribution (not a probability distribution) with fixed entropy of marginals of size `order`.

    Parameters
    ----------
    distribution : np.ndarray
        Joined probability distribution.
    order : int
        Size of the marginals to keep fixed.
    method : OptimisationMethod | None, optional
        Method to use for optimisation (default None).
    precalculated_entropies : None, optional
        Pre-calculated entropies of the marginals (not implemented yet).

    Returns
    -------
    float
        Computed maximum entropy.
    np.ndarray
        Maximum entropy distribution if `method` is a `MarginalMethod` else None.

    Raises
    ------
    ValueError
        If `method` is not recognised.
    NotImplementedError
        If `precalculated_entropies` is passed (not implemented yet).
    """
    dimension = len(distribution.shape)
    if isinstance(method, EntropyMethod):
        _distribution = convert(jl.Array[jl.Int64, dimension], distribution)
    elif isinstance(method, MarginalMethod):
        _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
    elif method is None:
        if np.issubdtype(distribution.dtype, np.integer):
            _distribution = convert(jl.Array[jl.Int64, dimension], distribution)
            method = RawPolymatroid()
        elif (
            np.issubdtype(distribution.dtype, np.floating)
            and not np.iscomplex(distribution).any()
        ):
            _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
            method = Ipfp()
        else:
            raise ValueError(
                f"Cannot infer type of optimisation from distribution dtype ('{distribution.dtype}')"
            )
    else:
        raise ValueError(f"Unrecognise method of type '{type(method)}'")

    _order = convert(jl.Int64, order)

    if precalculated_entropies is not None and isinstance(method, EntropyMethod):
        extras = {
            "precalculated_entropies": _format_precalculated_entropies(
                precalculated_entropies, dimension
            )
        }
    else:
        extras = {}

    if isinstance(method, EntropyMethod):
        return (
            jl.max_ent_fixed_ent_unnormalized(
                _distribution, _order, method.method, **extras
            ),
            None,
        )
    elif isinstance(method, MarginalMethod):
        ME = jl.maximise_entropy(
            _distribution,
            _order,
            method=method.method,
        )
        return ME.entropy, np.array(ME.joined_probability)


def DistributionEntropy(distribution: np.ndarray) -> float:
    """
    Compute the information entropy of a discrete probability distribution.

    Parameters
    ----------
    distribution : np.ndarray
        Discrete probability distribution (not necessarily normalized).

    Returns
    -------
    float
        Information entropy of the probability distribution in bits.

    Notes
    ----
    The entropy is computed as the negative sum of each probability value
    multiplied by its logarithm (base 2).

    Examples
    --------
    >>> DistributionEntropy([0.1, 0.4, 0.5])
    1.360964047443681
    """
    dimension = len(distribution.shape)
    _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
    return jl.distribution_entropy(_distribution)
