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

from juliacall import Main as jl, convert, JuliaError, AnyValue
import numpy as np
from pathlib import Path
import os
from typing import cast, Dict, Tuple

def _init_julia_env():
    env = Path(__file__).parent / "julia"
    if not os.path.isdir(env):
        os.mkdir(env)
        with open(env / "Project.toml", "w") as f:
            f.write('[deps]\nHordcoin = "5495aede-444c-4b33-a3d8-b01a3ffd757a"\n')

    jl.seval("using Pkg")
    jl.seval('Pkg.activate("{}")'.format(str(env)))
    jl.seval("Pkg.instantiate()")
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
                jl.seval(f"""
                    import Pkg
                    Pkg.add("{cls.init_string}")
                    using {cls.init_string}
                    """)
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

    def __repr__(self):
        return "SCS()"


class Mosek(AbstractOptimizer):
    init_string = "MosekTools"

    def __init__(self) -> None:
        """
        Initialises the Mosek optimiser.

        Calls the parent class's __init__ to load the Julia library, and then sets the Optimizer attribute to the Mosek optimiser.
        """
        super().__init__()
        self.Optimizer = jl.MosekTools.Optimizer()

    def __repr__(self):
        return "Mosek()"


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
        self.zhang_yeung = zhang_yeung
        self.optimiser = optimiser
        self.mle_correction = mle_correction
        self.method = jl.RawPolymatroid(
            convert(jl.Float64, mle_correction), zhang_yeung, optimiser.Optimizer
        )

    def __repr__(self):
        return f"RawPolymatroid(zhang_yeung={self.zhang_yeung}, optimiser={self.optimiser}, mle_correction={self.mle_correction})"


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
        self.zhang_yeung = zhang_yeung
        self.optimiser = optimiser
        self.tolerance = tolerance
        self.method = jl.GPolymatroid(
            zhang_yeung, optimiser.Optimizer, convert(jl.Float64, tolerance)
        )

    def __repr__(self):
        return f"GPolymatroid(zhang_yeung={self.zhang_yeung}, optimiser={self.optimiser}, tolerance={self.tolerance})"


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
        self.optimiser = optimiser
        self.method = jl.Cone(optimiser.Optimizer)

    def __repr__(self):
        return f"Cone(optimiser={self.optimiser})"


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
        self.iterations = iterations
        self.optimiser = optimiser
        self.method = jl.Gradient(convert(jl.Int64, iterations), optimiser.Optimizer)

    def __repr__(self):
        return f"Gradient(iterations={self.iterations}, optimiser={self.optimiser})"


class Ipfp(MarginalMethod):

    def __init__(
        self,
        iterations: int = 1000,
    ) -> None:
        """
        Initialises the Ipfp optimisation method for fixed marginal distribution optimisation.

        Parameters
        ----------
        iterations : int, optional
            Maximum number of iterations to run (default 1000).
        """

        super().__init__()
        self.iterations = iterations
        self.method = jl.Ipfp(convert(jl.Int64, iterations))

    def __repr__(self):
        return f"Ipfp(iterations={self.iterations})"


class EResult:
    julia_obj = None

    def __init__(self, julia_obj):
        self.julia_obj = julia_obj

    @property
    def entropy(self):
        return self.julia_obj.entropy

    @property
    def joint_probability(self):
        pass

    @property
    def marginal_entropies(self):
        pass


def _format_precalculated_entropies(precalculated_entropies: Dict[tuple[int, ...], float]|EResult, dimension: int):
    if isinstance(precalculated_entropies, EResult):
        if jl.Base.isa(precalculated_entropies.julia_obj, jl.EMFMEResult):
            return precalculated_entropies.julia_obj.marginal_entropies
        else:
            raise ValueError(
                f"Cannot create precalculated_entropies from {jl.Base.typeof(precalculated_entropies.julia_obj)}"
            )

    _precalculated_entropies = {}
    for k, v in precalculated_entropies.items():
        assert len(k) == len(set(k)), f"Repeated dimension index in key ({k})."
        for k1 in k:
            assert 0 < k1 <= dimension, f"Invalid dimension index {k1} in key ({k})."
        assert isinstance(v, float) or np.issubdtype(v, np.floating)
        _precalculated_entropies[convert(jl.Array[jl.Int64], k)] = v

    return convert(jl.Dict, _precalculated_entropies)


class EMResult(EResult):
    def __init__(
        self, entropy: float | AnyValue, joint_probability: None | np.ndarray = None
    ):
        if jl.Base.isa(entropy, jl.EMResult):
            super().__init__(entropy)
        elif isinstance(entropy, AnyValue):
            raise ValueError(f"Cannot create EMResult from {jl.Base.typeof(entropy)}")
        else:
            joint_probability = cast(np.ndarray, joint_probability)
            assert np.issubdtype(joint_probability.dtype, np.floating)
            dimension = len(joint_probability.shape)
            _distribution = convert(jl.Array[jl.Float64, dimension], joint_probability)
            super().__init__(jl.EMResult(entropy, _distribution))

    def __repr__(self):
        return f"EMResult(entropy={self.entropy}, joint_probability={repr(self.joint_probability)})"

    @property
    def joint_probability(self):
        return np.array(self.julia_obj.joint_probability)

    @property
    def marginal_entropies(self):
        raise NotImplementedError("marginal_entropies not implemented for EMResult")


class EMFMEResult(EResult):
    def __init__(
        self,
        entropy: float | AnyValue,
        marginal_entropies: None | Dict[tuple[int, ...], float] = None,
    ):
        if jl.Base.isa(entropy, jl.EMFMEResult):
            super().__init__(entropy)
        elif isinstance(entropy, AnyValue):
            raise ValueError(
                f"Cannot create EMFMEResult from {jl.Base.typeof(entropy)}"
            )
        else:
            marginal_entropies = cast(Dict[tuple[int, ...], float], marginal_entropies)
            dimension = max([len(k) for k in marginal_entropies.keys()])
            _marginal_entropies = _format_precalculated_entropies(
                marginal_entropies, dimension
            )
            super().__init__(jl.EMFMEResult(entropy, _marginal_entropies))

    def __repr__(self):
        return f"EMFMEResult(entropy={self.entropy}, marginal_entropies={self.marginal_entropies})"

    @property
    def marginal_entropies(self):
        return {
            tuple(k): float(v)
            for k, v in dict(self.julia_obj.marginal_entropies).items()
        }

    @property
    def joint_probability(self):
        raise NotImplementedError("joint_probability not implemented for EMFMEResult")


def _convert_EResult(result: AnyValue) -> EResult:
    if jl.Base.isa(result, jl.EMFMEResult):
        return EMFMEResult(result)
    elif jl.Base.isa(result, jl.EMResult):
        return EMResult(result)
    else:
        raise ValueError(f"Cannot convert {jl.Base.typeof(result)} to EResult")


def _convert_EResultDict(result: Dict[int, AnyValue]) -> Dict[int, EResult]:
    if jl.Base.isa(next(iter(result.values())), jl.EMFMEResult):
        format = EMFMEResult
    elif jl.Base.isa(next(iter(result.values())), jl.EMResult):
        format = EMResult
    else:
        raise ValueError(
            f"Cannot convert {jl.Base.typeof(next(iter(result.values())))} to EResult"
        )

    return {k: format(v) for k, v in result.items()}


def _get_julia_distribution(
    distribution: np.ndarray | EMResult,
) -> Tuple[jl.Array, int, bool]:
    if isinstance(distribution, EMResult):
        _distribution = distribution.julia_obj.joint_probability
        dimension = jl.Base.ndims(distribution)
        if jl.Base.isa(distribution, jl.Array[jl.AbstractFloat, dimension]):
            dist_is_float = True
        elif jl.Base.isa(distribution, jl.Array[jl.Integer, dimension]):
            dist_is_float = False
        else:
            raise ValueError(
                f"Cannot optimise a distribution with type '{jl.Base.typeof(distribution.julia_obj)}'."
            )
    elif isinstance(distribution, np.ndarray):
        dimension = len(distribution.shape)
        if np.issubdtype(distribution.dtype, np.floating):
            _distribution = convert(jl.Array[jl.Float64, dimension], distribution)
            dist_is_float = True
        elif np.issubdtype(distribution.dtype, np.integer):
            _distribution = convert(jl.Array[jl.Int64, dimension], distribution)
            dist_is_float = False
        else:
            raise ValueError(
                f"Cannot optimise a distribution with dtype '{distribution.dtype}'."
            )
    else:
        raise ValueError(
            f"Cannot optimise a distribution with type '{type(distribution)}'."
        )

    return _distribution, dimension, dist_is_float


def connected_information(
    distribution: np.ndarray | EMResult,
    orders: np.ndarray | list[int] | int,
    method: OptimisationMethod | None = None,
    precalculated_entropies: None | dict[tuple[int, ...], float] | EMFMEResult = None,
    full_output: bool = False,
) -> tuple[dict[int, float], dict[int, EResult] | None]:
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
    _distribution, dimension, dist_is_float = _get_julia_distribution(distribution)

    if isinstance(method, OptimisationMethod):
        if isinstance(method, GPolymatroid) and dist_is_float:
            raise ValueError(
                "Cannot use GPolymatroid method with floating point distribution."
            )
    elif method is None:
        if dist_is_float:
            method = Ipfp()
        else:
            method = RawPolymatroid()
    else:
        raise ValueError(f"Unrecognised method of type '{type(method)}'.")

    extras = {"full_output": convert(jl.Bool, full_output)}
    if precalculated_entropies is not None and isinstance(method, EntropyMethod):
        extras["precalculated_entropies"] = _format_precalculated_entropies(
            precalculated_entropies, dimension
        )

    if isinstance(orders, (np.ndarray, list)):
        _orders = convert(jl.Vector, np.array(orders).astype(int))
    else:
        _orders = convert(jl.Int64, orders)

    CI = jl.connected_information(_distribution, _orders, method.method, **extras)

    if full_output:
        return dict(CI[0]), _convert_EResultDict(CI[1])
    else:
        return dict(CI[0]), None


def maximise_entropy(
    distribution: np.ndarray | EMResult,
    order: int,
    method: OptimisationMethod | None = None,
    precalculated_entropies: None | dict[tuple[int, ...], float] | EMFMEResult = None,
) -> EResult:
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
    _distribution, dimension, dist_is_float = _get_julia_distribution(distribution)

    if isinstance(method, OptimisationMethod):
        if isinstance(method, GPolymatroid) and dist_is_float:
            raise ValueError(
                "Cannot use GPolymatroid method with floating point distribution."
            )
    elif method is None:
        if dist_is_float:
            method = Ipfp()
        else:
            method = RawPolymatroid()
    else:
        raise ValueError(f"Unrecognise method of type '{type(method)}'.")

    _order = convert(jl.Int64, order)

    extras = {}
    if precalculated_entropies is not None and isinstance(method, EntropyMethod):
        extras["precalculated_entropies"] = _format_precalculated_entropies(
            precalculated_entropies, dimension
        )

    max_ent = jl.maximise_entropy(_distribution, _order, method.method, **extras)
    return _convert_EResult(max_ent)


def distribution_entropy(distribution: np.ndarray) -> float:
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
