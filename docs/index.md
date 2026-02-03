# pyHordcoin - Python interface for Hordcoin.jl

This is a python interface for the Julia package Hordcoin.jl which provides methods for finding probability distributions with maximal Shannon entropy given a fixed marginal distribution or entropy up to a chosen order, and to compute the Connected Information. 

## Installing

Installation is easy, just:
1. (optional but very advised) create and activate a new virtual environment:
    ```bash
    pyhton -m venv hordcoin
    source hordcoin/bin/activate
    ```
2. change directory to the unpacked repository:
    ```bash
    cd /path/to/pyHordcoin
    ```
1. install the package:
    1. if you only want to use its components for your measures:
        ```bash
        pip install .
        ```
    1. if you want to contribute to the development: 
        ```bash
        pip install .[dev]
        ```

## Running
The first time you import the package it will take some time to install the Julia dependencies (and possibly Julia itself), it might take a few minutes. From the second import on it's much faster, although the import of Julia still takes some time. This will be repaid during the maximisation of the entropies.

## Usage
This package offers an interface to Hordcoin.jl which implements methods that maximize the Shannon entropy of a probability distribution with marginal distribution or entropic constraints and compute the Connected Information.

The input data must satisfy the following requirements:
- The probability distributions are stored as multidimensional Numpy arrays;
- Probabilities are non-negative and sum up to 1;
- OR are provided as (non-normalised) counts;
- The maximal order of the fixed marginal distributions has to be in [2, n-1], where n is the number of dimensions of the probability distribution.

### Connected Information

The main function of the package is `ConnectedInformation` that uses the the maximum entropy with constraints at different orders to compute the Connected Information. It takes as input the probability distribution or the (non-normalised) counts, along with the desired orders of Connected Information and the optimisation method.

When computing multiple Connected Information values for the same probability distribution, it is possible to pass the sizes (desired orders) as an array. This will speed up the process by chaining the computations, thereby reducing the number of maximizations.

If no method is passed, the kind of optimisation is decided by the data type of the input:
- `int` input triggers constraints on the marginal entropy and the `RawPolymatroid` method;
- `float` input triggers constraints on the marginal distributions and the `Ipfp` method.

It is possible to have complete control on the kind of constraints by passing a method explicitly:
- `Gradient`, `Cone` and `Ipfp` trigger constraints on the marginal distributions with both `int` and `float` inputs;
- `RawPolymatroid` and `GPolymatroid` require constraints on the marginal entropy. However, `GPolymatroid` will raise an error if used with `float` inputs as it needs the counts to compute the correction.

It's possible to pass a precomputed dictionary of entropies to speed up the computation when using entropic constraints. See note below on the structure of this dictionary.


The basic usage of `ConnectedInformation` is the following:
```Python
import pyHordcoin as hc
import numpy as np

counts=np.array([[[1, 2],[3, 4]], [[4, 2], [1, 3]]], dtype=int)
hc.ConnectedInformation(counts, 2)
```
Which will optimise (maximize entropy) constraining the marginal entropies (up to order 2) and should give a result similar to `{2: 0.09310598013744764}`

Notably, the following operations all give the same results:
```Python
hc.ConnectedInformation(counts, [2])
hc.ConnectedInformation(counts, 2, hc.RawPolymatroid())

hc.frequencies = counts.astype(float) ./ sum(counts)
hc.ConnectedInformation(frequencies, 2, hc.RawPolymatroid())
```

Alternatively, it's possible to trigger the marginal distribution constraints with these equivalent lines:
```Python
hc.ConnectedInformation(frequencies, 2)
hc.ConnectedInformation(counts, 2, hc.Ipfp())
hc.ConnectedInformation(frequencies, [2], hc.Ipfp())
```

Or similar results with:
```Python
hc.ConnectedInformation(frequencies, 2, hc.Gradient())
hc.ConnectedInformation(frequencies, 2, hc.Cone())
hc.ConnectedInformation(frequencies, 2, hc.Cone(hc.SCS()))
hc.ConnectedInformation(frequencies, 2, hc.Cone(hc.Mosek()))
```
Where the last one requires a Mosek license. (Academic licence easy to obtain at https://www.mosek.com/products/academic-licenses/).

Other useful parameters for the Polymatroid methods are:
- zhang_yeung: to enable the Zhang-Yeung inequalities complementing the Shannon inequalities and improving the approximation at higher orders (see paper),
- optimizer: to chose between the `SCS` and the `Mosek` optimiser
- mle_correction: (only `RawPolymatroid`) enables a rough correction for the finite sample
- tolerance: (only `GPolymatroid`) enables a relaxation of the constraints to help convergence (sometimes if fails with the corrected entropies). **Note**: CI estimate can become negative due to the relaxed constraints.

### Other functions

This interface currently implements a single function to access the entropy maximisation. The function `MaximiseEntropy` works for marginal constraints entropic constraints selecting the appropriate set of constraints using the same rules as `ConnectedInformation`  (with the exception that fixed entropy maximisation from a normalised distribution is not allowed). `MaximiseEntropy` takes as an input a probability distribution and the order of marginal distributions to constrain (or the order up to which the marginal entropies must be fixed). The optimiser is an optional parameter that can have further specified parameters (such as the number of iterations, etc.). The function returns the maximum entropy and, in case of fixed marginals, the probability distribution with maximal entropy as an `np.ndarray`. It's possible to pass a precomputed dictionary of entropies to speed up the computation when using entropic constraints.

The basic usage is the following:
```Python
import pyHordcoin as hc

probability_distribution = np.array([[[1/16, 3/16], [3/16, 1/16]], [[1/16, 3/16], [3/16, 1/16]]])
marginal_size = 2
hc.MaximiseEntropy(probability_distribution, marginal_size)
```
Running the code with the optional parameter `method`:
```Python
hc.MaximiseEntropy(probability_distribution, marginal_size, method = hc.Gradient(10, hc.SCS()))
```

The package also contains one utility function: `DistributionEntropy` computes the information entropy of a probability distribution.

Usage of the functions:
```Python
hc.DistributionEntropy(probability_distribution)
```

#### NOTES

- All the entropies are measured in bits. This applies also to the precalculated entropies provided by the user.

- The precalculated entropies must be stored in a dictionary whose keys are tuples of dimensions indexed from 1. These are the dimensions kept in the marginal from which the entropy is calculated. In a 4-D distribution, suppose I sum away dimensions 2 and 4 (`marginal=distribution.sum((2,4,))`), then the dictionary will contain `{(1,3,):-np.sum(marginal*np.log2(marginal))}`

A full example with precalculated entropies may look like this:
```python
from itertools import combinations

A = np.random.randint(1000, size=[2, 2, 2]).astype(np.float64)
A /= A.sum()

marginal_entropies = {}
for i in range(3):
    for a in combinations(range(3), i):
        m = tuple(set(range(3)) - set(a))
        tmp = A.sum(a)
        k = tuple(b + 1 for b in m)
        marginal_entropies[k] = hc.DistributionEntropy(tmp)

hc.ConnectedInformation(
    A, 2, hc.RawPolymatroid(), precalculated_entropies=marginal_entropies
)

```


## Recommendations

The most efficient method when computing with fixed marginal distributions is the `Cone` method with `Mosek` optimiser. This requires a license to use the MOSEK solver. Without the license, it is possible to use `SCS` instead, but it is less accurate and slower.

Without a MOSEK license, use the `Ipfp` method (default). It is accurate and not the slowest. It can also be parametrized with the number of iterations, but it is not necessary. The default value is 10.

The `Gradient` method is the slowest and may fail during execution due to limitations of Second Order Cone constraints in solvers.

When computing with fixed entropies and a small number of samples, the recommended method is the `GPolymatroid` with `Mosek`. When the distribution is sampled enough, you can use `RawPolymatroid` to estimate the entropy with the plug-in estimator. More information can be found in the paper.


## How to cite

If you use this code for a scientific publication, please cite:

> Tani Raffaelli G., Kislinger J., Kroupa T., and Hlinka J., "HORDCOIN: A Software Library for Higher Order Connected Information and Entropic Constraints Approximation"
