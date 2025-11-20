"""Sampling methods for generating points in domains."""

__all__ = ["sample"]

import numpy as np

def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError(f"{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Generate pseudorandom samples."""
    return np.random.random(size=(n_samples, dimension))


def quasirandom(n_samples, dimension, sampler):
    """Generate quasirandom samples using various methods."""
    try:
        import skopt
    except ImportError:
        print("Warning: scikit-optimize not installed. Falling back to pseudorandom.")
        return pseudorandom(n_samples, dimension)
    
    skip = 0
    if sampler == "LHS":
        sampler_obj = skopt.sampler.Lhs()
    elif sampler == "Halton":
        sampler_obj = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Sobol":
        sampler_obj = skopt.sampler.Sobol(randomize=False)
        skip = 2 if dimension >= 3 else 1
    
    space = [(0.0, 1.0)] * dimension
    return np.asarray(sampler_obj.generate(space, n_samples + skip)[skip:])
