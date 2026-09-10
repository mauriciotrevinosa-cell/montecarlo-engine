"""Normal increment generators.

Three samplers, one signature. Each returns an array of shape
``(n_paths, n_steps)`` of standard normal variates; scaling by ``sqrt(dt)`` is
the caller's job, because a process may need to scale differently per factor.

The returned path count may differ from the requested one — antithetic sampling
needs an even count, Sobol wants a power of two — so every generator reports the
count it actually produced instead of quietly resizing the caller's expectation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .config import Draws


def _next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def standard_normals(
    n_paths: int,
    n_steps: int,
    scheme: Draws,
    rng: np.random.Generator,
    seed: int | None = None,
) -> np.ndarray:
    """Draw standard normals under the requested scheme.

    Args:
        n_paths: Requested number of paths.
        n_steps: Steps per path.
        scheme: Sampling scheme.
        rng: Generator used for pseudo-random and antithetic schemes.
        seed: Passed to the Sobol engine, which keeps its own state.

    Returns:
        Array of shape ``(m, n_steps)`` where ``m >= n_paths``.
    """
    if scheme is Draws.PSEUDO:
        return rng.standard_normal((n_paths, n_steps))

    if scheme is Draws.ANTITHETIC:
        # Every draw is used twice, once with each sign. The estimator's two
        # halves are negatively correlated, which is where the variance goes.
        half = (n_paths + 1) // 2
        base = rng.standard_normal((half, n_steps))
        return np.vstack([base, -base])

    if scheme is Draws.SOBOL:
        # Sobol's equidistribution guarantees hold on full powers of two; asking
        # for 10,000 points of a 2^14 sequence throws the balance away, so round
        # up rather than truncate.
        m = int(np.log2(_next_power_of_two(n_paths)))
        engine = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
        uniforms = engine.random_base2(m=m)
        # Scrambling keeps the points off the boundary, but clip anyway: a single
        # exact 0 or 1 becomes an infinite normal and poisons the whole estimate.
        uniforms = np.clip(uniforms, 1e-12, 1 - 1e-12)
        return stats.norm.ppf(uniforms)

    raise ValueError(f"unknown draw scheme: {scheme}")
