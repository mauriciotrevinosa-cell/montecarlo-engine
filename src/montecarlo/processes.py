"""Stochastic price processes.

Three processes, each a closed-form or Euler discretisation driven by the
samplers in `draws.py`. All of them are fully vectorised: the loop over time
steps only survives where the scheme genuinely needs the previous state
(Heston's variance), and never over paths.

Each returns a `PathSet` that carries the paths together with the config that
produced them, so a downstream number can always be traced back to its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Draws, SimulationConfig
from .draws import standard_normals


@dataclass
class PathSet:
    """Simulated paths plus the settings that generated them."""

    paths: np.ndarray  # shape (n_paths, n_steps + 1), column 0 is spot
    config: SimulationConfig
    process: str
    params: dict = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        """Paths actually produced, which may exceed the count requested."""
        return int(self.paths.shape[0])

    @property
    def terminal(self) -> np.ndarray:
        """Value at the horizon, one per path."""
        return self.paths[:, -1]

    @property
    def times(self) -> np.ndarray:
        """Time grid in years, aligned with the path columns."""
        return np.linspace(0.0, self.config.horizon, self.config.n_steps + 1)


def _rng(config: SimulationConfig) -> np.random.Generator:
    return np.random.default_rng(config.seed)


def _prepend_spot(spot: float, log_increments: np.ndarray) -> np.ndarray:
    """Turn per-step log returns into a price path starting at ``spot``."""
    log_paths = np.cumsum(log_increments, axis=1)
    paths = spot * np.exp(log_paths)
    return np.hstack([np.full((paths.shape[0], 1), spot), paths])


def gbm(
    config: SimulationConfig,
    spot: float,
    drift: float,
    vol: float,
) -> PathSet:
    """Geometric Brownian motion.

        dS = mu S dt + sigma S dW

    Simulated from the exact solution rather than an Euler step, so the
    discretisation carries no bias at any step size:

        S(t) = S(0) exp((mu - sigma^2 / 2) t + sigma W(t))
    """
    if vol < 0:
        raise ValueError("vol must be non-negative")

    dt = config.dt
    z = standard_normals(config.n_paths, config.n_steps, config.draws, _rng(config), config.seed)
    increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    return PathSet(
        paths=_prepend_spot(spot, increments),
        config=config,
        process="gbm",
        params={"spot": spot, "drift": drift, "vol": vol},
    )


def merton_jump_diffusion(
    config: SimulationConfig,
    spot: float,
    drift: float,
    vol: float,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
) -> PathSet:
    """Merton jump-diffusion.

    A GBM whose log price also takes Poisson-timed normal jumps. The drift
    carries a compensator so that the expected terminal value still equals
    ``spot * exp(drift * horizon)``; without it, adding jumps silently changes
    the mean and every comparison against GBM becomes meaningless.

    The number of jumps in a step is Poisson; conditional on that count the sum
    of the jumps is normal, which is what lets the whole thing stay vectorised
    instead of looping per path as the naive implementation does.
    """
    if jump_intensity < 0:
        raise ValueError("jump_intensity must be non-negative")

    dt = config.dt
    rng = _rng(config)
    z = standard_normals(config.n_paths, config.n_steps, config.draws, rng, config.seed)
    n_paths = z.shape[0]

    # E[e^J - 1] for a lognormal jump: the drift correction that keeps the mean put.
    compensator = np.exp(jump_mean + 0.5 * jump_vol**2) - 1.0

    counts = rng.poisson(jump_intensity * dt, size=(n_paths, config.n_steps))
    jump_noise = rng.standard_normal((n_paths, config.n_steps))
    # Sum of N iid Normal(m, s^2) is Normal(N m, N s^2).
    jumps = counts * jump_mean + np.sqrt(counts) * jump_vol * jump_noise

    increments = (
        (drift - 0.5 * vol**2 - jump_intensity * compensator) * dt
        + vol * np.sqrt(dt) * z
        + jumps
    )
    return PathSet(
        paths=_prepend_spot(spot, increments),
        config=config,
        process="merton_jump_diffusion",
        params={
            "spot": spot,
            "drift": drift,
            "vol": vol,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_vol": jump_vol,
        },
    )


def heston(
    config: SimulationConfig,
    spot: float,
    drift: float,
    var0: float,
    kappa: float,
    theta: float,
    vol_of_vol: float,
    rho: float,
) -> PathSet:
    """Heston stochastic volatility, full-truncation Euler.

        dS = mu S dt + sqrt(v) S dW1
        dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

    Full truncation means the variance is floored at zero everywhere it is read
    but is allowed to go negative in its own state before being floored. The
    alternative — reflecting it — is what most quick implementations do, and it
    biases the variance upward. Whether the Feller condition
    ``2 kappa theta > xi^2`` holds is reported rather than enforced, because
    calibrated parameter sets routinely violate it and the scheme still works.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    dt = config.dt
    sqrt_dt = np.sqrt(dt)
    rng = _rng(config)

    # Two correlated factors need twice the dimension; splitting one wide draw
    # keeps Sobol's balance across the pair instead of across each half alone.
    wide = standard_normals(config.n_paths, 2 * config.n_steps, config.draws, rng, config.seed)
    z1 = wide[:, : config.n_steps]
    z2 = rho * z1 + np.sqrt(1.0 - rho**2) * wide[:, config.n_steps :]

    n_paths = wide.shape[0]
    prices = np.empty((n_paths, config.n_steps + 1))
    prices[:, 0] = spot
    variance = np.full(n_paths, float(var0))

    for t in range(config.n_steps):
        v_plus = np.maximum(variance, 0.0)
        prices[:, t + 1] = prices[:, t] * np.exp(
            (drift - 0.5 * v_plus) * dt + np.sqrt(v_plus) * sqrt_dt * z1[:, t]
        )
        variance = (
            variance
            + kappa * (theta - v_plus) * dt
            + vol_of_vol * np.sqrt(v_plus) * sqrt_dt * z2[:, t]
        )

    return PathSet(
        paths=prices,
        config=config,
        process="heston",
        params={
            "spot": spot,
            "drift": drift,
            "var0": var0,
            "kappa": kappa,
            "theta": theta,
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "feller_satisfied": bool(2 * kappa * theta > vol_of_vol**2),
        },
    )
