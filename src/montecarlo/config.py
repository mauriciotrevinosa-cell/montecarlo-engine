"""Simulation configuration.

The config is a frozen dataclass so that a result can carry the exact settings
that produced it. A Monte Carlo number without its draw count, its horizon and
its seed is not reproducible, and a number that cannot be reproduced cannot be
checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Draws(str, Enum):
    """How the normal increments are generated.

    This is a property of the *sampler*, not of the stochastic process, which is
    why it lives here rather than inside the process definitions. Any process in
    `processes.py` can be driven by any of these.
    """

    PSEUDO = "pseudo_random"
    ANTITHETIC = "antithetic_variates"
    SOBOL = "sobol_scrambled"


@dataclass(frozen=True)
class SimulationConfig:
    """Everything needed to reproduce a run.

    Attributes:
        n_paths: Number of simulated paths. Under ANTITHETIC this is rounded up
            to the nearest even number; under SOBOL, up to the next power of two.
            Both adjustments are reported back on the result rather than applied
            silently.
        n_steps: Time steps per path.
        horizon: Total simulated time in years.
        draws: Sampling scheme.
        seed: Fixed for reproducibility. ``None`` means non-deterministic, which
            is fine for exploration and wrong for anything reported.
    """

    n_paths: int = 10_000
    n_steps: int = 252
    horizon: float = 1.0
    draws: Draws = Draws.ANTITHETIC
    seed: int | None = 7

    def __post_init__(self) -> None:
        if self.n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

    @property
    def dt(self) -> float:
        """Length of one step, in years."""
        return self.horizon / self.n_steps
