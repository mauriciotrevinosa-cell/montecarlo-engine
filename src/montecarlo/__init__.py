"""A small Monte Carlo engine for path-dependent asset simulation.

Reduced showcase extract. See README.md for what was left out and why.
"""

from .config import Draws, SimulationConfig
from .draws import standard_normals
from .pricing import Estimate, black_scholes, payoff_european, price_european
from .processes import PathSet, gbm, heston, merton_jump_diffusion

__all__ = [
    "Draws",
    "SimulationConfig",
    "standard_normals",
    "PathSet",
    "gbm",
    "heston",
    "merton_jump_diffusion",
    "Estimate",
    "price_european",
    "payoff_european",
    "black_scholes",
]

__version__ = "0.1.0"
