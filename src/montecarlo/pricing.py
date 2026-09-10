"""European option pricing, and the error bars that belong with it.

A Monte Carlo price without an error bar is not a price, it is a draw. This
module is built around that: every estimate carries a standard error computed
the way its own sampling scheme demands, and refuses to report one when the
scheme does not admit a valid estimate from a single run.

Three cases, three different answers:

* **Pseudo-random.** Paths are iid. The textbook ``s / sqrt(n)`` is correct.
* **Antithetic.** Paths come in negatively correlated pairs, so they are not
  iid and ``s / sqrt(n)`` is wrong — it reads the induced correlation as noise
  and reports an interval wider than the truth. The valid estimator treats each
  *pair average* as one observation and takes ``s_pair / sqrt(n/2)``.
* **Scrambled Sobol.** The points are neither independent nor exchangeable, and
  no unbiased error estimate exists from one replication. Getting one requires
  several independent scrambles. This module reports ``None`` rather than a
  number that would look like an error bar without being one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .config import Draws
from .processes import PathSet

_Z95 = 1.959963984540054


@dataclass
class Estimate:
    """A Monte Carlo estimate with its uncertainty, or an honest absence of it."""

    value: float
    std_error: float | None
    n_paths: int
    scheme: str
    note: str = ""

    @property
    def ci95(self) -> tuple[float, float] | None:
        """95% confidence interval, or ``None`` when no valid error exists."""
        if self.std_error is None:
            return None
        half = _Z95 * self.std_error
        return (self.value - half, self.value + half)

    def __str__(self) -> str:
        if self.std_error is None:
            return f"{self.value:.6f}  (no valid single-run error: {self.note})"
        low, high = self.ci95  # type: ignore[misc]
        return f"{self.value:.6f} ± {_Z95 * self.std_error:.6f}  95% CI [{low:.6f}, {high:.6f}]"


def _mean_and_error(samples: np.ndarray, scheme: Draws) -> tuple[float, float | None, str]:
    """Mean of the samples, with the error estimator that scheme actually allows."""
    n = samples.size
    mean = float(np.mean(samples))

    if scheme is Draws.PSEUDO:
        return mean, float(np.std(samples, ddof=1) / math.sqrt(n)), ""

    if scheme is Draws.ANTITHETIC:
        # The draws were stacked as [base; -base], so path i and path i + n/2 are
        # the two halves of one pair. Averaging within a pair restores
        # independence across pairs, which is what the CLT needs.
        half = n // 2
        pair_means = 0.5 * (samples[:half] + samples[half : 2 * half])
        error = float(np.std(pair_means, ddof=1) / math.sqrt(half))
        return mean, error, "error computed across antithetic pairs, not paths"

    if scheme is Draws.SOBOL:
        return (
            mean,
            None,
            "scrambled QMC needs independent replications for a valid error estimate",
        )

    raise ValueError(f"unknown scheme: {scheme}")


def payoff_european(terminal: np.ndarray, strike: float, kind: str) -> np.ndarray:
    """Terminal payoff of a European call or put."""
    if kind == "call":
        return np.maximum(terminal - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - terminal, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def price_european(
    path_set: PathSet,
    strike: float,
    rate: float,
    kind: str = "call",
    control_variate: bool = False,
) -> Estimate:
    """Price a European option from simulated paths.

    The paths must have been generated under the risk-neutral measure — that is,
    with ``drift`` set to ``rate``. Simulating under the real-world drift and
    discounting at the risk-free rate is the most common way to get a Monte
    Carlo option price quietly wrong, so the mismatch raises rather than warns.

    Args:
        path_set: Simulated paths.
        strike: Strike price.
        rate: Risk-free rate, used for discounting.
        kind: ``"call"`` or ``"put"``.
        control_variate: Use the terminal spot as a control. Its expectation
            ``S0 exp(rT)`` is known exactly, and it correlates strongly with the
            payoff, so subtracting its error subtracts much of the payoff's.

    Returns:
        The discounted estimate, with a standard error when one is valid.
    """
    drift = path_set.params.get("drift")
    if drift is not None and not math.isclose(drift, rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"paths were simulated with drift={drift} but priced at rate={rate}; "
            "risk-neutral pricing requires drift == rate"
        )

    horizon = path_set.config.horizon
    discount = math.exp(-rate * horizon)
    terminal = path_set.terminal
    samples = discount * payoff_european(terminal, strike, kind)

    if control_variate:
        # Y = X - b (C - E[C]), with b = Cov(X, C) / Var(C).
        control = discount * terminal
        expected_control = path_set.params["spot"]  # discount * S0 * exp(rT) == S0
        var_control = float(np.var(control, ddof=1))
        if var_control > 0:
            beta = float(np.cov(samples, control, ddof=1)[0, 1] / var_control)
            samples = samples - beta * (control - expected_control)

    value, error, note = _mean_and_error(samples, path_set.config.draws)
    if control_variate and note:
        note = f"{note}; control variate applied"
    elif control_variate:
        note = "control variate applied"

    return Estimate(
        value=value,
        std_error=error,
        n_paths=path_set.n_paths,
        scheme=path_set.config.draws.value,
        note=note,
    )


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    horizon: float,
    kind: str = "call",
) -> float:
    """Closed-form Black-Scholes price.

    Present so the simulator can be checked against a known answer rather than
    against itself. A Monte Carlo engine that is never compared to an analytic
    benchmark is an engine nobody has tested.
    """
    if horizon <= 0 or vol <= 0:
        return float(payoff_european(np.array([spot]), strike, kind)[0])

    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * horizon) / (vol * math.sqrt(horizon))
    d2 = d1 - vol * math.sqrt(horizon)
    discount = math.exp(-rate * horizon)

    if kind == "call":
        return spot * stats.norm.cdf(d1) - strike * discount * stats.norm.cdf(d2)
    if kind == "put":
        return strike * discount * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")
