"""Pricing is checked against Black-Scholes, and the error bars against theory."""

import math

import numpy as np
import pytest

from montecarlo import (
    Draws,
    SimulationConfig,
    black_scholes,
    gbm,
    price_european,
)

SPOT, STRIKE, RATE, VOL, HORIZON = 100.0, 105.0, 0.03, 0.25, 1.0


def _paths(scheme: Draws, n_paths: int = 100_000, seed: int = 11):
    config = SimulationConfig(
        n_paths=n_paths, n_steps=1, horizon=HORIZON, draws=scheme, seed=seed
    )
    # One step is enough: GBM is simulated from its exact solution, so the
    # terminal distribution is exact regardless of how many steps are taken.
    return gbm(config, spot=SPOT, drift=RATE, vol=VOL)


def test_monte_carlo_call_matches_black_scholes():
    analytic = black_scholes(SPOT, STRIKE, RATE, VOL, HORIZON, "call")
    estimate = price_european(_paths(Draws.ANTITHETIC), STRIKE, RATE, "call")

    low, high = estimate.ci95
    assert low < analytic < high, (
        f"analytic {analytic:.4f} fell outside the 95% CI [{low:.4f}, {high:.4f}]"
    )


def test_monte_carlo_put_matches_black_scholes():
    analytic = black_scholes(SPOT, STRIKE, RATE, VOL, HORIZON, "put")
    estimate = price_european(_paths(Draws.ANTITHETIC, seed=12), STRIKE, RATE, "put")

    low, high = estimate.ci95
    assert low < analytic < high


def test_put_call_parity_holds_on_the_same_paths():
    """C - P = S - K exp(-rT), and it must hold path by path, not just on average."""
    path_set = _paths(Draws.ANTITHETIC, seed=13)
    call = price_european(path_set, STRIKE, RATE, "call").value
    put = price_european(path_set, STRIKE, RATE, "put").value

    parity = SPOT - STRIKE * math.exp(-RATE * HORIZON)
    # Not exact: the simulated E[S_T] carries sampling error, and both legs
    # inherit it. The residual is that error, so it is bounded, not zero.
    assert call - put == pytest.approx(parity, abs=0.05)


def test_antithetic_beats_pseudo_random():
    """The whole point of antithetic sampling is a smaller error at equal cost."""
    pseudo = price_european(_paths(Draws.PSEUDO, seed=21), STRIKE, RATE, "call")
    anti = price_european(_paths(Draws.ANTITHETIC, seed=21), STRIKE, RATE, "call")

    assert anti.std_error < pseudo.std_error
    variance_ratio = (pseudo.std_error / anti.std_error) ** 2
    assert variance_ratio > 1.2, f"only {variance_ratio:.2f}x variance reduction"


def test_control_variate_beats_plain_sampling():
    path_set = _paths(Draws.PSEUDO, seed=22)
    plain = price_european(path_set, STRIKE, RATE, "call")
    controlled = price_european(path_set, STRIKE, RATE, "call", control_variate=True)

    variance_ratio = (plain.std_error / controlled.std_error) ** 2
    assert variance_ratio > 2.0, f"only {variance_ratio:.2f}x variance reduction"
    assert "control variate applied" in controlled.note


def test_antithetic_error_is_computed_across_pairs_not_paths():
    """The naive s/sqrt(n) is wrong here, and must not be what is reported."""
    path_set = _paths(Draws.ANTITHETIC, seed=23)
    estimate = price_european(path_set, STRIKE, RATE, "call")

    discount = math.exp(-RATE * HORIZON)
    samples = discount * np.maximum(path_set.terminal - STRIKE, 0.0)
    naive = float(np.std(samples, ddof=1)) / math.sqrt(samples.size)

    assert estimate.std_error != pytest.approx(naive, rel=1e-6)
    assert estimate.std_error < naive, "pairing should tighten the interval, not widen it"
    assert "antithetic pairs" in estimate.note


def test_sobol_declines_to_report_an_error_it_cannot_justify():
    estimate = price_european(_paths(Draws.SOBOL, seed=24), STRIKE, RATE, "call")

    assert estimate.std_error is None
    assert estimate.ci95 is None
    assert "replications" in estimate.note
    # The point estimate is still good — it is only the error bar that is absent.
    analytic = black_scholes(SPOT, STRIKE, RATE, VOL, HORIZON, "call")
    assert estimate.value == pytest.approx(analytic, rel=0.02)


def test_pricing_off_measure_is_refused():
    """Simulating under the real-world drift and discounting at r is a silent bug."""
    config = SimulationConfig(n_paths=1_000, n_steps=1, horizon=HORIZON, seed=25)
    real_world = gbm(config, spot=SPOT, drift=0.12, vol=VOL)

    with pytest.raises(ValueError, match="risk-neutral pricing requires drift == rate"):
        price_european(real_world, STRIKE, RATE, "call")


def test_deep_out_of_the_money_call_is_worth_almost_nothing():
    estimate = price_european(_paths(Draws.ANTITHETIC, seed=26), strike=400.0, rate=RATE)
    analytic = black_scholes(SPOT, 400.0, RATE, VOL, HORIZON, "call")

    assert estimate.value < 0.05
    assert analytic < 0.05


def test_seed_makes_a_run_reproducible():
    first = price_european(_paths(Draws.ANTITHETIC, seed=99), STRIKE, RATE).value
    second = price_european(_paths(Draws.ANTITHETIC, seed=99), STRIKE, RATE).value
    assert first == second
