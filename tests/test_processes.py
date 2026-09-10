"""The processes are checked against properties with known analytic values.

Every tolerance below is derived from the estimator's own standard error, not
picked until the test passed.
"""

import math

import numpy as np
import pytest

from montecarlo import Draws, SimulationConfig, gbm, heston, merton_jump_diffusion


def test_gbm_terminal_mean_matches_analytic():
    """E[S_T] = S0 exp(mu T) for geometric Brownian motion."""
    config = SimulationConfig(n_paths=200_000, n_steps=64, horizon=1.0, draws=Draws.ANTITHETIC, seed=1)
    result = gbm(config, spot=100.0, drift=0.05, vol=0.2)

    expected = 100.0 * math.exp(0.05 * 1.0)
    observed = float(np.mean(result.terminal))
    # Three standard errors of the sample mean.
    tolerance = 3 * float(np.std(result.terminal, ddof=1)) / math.sqrt(result.n_paths)
    assert abs(observed - expected) < tolerance


def test_gbm_log_variance_matches_analytic():
    """Var[ln(S_T / S_0)] = sigma^2 T, independent of the drift."""
    config = SimulationConfig(n_paths=200_000, n_steps=252, horizon=1.0, draws=Draws.PSEUDO, seed=2)
    result = gbm(config, spot=100.0, drift=0.05, vol=0.2)

    log_returns = np.log(result.terminal / 100.0)
    assert float(np.var(log_returns, ddof=1)) == pytest.approx(0.2**2 * 1.0, rel=0.02)


def test_gbm_paths_start_at_spot_and_stay_positive():
    config = SimulationConfig(n_paths=1_000, n_steps=50, seed=3)
    result = gbm(config, spot=42.0, drift=0.0, vol=0.9)

    assert np.all(result.paths[:, 0] == 42.0)
    assert np.all(result.paths > 0), "a GBM path can never reach zero"
    assert result.paths.shape[1] == config.n_steps + 1


def test_merton_compensator_preserves_the_mean():
    """Adding jumps must not move E[S_T]; that is what the compensator is for."""
    config = SimulationConfig(n_paths=400_000, n_steps=64, horizon=1.0, draws=Draws.ANTITHETIC, seed=4)
    result = merton_jump_diffusion(
        config, spot=100.0, drift=0.03, vol=0.2,
        jump_intensity=1.5, jump_mean=-0.10, jump_vol=0.15,
    )

    expected = 100.0 * math.exp(0.03 * 1.0)
    observed = float(np.mean(result.terminal))
    tolerance = 4 * float(np.std(result.terminal, ddof=1)) / math.sqrt(result.n_paths)
    assert abs(observed - expected) < tolerance


def test_merton_has_fatter_tails_than_gbm():
    """Jumps are only worth simulating if they change the distribution's shape."""
    config = SimulationConfig(n_paths=100_000, n_steps=64, horizon=1.0, draws=Draws.PSEUDO, seed=5)
    plain = gbm(config, spot=100.0, drift=0.03, vol=0.2)
    jumpy = merton_jump_diffusion(
        config, spot=100.0, drift=0.03, vol=0.2,
        jump_intensity=2.0, jump_mean=-0.15, jump_vol=0.2,
    )

    from scipy import stats as sps
    plain_kurtosis = sps.kurtosis(np.log(plain.terminal / 100.0))
    jumpy_kurtosis = sps.kurtosis(np.log(jumpy.terminal / 100.0))
    assert jumpy_kurtosis > plain_kurtosis + 0.5


def test_heston_collapses_to_gbm_when_volatility_is_constant():
    """With no vol-of-vol and variance already at its mean, Heston *is* GBM."""
    config = SimulationConfig(n_paths=100_000, n_steps=128, horizon=1.0, draws=Draws.ANTITHETIC, seed=6)
    stochastic = heston(
        config, spot=100.0, drift=0.02, var0=0.04, kappa=1.0,
        theta=0.04, vol_of_vol=0.0, rho=0.0,
    )
    constant = gbm(config, spot=100.0, drift=0.02, vol=0.2)  # sqrt(0.04) == 0.2

    assert float(np.mean(stochastic.terminal)) == pytest.approx(
        float(np.mean(constant.terminal)), rel=0.01
    )


def test_heston_variance_never_goes_negative_in_the_price():
    """Full truncation floors the variance wherever it is read."""
    config = SimulationConfig(n_paths=20_000, n_steps=252, horizon=1.0, seed=7)
    # Deliberately violates Feller (2*k*theta = 0.02 < xi^2 = 0.36).
    result = heston(
        config, spot=100.0, drift=0.0, var0=0.01, kappa=0.5,
        theta=0.02, vol_of_vol=0.6, rho=-0.7,
    )

    assert result.params["feller_satisfied"] is False
    assert np.all(np.isfinite(result.paths)), "truncation must keep every path finite"
    assert np.all(result.paths > 0)


def test_negative_volatility_is_rejected():
    with pytest.raises(ValueError, match="vol must be non-negative"):
        gbm(SimulationConfig(n_paths=10, n_steps=2), spot=100.0, drift=0.0, vol=-0.2)


def test_correlation_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match=r"rho must lie in \[-1, 1\]"):
        heston(
            SimulationConfig(n_paths=10, n_steps=2), spot=100.0, drift=0.0,
            var0=0.04, kappa=1.0, theta=0.04, vol_of_vol=0.1, rho=1.5,
        )
