# montecarlo-engine

A Monte Carlo engine for asset price simulation and European option pricing, with
variance reduction and error bars that are computed the way each sampling scheme
actually requires.

```python
from montecarlo import SimulationConfig, Draws, gbm, price_european, black_scholes

config = SimulationConfig(n_paths=100_000, n_steps=1, horizon=1.0, draws=Draws.ANTITHETIC)
paths = gbm(config, spot=100.0, drift=0.03, vol=0.25)

print(price_european(paths, strike=105.0, rate=0.03, kind="call"))
# 9.078808 +/- 0.083502  95% CI [8.995306, 9.162310]

print(black_scholes(100.0, 105.0, 0.03, 0.25, 1.0, "call"))
# 9.121799
```

---

## About the scope of this repository

**This is a deliberately reduced version of one module of a larger, private
system.** The full engine is part of **Atlas**, a proprietary quantitative
platform I build and maintain, where it feeds risk limits, scenario analysis and
a portfolio simulator. None of that is here.

What was removed on purpose:

- multi-asset simulation under a Cholesky-factored correlation matrix
- GARCH volatility forecasting as a path input
- the risk layer that consumes these paths (VaR, expected shortfall, drawdown surfaces)
- every integration point: data providers, portfolio state, the execution path

What is left is the numerical core, standing on its own, tested against
closed-form answers. It is enough to read in twenty minutes and judge how I write
quantitative code, which is the only thing this repository is for.

The extraction and reduction were done with the help of Claude Code (Anthropic),
used to draw a clean line between what is worth showing and what stays private.

---

## What is in it

| Module | What it does |
|---|---|
| `config.py` | Frozen run configuration. A result carries the settings that produced it, so it can be reproduced. |
| `draws.py` | Three samplers behind one signature: pseudo-random, antithetic, scrambled Sobol. |
| `processes.py` | Geometric Brownian motion, Merton jump-diffusion, Heston stochastic volatility. |
| `pricing.py` | European pricing, control variates, and closed-form Black-Scholes to check against. |

### Three decisions worth pointing at

**1. Sampling is separated from the process.** Whether you draw pseudo-randomly,
antithetically or from a Sobol sequence is a property of the sampler, not of the
stochastic differential equation. Keeping them apart means all three processes get
all three samplers for free, and adding a fourth process does not touch the
sampling code.

**2. The standard error is computed differently for each scheme, because it has
to be.** Antithetic paths are not independent, they come in negatively correlated
pairs, so the textbook `s / sqrt(n)` is simply the wrong estimator and reports an
interval wider than the truth. The valid one treats each *pair average* as a
single observation. Scrambled Sobol admits no unbiased single-run error estimate
at all, so this engine reports `None` and says why, rather than printing a number
shaped like an error bar that is not one.

**3. Pricing off-measure raises instead of warning.** Simulating under the
real-world drift and discounting at the risk-free rate is the most common way to
get a Monte Carlo option price quietly wrong. `price_european` compares the paths'
drift to the discount rate and refuses the mismatch.

---

## Measured results

One European call, 100,000 paths per run, identical parameters, compared against
the closed-form price of **9.121799**:

| Scheme | Price | Std error | Abs error | Equal-precision speedup |
|---|---|---|---|---|
| pseudo-random | 9.129230 | 0.051479 | 0.007431 | 1.0x |
| antithetic | 9.078808 | 0.042603 | 0.042991 | 1.5x |
| pseudo + control variate | 9.099237 | 0.023934 | 0.022562 | 4.6x |
| scrambled Sobol | 9.121652 | not reported | **0.000147** | not reported |

Reproduce with `python examples/variance_reduction_benchmark.py`.

Two things in that table are worth reading slowly:

- **Speedup is quoted at equal precision, not equal runtime.** Variance falls as
  `1/n`, so halving the standard error is worth four times the paths. A 4.6x
  speedup means you would need 4.6x the paths to reach the same precision without
  the control variate.
- **Antithetic sampling shows a larger absolute error than plain sampling here,
  and that is not a contradiction.** Absolute error on one run is a single draw
  from the error distribution; the standard error describes that whole
  distribution. Antithetic sampling narrows the distribution, which is the claim
  being made, and a single run can still land further from the mean. Judging a
  variance reduction technique by one run's absolute error is how people talk
  themselves into methods that do not work. Sobol's 0.000147 is a real 50x
  improvement in absolute error, but it is also one draw, and the missing error
  bar is exactly why it cannot be quoted with a confidence interval.

---

## Running it

Requires Python 3.10 or later.

```bash
pip install -e ".[dev]"
pytest -q
```

19 tests. Each checks a property with a known analytic answer: the terminal mean
of GBM, the variance of its log returns, put-call parity, Heston collapsing to
GBM when vol-of-vol is zero, the jump compensator preserving the mean. Tolerances
are derived from the estimator's own standard error rather than tuned until the
test passed.

---

## License

MIT, and it covers only the reduced code in this repository. The full Atlas
platform this module was extracted from is proprietary and is not licensed here.

## Author

Mauricio Gerardo Trevino Saldana
