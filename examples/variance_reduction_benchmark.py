"""How much does each variance reduction technique actually buy?

Prices one European call four ways at an identical path budget and compares
each result to the closed-form Black-Scholes value. The interesting column is
the last one: how many times fewer paths you would need for the same precision.

Run:  python examples/variance_reduction_benchmark.py
"""

from __future__ import annotations

from montecarlo import Draws, SimulationConfig, black_scholes, gbm, price_european

SPOT, STRIKE, RATE, VOL, HORIZON = 100.0, 105.0, 0.03, 0.25, 1.0
N_PATHS = 100_000


def run(scheme: Draws, control_variate: bool = False):
    config = SimulationConfig(
        n_paths=N_PATHS, n_steps=1, horizon=HORIZON, draws=scheme, seed=2024
    )
    paths = gbm(config, spot=SPOT, drift=RATE, vol=VOL)
    return price_european(paths, STRIKE, RATE, "call", control_variate=control_variate)


def main() -> None:
    analytic = black_scholes(SPOT, STRIKE, RATE, VOL, HORIZON, "call")
    print(f"Black-Scholes closed form: {analytic:.6f}")
    print(f"Paths per run: {N_PATHS:,}\n")

    rows = [
        ("pseudo-random", run(Draws.PSEUDO)),
        ("antithetic", run(Draws.ANTITHETIC)),
        ("pseudo + control variate", run(Draws.PSEUDO, control_variate=True)),
        ("scrambled Sobol", run(Draws.SOBOL)),
    ]

    baseline = rows[0][1].std_error

    header = f"{'scheme':<26} {'price':>10} {'std error':>12} {'abs error':>11} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for label, est in rows:
        abs_error = abs(est.value - analytic)
        if est.std_error is None:
            print(f"{label:<26} {est.value:>10.6f} {'n/a':>12} {abs_error:>11.6f} {'n/a':>9}")
        else:
            # Equal-precision speedup: variance scales as 1/n, so halving the
            # standard error is worth four times the paths.
            speedup = (baseline / est.std_error) ** 2
            print(
                f"{label:<26} {est.value:>10.6f} {est.std_error:>12.6f} "
                f"{abs_error:>11.6f} {speedup:>8.1f}x"
            )

    print("\nSobol reports no standard error on purpose: a scrambled QMC run")
    print("gives no valid single-run error estimate, and a number that looks")
    print("like an error bar without being one is worse than none.")


if __name__ == "__main__":
    main()
