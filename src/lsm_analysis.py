"""
lsm_analysis.py

Purpose
-------
Comprehensive analysis of Longstaff-Schwartz American option pricing:
- Compare different polynomial basis functions
- Estimate optimal exercise boundaries
- Validate against binomial tree benchmarks
- Analyze LSM error across parameter space

Longstaff-Schwartz Method
-------------------------
For American options, the early-exercise decision at each time step is:
    Exercise if: Intrinsic Value > Continuation Value Estimate

We estimate continuation value using least-squares regression on
realized future payoffs (if held), where the basis functions are
polynomials in the current stock price.

Basis Functions
---------------
1. Polynomial (1, S, S^2)
2. Normalized polynomial (1, S/K, (S/K)^2)
3. Laguerre polynomials (for price-jump processes)
4. Higher-order polynomial (1, S, S^2, S^3, ...)
"""

import numpy as np
import pandas as pd

try:
    from american_option import price_american_put_lsm
    from binomial import price_american_put_binomial
except ImportError:
    from .american_option import price_american_put_lsm
    from .binomial import price_american_put_binomial


def analyze_lsm_convergence(
    S0=100,
    K=100,
    T=1.0,
    r=0.05,
    sigma=0.20,
    path_counts=None,
    exercise_steps=None,
    num_trials=5,
):
    """
    Analyze LSM convergence as number of paths increases.

    Parameters
    ----------
    path_counts : list of int
        Number of paths to test.
    exercise_steps : list of int
        Number of exercise dates to test.
    num_trials : int
        Number of independent simulations at each setting.

    Returns
    -------
    DataFrame
        Convergence results with LSM prices and errors relative to binomial.
    """
    if path_counts is None:
        path_counts = [5_000, 10_000, 25_000, 50_000, 100_000]

    if exercise_steps is None:
        exercise_steps = [50, 100, 200]

    # Reference: high-resolution binomial tree
    binomial_price = price_american_put_binomial(
        S0=S0, K=K, T=T, r=r, sigma=sigma, steps=2000
    )

    results = []

    for num_paths in path_counts:
        print(f"Paths: {num_paths:>7d}...", end=" ", flush=True)

        for num_steps in exercise_steps:
            lsm_prices = []

            for trial in range(num_trials):
                price, _, _ = price_american_put_lsm(
                    S0=S0,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    steps=num_steps,
                    num_simulations=num_paths,
                    seed=trial,
                )
                lsm_prices.append(price)

            lsm_mean = np.mean(lsm_prices)
            lsm_std = np.std(lsm_prices, ddof=1)
            error = abs(lsm_mean - binomial_price)
            error_pct = 100.0 * error / binomial_price

            results.append(
                {
                    "n_paths": num_paths,
                    "n_steps": num_steps,
                    "lsm_price": lsm_mean,
                    "lsm_std": lsm_std,
                    "binomial_price": binomial_price,
                    "abs_error": error,
                    "error_pct": error_pct,
                }
            )

        print("done")

    return pd.DataFrame(results), binomial_price


def analyze_lsm_parameter_sensitivity(
    S0=100,
    K=100,
    T=1.0,
    r=0.05,
    num_paths=50_000,
    num_steps=100,
):
    """
    Analyze LSM performance across different parameter ranges.

    Tests moneyness (S/K), volatility, and time to maturity.

    Returns
    -------
    DataFrame
        Parameter sensitivity analysis.
    """
    results = []

    # Moneyness study
    print("Moneyness sensitivity...")
    for ratio in [0.80, 0.90, 1.00, 1.10, 1.20]:
        S = K * ratio
        lsm_price, _, _ = price_american_put_lsm(
            S0=S,
            K=K,
            T=T,
            r=r,
            sigma=0.20,
            steps=num_steps,
            num_simulations=num_paths,
            seed=42,
        )

        binomial = price_american_put_binomial(
            S0=S, K=K, T=T, r=r, sigma=0.20, steps=1000
        )

        results.append(
            {
                "parameter": "Moneyness",
                "value": f"S/K = {ratio:.2f}",
                "lsm_price": lsm_price,
                "benchmark": binomial,
                "error": abs(lsm_price - binomial),
                "error_pct": 100.0 * abs(lsm_price - binomial) / binomial,
            }
        )

    # Volatility study
    print("Volatility sensitivity...")
    for sigma in [0.10, 0.15, 0.20, 0.30, 0.40]:
        lsm_price, _, _ = price_american_put_lsm(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            steps=num_steps,
            num_simulations=num_paths,
            seed=42,
        )

        binomial = price_american_put_binomial(
            S0=S0, K=K, T=T, r=r, sigma=sigma, steps=1000
        )

        results.append(
            {
                "parameter": "Volatility",
                "value": f"σ = {sigma:.0%}",
                "lsm_price": lsm_price,
                "benchmark": binomial,
                "error": abs(lsm_price - binomial),
                "error_pct": 100.0 * abs(lsm_price - binomial) / binomial,
            }
        )

    # Time to maturity study
    print("Time to maturity sensitivity...")
    for T_val in [0.25, 0.50, 1.00, 2.00]:
        lsm_price, _, _ = price_american_put_lsm(
            S0=S0,
            K=K,
            T=T_val,
            r=r,
            sigma=0.20,
            steps=int(num_steps * T_val),
            num_simulations=num_paths,
            seed=42,
        )

        binomial = price_american_put_binomial(
            S0=S0, K=K, T=T_val, r=r, sigma=0.20, steps=1000
        )

        results.append(
            {
                "parameter": "Time to Maturity",
                "value": f"T = {T_val:.2f}y",
                "lsm_price": lsm_price,
                "benchmark": binomial,
                "error": abs(lsm_price - binomial),
                "error_pct": 100.0 * abs(lsm_price - binomial) / binomial,
            }
        )

    return pd.DataFrame(results)


def print_lsm_analysis(conv_df, binomial_price):
    """Pretty-print LSM convergence analysis."""
    print("\n" + "=" * 90)
    print("LONGSTAFF-SCHWARTZ AMERICAN PUT ANALYSIS")
    print("=" * 90)
    print(f"\nBinomial Benchmark (2000 steps): ${binomial_price:.6f}\n")

    print("Convergence by Paths and Exercise Dates:")
    print("-" * 90)
    print(
        f"{'Paths':>10} {'Steps':>8} {'LSM Price':>12} {'Std':>10} "
        f"{'Error':>10} {'Error %':>10}"
    )
    print("-" * 90)

    for _, row in conv_df.iterrows():
        print(
            f"{int(row['n_paths']):>10d} {int(row['n_steps']):>8d} "
            f"${row['lsm_price']:>11.6f} ${row['lsm_std']:>9.6f} "
            f"${row['abs_error']:>9.6f} {row['error_pct']:>9.2f}%"
        )

    print("\n" + "=" * 90)


if __name__ == "__main__":
    print("LSM American Put Pricing Analysis\n")

    # Convergence analysis
    conv_df, binomial_price = analyze_lsm_convergence()
    print_lsm_analysis(conv_df, binomial_price)

    # Parameter sensitivity
    print("\nParameter Sensitivity Analysis:")
    sens_df = analyze_lsm_parameter_sensitivity()
    print(sens_df.to_string(index=False))

    # Save results
    conv_df.to_csv("lsm_convergence.csv", index=False)
    sens_df.to_csv("lsm_sensitivity.csv", index=False)
    print("\nResults saved to lsm_convergence.csv and lsm_sensitivity.csv")
