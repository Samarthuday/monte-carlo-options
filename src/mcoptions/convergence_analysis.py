"""
convergence_analysis.py

Purpose
-------
Conduct numerical convergence experiments to verify Monte Carlo behavior
and compare different variance reduction techniques.

Key Findings
------------
We empirically verify that:
1. Monte Carlo RMSE decreases as O(N^-1/2)
2. Variance reduction techniques reduce the constant factor
3. Different methods trade off variance, runtime, and implementation complexity

Experiment
----------
For each simulation count N in {1000, 2500, 5000, 10000, 25000, 50000, 100000, ...}:
- Run 100 independent simulations with different seeds
- Calculate bias, standard deviation, RMSE
- Compare against Black-Scholes analytical price
- Plot RMSE on log-log scale to verify O(N^-1/2) rate
"""

import math

import numpy as np
import pandas as pd

try:
    from black_scholes import black_scholes_call
    from monte_carlo import price_european_option_mc_terminal
    from variance_reduction import (
        price_european_option_mc_antithetic,
        price_european_option_mc_control_variate,
    )
except ImportError:
    from .black_scholes import black_scholes_call
    from .monte_carlo import price_european_option_mc_terminal
    from .variance_reduction import (
        price_european_option_mc_antithetic,
        price_european_option_mc_control_variate,
    )


def run_convergence_experiment(
    S0=100,
    K=110,
    T=1.0,
    r=0.05,
    sigma=0.20,
    q=0.0,
    simulation_counts=None,
    num_trials=100,
    option_type="call",
):
    """
    Run independent simulations at various path counts to measure convergence.

    Parameters
    ----------
    S0, K, T, r, sigma, q : float
        Standard option parameters.
    simulation_counts : list of int
        Number of paths to test. Default: [1000, 2500, 5000, 10000, 25000, 50000, 100000]
    num_trials : int
        Number of independent runs at each path count.
    option_type : str
        "call" or "put".

    Returns
    -------
    DataFrame
        Columns: N, method, price_mean, price_std, bias, rmse, runtime_mean
    """
    if simulation_counts is None:
        simulation_counts = [1000, 2500, 5000, 10000, 25000, 50000, 100_000]

    # Analytical benchmark
    if option_type == "call":
        analytical_price = black_scholes_call(S0, K, T, r, sigma, q)
    else:
        from black_scholes import black_scholes_put
        analytical_price = black_scholes_put(S0, K, T, r, sigma, q)

    results = []

    methods = [
        ("Standard MC", price_european_option_mc_terminal),
        ("Antithetic", price_european_option_mc_antithetic),
        ("Control Variate", price_european_option_mc_control_variate),
    ]

    for n_paths in simulation_counts:
        print(f"Testing N = {n_paths:>7d}...", end=" ", flush=True)

        for method_name, pricing_func in methods:
            prices = []
            runtimes = []

            for trial in range(num_trials):
                import time

                start = time.time()
                result = pricing_func(
                    S0=S0,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    num_simulations=n_paths,
                    option_type=option_type,
                    q=q,
                    seed=trial,  # Different seed for each trial
                )
                elapsed = time.time() - start

                prices.append(result["price"])
                runtimes.append(elapsed)

            prices = np.array(prices)

            price_mean = np.mean(prices)
            price_std = np.std(prices, ddof=1)
            bias = price_mean - analytical_price
            rmse = np.sqrt(np.mean((prices - analytical_price) ** 2))
            runtime_mean = np.mean(runtimes)

            results.append(
                {
                    "N": n_paths,
                    "method": method_name,
                    "price_mean": price_mean,
                    "price_std": price_std,
                    "bias": bias,
                    "rmse": rmse,
                    "runtime_mean": runtime_mean,
                }
            )

        print("done")

    df = pd.DataFrame(results)

    # Calculate theoretical N^-1/2 reference line
    df["theoretical_rmse_ratio"] = 1.0 / np.sqrt(df["N"])

    return df, analytical_price


def print_convergence_results(df, analytical_price):
    """Pretty-print convergence analysis results."""
    print("\n" + "=" * 100)
    print("MONTE CARLO CONVERGENCE ANALYSIS")
    print("=" * 100)
    print(f"\nAnalytical Black-Scholes Price: ${analytical_price:.6f}\n")

    for method in df["method"].unique():
        method_df = df[df["method"] == method].reset_index(drop=True)

        print(f"\n{method}")
        print("-" * 100)
        print(
            f"{'N':>10} {'Price':>12} {'Std':>12} {'Bias':>12} "
            f"{'RMSE':>12} {'Time (ms)':>12} {'Variance Reduction':>15}"
        )
        print("-" * 100)

        for _, row in method_df.iterrows():
            n = int(row["N"])
            price = row["price_mean"]
            std = row["price_std"]
            bias = row["bias"]
            rmse = row["rmse"]
            runtime_ms = row["runtime_mean"] * 1000

            # Variance reduction factor relative to theoretical N^-1/2
            theoretical_rmse = (
                analytical_price * 0.05 / math.sqrt(n)
            )  # rough scaling
            var_reduction_factor = theoretical_rmse / rmse if rmse > 0 else 0

            print(
                f"{n:>10d} ${price:>11.6f} ${std:>11.6f} ${bias:>11.6f} "
                f"${rmse:>11.6f} {runtime_ms:>11.3f} {var_reduction_factor:>14.2f}x"
            )

        # Verify O(N^-1/2) scaling
        if len(method_df) > 1:
            n1, rmse1 = method_df.iloc[0]["N"], method_df.iloc[0]["rmse"]
            n2, rmse2 = method_df.iloc[-1]["N"], method_df.iloc[-1]["rmse"]

            theoretical_ratio = math.sqrt(n2 / n1)
            actual_ratio = rmse1 / rmse2

            print("\nEmpirical O(N^-1/2) Verification (first vs last):")
            print(f"  Theoretical ratio: {theoretical_ratio:.2f}x")
            print(f"  Actual RMSE ratio: {actual_ratio:.2f}x")

    print("\n" + "=" * 100)


def plot_convergence_rmse(df, analytical_price):
    """Plot RMSE vs N on log-log axes with O(N^-1/2) reference line."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    plt.figure(figsize=(10, 6))

    for method in df["method"].unique():
        method_df = df[df["method"] == method].sort_values("N")
        plt.loglog(method_df["N"], method_df["rmse"], marker="o", label=method, linewidth=2)

    # Reference line: O(N^-1/2)
    n_vals = np.array([df["N"].min(), df["N"].max()])
    # Scale reference to match data approximately
    scale = df["rmse"].iloc[0] * (df["N"].iloc[0] ** 0.5)
    reference_rmse = scale / np.sqrt(n_vals)
    plt.loglog(n_vals, reference_rmse, "k--", label="O(N^-1/2) reference", linewidth=1.5)

    plt.xlabel("Number of Paths (N)")
    plt.ylabel("Root Mean Square Error (RMSE)")
    plt.title("Monte Carlo Convergence: RMSE vs. Number of Paths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    import pathlib
    output_dir = pathlib.Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "mc_convergence.png", dpi=150)
    plt.close()
    print(f"Saved: results/figures/mc_convergence.png")


def plot_variance_reduction(df):
    """Plot variance-reduction factor (σ_Standard / σ_Method) vs N.

    Shows how much variance reduction each method achieves relative to standard MC.
    A factor of 2.0 means the method has half the standard deviation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    # Pivot so each N has columns for each method's price_std
    pivot_df = df.pivot(index="N", columns="method", values="price_std")

    plt.figure(figsize=(10, 6))

    # Compute reduction factors for Antithetic and Control Variate
    if "Standard MC" in pivot_df.columns:
        standard_std = pivot_df["Standard MC"]

        if "Antithetic" in pivot_df.columns:
            antithetic_factor = standard_std / pivot_df["Antithetic"]
            plt.semilogx(antithetic_factor.index, antithetic_factor.values,
                        marker="o", label="Antithetic", linewidth=2)

        if "Control Variate" in pivot_df.columns:
            control_factor = standard_std / pivot_df["Control Variate"]
            plt.semilogx(control_factor.index, control_factor.values,
                        marker="s", label="Control Variate", linewidth=2)

    # Baseline: no reduction
    n_min, n_max = df["N"].min(), df["N"].max()
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1.5, label="No reduction (baseline)")

    plt.xlabel("Number of Paths (N)")
    plt.ylabel("Reduction Factor (σ_Standard / σ_Method)")
    plt.title("Variance Reduction Factor by Method")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    import pathlib
    output_dir = pathlib.Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "variance_reduction.png", dpi=150)
    plt.close()
    print(f"Saved: results/figures/variance_reduction.png")


if __name__ == "__main__":
    import pathlib

    print("Running convergence analysis...")

    # Create results directory
    results_dir = pathlib.Path("results/data")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run standard convergence experiment
    df, analytical = run_convergence_experiment(
        S0=100,
        K=110,
        T=1.0,
        r=0.05,
        sigma=0.20,
        simulation_counts=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000],
        num_trials=100,
        option_type="call",
    )

    print_convergence_results(df, analytical)

    # Save results to CSV
    df.to_csv(results_dir / "convergence_results.csv", index=False)
    print(f"\nResults saved to results/data/convergence_results.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence_rmse(df, analytical)
    plot_variance_reduction(df)
