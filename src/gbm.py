"""
gbm.py

Purpose
-------
Generate stock-price paths using Geometric Brownian Motion (GBM).

Discrete GBM:
    S_{t+dt} = S_t * exp(
        (mu - 0.5 * sigma^2) * dt
        + sigma * sqrt(dt) * Z
    )

where:
    S_t   = current stock price
    mu    = expected annual return
    sigma = annual volatility
    dt    = time step in years
    Z     = standard normal random shock, Z ~ N(0, 1)

Important
---------
This file demonstrates GBM using a general drift `mu`.

For OPTION PRICING, we use the risk-neutral drift `r` instead.
See `monte_carlo.py` for the pricing implementation.
"""

import numpy as np


def simulate_gbm_paths(
    S0,
    mu,
    sigma,
    T,
    steps,
    num_simulations,
    seed=None,
):
    """
    Generate multiple GBM price paths.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Annual expected return / drift.
    sigma : float
        Annual volatility.
    T : float
        Time to maturity in years.
    steps : int
        Number of time steps.
    num_simulations : int
        Number of simulated paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : np.ndarray
        Shape: (num_simulations, steps + 1)

        Rows    = simulations
        Columns = time points

        Column 0 is S0.
        Column -1 is the terminal price ST.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if steps <= 0 or num_simulations <= 0:
        raise ValueError("steps and num_simulations must be positive.")

    rng = np.random.default_rng(seed)

    dt = T / steps

    # steps movements produce steps + 1 prices:
    # S0, S1, ..., S_steps.
    paths = np.zeros((num_simulations, steps + 1))

    # Every simulated path starts from the same initial price.
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        # One independent standard-normal shock for each path.
        z = rng.standard_normal(num_simulations)

        # Each new price uses the previous price:
        # S_t is calculated from S_{t-1}.
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
        )

    return paths


if __name__ == "__main__":
    # Learning example.
    S0 = 100
    mu = 0.08
    sigma = 0.20
    T = 1.0
    steps = 252
    num_simulations = 3

    paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        steps=steps,
        num_simulations=num_simulations,
        seed=42,
    )

    print("Shape:", paths.shape)
    print("First 3 paths, first 5 time points:")
    print(paths[:3, :5])
