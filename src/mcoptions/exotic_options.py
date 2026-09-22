"""
exotic_options.py

Purpose
-------
Price path-dependent exotic options using Monte Carlo.
Demonstrates use cases where Monte Carlo is essential (no closed form).

Options
-------
1. Arithmetic Asian Call
   - Payoff: max(A - K, 0) where A = (1/m) * sum(S_ti)
   - Average stock price determined at m observation dates
   - No closed form; Monte Carlo is standard method

2. Geometric Asian Call (for control variate)
   - Payoff: max(G - K, 0) where G = (product(S_ti))^(1/m)
   - Has closed-form approximation; excellent control variate for arithmetic
   - Correlation between arithmetic and geometric is very high (~0.99)

Control Variate Strategy
------------------------
The geometric Asian option is an excellent control variate because:
1. Has closed-form price (or very good approximation)
2. Highly correlated with arithmetic Asian (ρ ≈ 0.99)
3. Reduces variance by 80-90% without extra computational cost
"""

from __future__ import annotations

import math

import numpy as np

try:
    from payoff import call_payoffs
except ImportError:
    from .payoff import call_payoffs


def price_asian_arithmetic_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    observation_dates: int | list,
    num_simulations: int,
    q: float = 0.0,
    seed: int | None = None,
    use_control_variate: bool = False,
) -> dict:
    """
    Price arithmetic Asian option using Monte Carlo.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    observation_dates : int or list
        If int: number of equally-spaced observation dates.
        If list: specific observation times in [0, T].
    num_simulations : int
        Number of paths to simulate.
    q : float
        Dividend yield.
    seed : int or None
        Random seed for reproducibility.
    use_control_variate : bool
        If True, use geometric Asian as control variate (reduces variance).

    Returns
    -------
    dict
        Dictionary with 'price', 'standard_error', 'confidence_interval'.
        If use_control_variate, also includes 'beta' and 'variance_reduction_factor'.
    """
    rng = np.random.default_rng(seed)

    # Determine observation times
    if isinstance(observation_dates, int):
        m = observation_dates
        obs_times = np.linspace(0, T, m + 1)[1:]  # Exclude t=0
    else:
        obs_times = np.array(observation_dates)
        m = len(obs_times)

    # Generate path at observation dates
    dt_between = np.diff(np.concatenate([[0], obs_times]))
    log_returns = (r - q - 0.5 * sigma**2) * dt_between + sigma * np.sqrt(
        dt_between
    ) * rng.standard_normal((num_simulations, m))

    # Cumulative log returns => stock prices
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)

    # Arithmetic average
    arithmetic_avg = np.mean(prices, axis=1)
    payoffs_arith = call_payoffs(arithmetic_avg, K)

    discount_factor = math.exp(-r * T)
    price_arithmetic = discount_factor * np.mean(payoffs_arith)

    if not use_control_variate:
        # Standard MC
        payoff_std = np.std(payoffs_arith, ddof=1)
        standard_error = discount_factor * payoff_std / math.sqrt(num_simulations)

        confidence_low = price_arithmetic - 1.96 * standard_error
        confidence_high = price_arithmetic + 1.96 * standard_error

        return {
            "price": price_arithmetic,
            "standard_error": standard_error,
            "confidence_interval": (confidence_low, confidence_high),
            "expected_payoff": np.mean(payoffs_arith),
        }

    else:
        # Geometric Asian as control variate
        geometric_avg = np.exp(np.mean(np.log(prices), axis=1))
        payoffs_geom = call_payoffs(geometric_avg, K)

        # Theoretical price of geometric Asian (closed form)
        # Using Kemna-Vorst approximation or exact formula
        price_geom_mc = discount_factor * np.mean(payoffs_geom)
        price_geom_analytical = _geometric_asian_price(
            S0, K, T, r, sigma, m, q
        )

        # Optimal control variate coefficient
        payoffs_arith_disc = discount_factor * payoffs_arith
        payoffs_geom_disc = discount_factor * payoffs_geom

        covariance = np.cov(payoffs_arith_disc, payoffs_geom_disc)[0, 1]
        variance_geom = np.var(payoffs_geom_disc, ddof=1)

        beta_star = covariance / variance_geom if variance_geom > 1e-10 else 0.0

        # Adjusted payoffs
        adjusted_payoffs = (
            payoffs_arith_disc
            - beta_star * (payoffs_geom_disc - price_geom_analytical)
        )

        price_adjusted = np.mean(adjusted_payoffs)
        payoff_std = np.std(adjusted_payoffs, ddof=1)
        standard_error = payoff_std / math.sqrt(num_simulations)

        # Variance reduction factor
        variance_standard = np.var(payoffs_arith_disc, ddof=1)
        variance_adjusted = np.var(adjusted_payoffs, ddof=1)
        var_reduction_factor = (
            variance_standard / variance_adjusted
            if variance_adjusted > 1e-10
            else 0.0
        )

        confidence_low = price_adjusted - 1.96 * standard_error
        confidence_high = price_adjusted + 1.96 * standard_error

        return {
            "price": price_adjusted,
            "standard_error": standard_error,
            "confidence_interval": (confidence_low, confidence_high),
            "expected_payoff": np.mean(payoffs_arith),
            "beta": beta_star,
            "variance_reduction_factor": var_reduction_factor,
            "price_geometric_mc": price_geom_mc,
            "price_geometric_analytical": price_geom_analytical,
        }


def _geometric_asian_price(S0: float, K: float, T: float, r: float, sigma: float, num_obs: int, q: float = 0.0) -> float:
    """
    Analytical price of geometric Asian option using exact distribution.

    For equally-spaced observation times, the log-geometric average of
    lognormal prices is exactly normal (under GBM), allowing closed-form
    Black-Scholes-style valuation.

    Parameters
    ----------
    num_obs : int
        Number of observation dates (excluding t=0).

    References
    ----------
    Kemna, A. G., Vorst, A. C. (1990). "A pricing method for options based
    on average asset values." Journal of Banking & Finance, 14(1), 113–129.
    """
    # For m equally-spaced observations, exact mean and variance of log(G):
    # E[log G] = log(S0) + (r - q - 0.5*sigma^2) * T * (m+1)/(2m)
    # Var[log G] = sigma^2 * T * (m+1)*(2m+1) / (6*m^2)

    m = num_obs

    # Effective drift for geometric average
    mu_g = (r - q - 0.5 * sigma**2) * T * (m + 1) / (2.0 * m)

    # Effective variance of log-geometric average
    var_log_g = sigma**2 * T * (m + 1) * (2 * m + 1) / (6.0 * m**2)
    sigma_g = math.sqrt(var_log_g)

    # Black-Scholes valuation with adjusted parameters
    # G_T is lognormal with parameters (log(S0) + mu_g, sigma_g)
    # This is equivalent to vanilla BS with S0, effective rate r_eff, vol sigma_g
    from .black_scholes import black_scholes_call

    # Adjusted risk-neutral parameters for the geometric average
    # We want: E[log G_T] = log S0 + (r_eff - 0.5*sigma_g^2)*T
    # where E[log G_T] = log S0 + (r - q - 0.5*sigma^2)*T*(m+1)/(2m)
    # So: (r_eff - 0.5*sigma_g^2)*T = (r - q - 0.5*sigma^2)*T*(m+1)/(2m)
    r_eff = (r - q - 0.5 * sigma**2) * (m + 1) / (2.0 * m) + 0.5 * sigma_g**2

    return black_scholes_call(S0, K, T, r_eff, sigma_g, q)


if __name__ == "__main__":
    import time

    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    observation_dates = 252  # Daily observations

    print("Arithmetic Asian Call Option")
    print("=" * 70)

    # Standard MC
    print("\nStandard Monte Carlo:")
    start = time.time()
    result_standard = price_asian_arithmetic_mc(
        S0, K, T, r, sigma, observation_dates, 100_000, seed=42
    )
    time_standard = time.time() - start

    print(f"  Price: ${result_standard['price']:.4f}")
    print(f"  SE:    ${result_standard['standard_error']:.6f}")
    print(f"  Time:  {time_standard:.4f}s")

    # With geometric control variate
    print("\nWith Geometric Control Variate:")
    start = time.time()
    result_control = price_asian_arithmetic_mc(
        S0,
        K,
        T,
        r,
        sigma,
        observation_dates,
        100_000,
        use_control_variate=True,
        seed=42,
    )
    time_control = time.time() - start

    print(f"  Price: ${result_control['price']:.4f}")
    print(f"  SE:    ${result_control['standard_error']:.6f}")
    print(f"  Time:  {time_control:.4f}s")
    print(f"  Variance Reduction: {result_control['variance_reduction_factor']:.1f}x")
    print(f"  Beta:  {result_control['beta']:.6f}")

    print("\n" + "=" * 70)
