"""
mc_greeks.py

Purpose
-------
Estimate Greeks (sensitivities) using Monte Carlo simulation with
common random numbers for variance reduction.

Key Ideas
---------
Greeks measure the sensitivity of option prices to changes in parameters:

- Delta: dV/dS (sensitivity to stock price)
- Gamma: d²V/dS² (convexity, delta sensitivity)
- Vega: dV/dσ (sensitivity to volatility)
- Theta: -dV/dt (time decay)
- Rho: dV/dr (interest rate sensitivity)

Common Random Numbers (CRN)
---------------------------
When estimating Greeks via bump-and-revalue (finite differences), using the
same random draws for S0 and S0±h drastically reduces estimator variance:

    Delta ≈ [V(S0+h) - V(S0-h)] / (2h)

With CRN, V(S0+h) and V(S0-h) are highly correlated, so their difference
is much less noisy than if using independent random numbers.

Without CRN, you're measuring MC noise as much as the actual derivative.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from black_scholes import (
        delta_call,
        delta_put,
        gamma,
        rho_call,
        rho_put,
        theta_call,
        theta_put,
        vega,
    )
    from payoff import call_payoffs, put_payoffs
except ImportError:
    from .black_scholes import (
        delta_call,
        delta_put,
        gamma,
        rho_call,
        rho_put,
        theta_call,
        theta_put,
        vega,
    )
    from .payoff import call_payoffs, put_payoffs


def mc_delta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int,
    option_type: str = "call",
    q: float = 0.0,
    bump_size: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Estimate Delta using finite differences with common random numbers.

    Delta = dV/dS ≈ [V(S0+h) - V(S0-h)] / (2h)

    Using common random numbers ensures that the payoff difference at S0+h
    vs S0-h comes from the price change, not from Monte Carlo noise.

    Parameters
    ----------
    bump_size : float
        Absolute bump amount (e.g., 0.01 means $0.01 bump).

    Returns
    -------
    dict
        Contains 'delta_mc' and 'delta_bs' (analytical benchmark).
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(num_simulations)

    exp_term = (r - q - 0.5 * sigma**2) * T
    sqrt_term = sigma * math.sqrt(T)

    # Generate terminal prices for S0 - bump, S0, S0 + bump
    # All using same random draws (common random numbers)
    S_T_down = (S0 - bump_size) * np.exp(exp_term + sqrt_term * z)
    S_T_up = (S0 + bump_size) * np.exp(exp_term + sqrt_term * z)

    if option_type == "call":
        payoffs_down = call_payoffs(S_T_down, K)
        payoffs_up = call_payoffs(S_T_up, K)
    else:
        payoffs_down = put_payoffs(S_T_down, K)
        payoffs_up = put_payoffs(S_T_up, K)

    discount_factor = math.exp(-r * T)
    prices_down = discount_factor * np.mean(payoffs_down)
    prices_up = discount_factor * np.mean(payoffs_up)

    # Central difference
    delta_mc = (prices_up - prices_down) / (2.0 * bump_size)

    # Analytical benchmark
    if option_type == "call":
        delta_bs = delta_call(S0, K, T, r, sigma, q)
    else:
        delta_bs = delta_put(S0, K, T, r, sigma, q)

    return {
        "delta_mc": delta_mc,
        "delta_bs": delta_bs,
        "error": abs(delta_mc - delta_bs),
    }


def mc_gamma(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int,
    option_type: str = "call",
    q: float = 0.0,
    bump_size: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Estimate Gamma using finite differences with common random numbers.

    Gamma = d²V/dS² ≈ [V(S0+h) - 2*V(S0) + V(S0-h)] / h²
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(num_simulations)

    exp_term = (r - q - 0.5 * sigma**2) * T
    sqrt_term = sigma * math.sqrt(T)

    S_T_down = (S0 - bump_size) * np.exp(exp_term + sqrt_term * z)
    S_T_mid = S0 * np.exp(exp_term + sqrt_term * z)
    S_T_up = (S0 + bump_size) * np.exp(exp_term + sqrt_term * z)

    if option_type == "call":
        payoffs_down = call_payoffs(S_T_down, K)
        payoffs_mid = call_payoffs(S_T_mid, K)
        payoffs_up = call_payoffs(S_T_up, K)
    else:
        payoffs_down = put_payoffs(S_T_down, K)
        payoffs_mid = put_payoffs(S_T_mid, K)
        payoffs_up = put_payoffs(S_T_up, K)

    discount_factor = math.exp(-r * T)
    prices_down = discount_factor * np.mean(payoffs_down)
    prices_mid = discount_factor * np.mean(payoffs_mid)
    prices_up = discount_factor * np.mean(payoffs_up)

    # Second finite difference
    gamma_mc = (
        prices_up - 2.0 * prices_mid + prices_down
    ) / (bump_size**2)

    # Analytical benchmark
    gamma_bs = gamma(S0, K, T, r, sigma, q)

    return {
        "gamma_mc": gamma_mc,
        "gamma_bs": gamma_bs,
        "error": abs(gamma_mc - gamma_bs),
    }


def mc_vega(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int,
    option_type: str = "call",
    q: float = 0.0,
    bump_size: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Estimate Vega using finite differences with common random numbers.

    Vega = dV/dσ ≈ [V(σ+h) - V(σ-h)] / (2h)

    Note: Vega is typically reported per 1% move in volatility.
    """
    rng = np.random.default_rng(seed)

    # Generate random numbers once
    z_base = rng.standard_normal(num_simulations)

    discount_factor = math.exp(-r * T)
    sqrt_T = math.sqrt(T)

    def price_at_vol(vol):
        """Price option at given volatility using same random draws."""
        exp_term = (r - q - 0.5 * vol**2) * T
        sqrt_term = vol * sqrt_T
        S_T = S0 * np.exp(exp_term + sqrt_term * z_base)

        if option_type == "call":
            payoffs = call_payoffs(S_T, K)
        else:
            payoffs = put_payoffs(S_T, K)

        return discount_factor * np.mean(payoffs)

    sigma_down = sigma - bump_size
    sigma_up = sigma + bump_size

    price_down = price_at_vol(sigma_down)
    price_up = price_at_vol(sigma_up)

    # Central difference
    vega_mc = (price_up - price_down) / (2.0 * bump_size)

    # Analytical benchmark (vega per 1% change)
    vega_bs = vega(S0, K, T, r, sigma, q)

    return {
        "vega_mc": vega_mc,
        "vega_bs": vega_bs,
        "error": abs(vega_mc - vega_bs),
    }


def mc_theta(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int,
    option_type: str = "call",
    q: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Estimate Theta (daily time decay) using finite differences.

    Theta = -dV/dt ≈ [V(T - dT) - V(T)] / dT

    We bump time forward by one day (1/365).
    """
    rng = np.random.default_rng(seed)

    # Use same random draws for both times
    z = rng.standard_normal(num_simulations)

    sqrt_T = math.sqrt(T)
    sqrt_T_tomorrow = math.sqrt(T - 1.0 / 365.0) if T > 1.0 / 365.0 else 0.0

    discount_factor = math.exp(-r * T)
    discount_factor_tomorrow = math.exp(-r * (T - 1.0 / 365.0))

    # Terminal price at time T
    exp_term = (r - q - 0.5 * sigma**2) * T
    S_T = S0 * np.exp(exp_term + sigma * sqrt_T * z)

    # Terminal price at time T - 1 day (adjusted path)
    if sqrt_T_tomorrow > 0:
        exp_term_tomorrow = (r - q - 0.5 * sigma**2) * (T - 1.0 / 365.0)
        S_T_tomorrow = S0 * np.exp(
            exp_term_tomorrow + sigma * sqrt_T_tomorrow * z
        )
    else:
        S_T_tomorrow = S_T  # If less than a day left, use same

    if option_type == "call":
        payoffs_today = call_payoffs(S_T, K)
        payoffs_tomorrow = call_payoffs(S_T_tomorrow, K)
    else:
        payoffs_today = put_payoffs(S_T, K)
        payoffs_tomorrow = put_payoffs(S_T_tomorrow, K)

    price_today = discount_factor * np.mean(payoffs_today)
    price_tomorrow = discount_factor_tomorrow * np.mean(payoffs_tomorrow)

    # Theta is negative of time derivative, but we report daily decay
    theta_mc = (price_tomorrow - price_today)  # Negative = decay

    # Analytical benchmark (daily theta)
    if option_type == "call":
        theta_bs = theta_call(S0, K, T, r, sigma, q)
    else:
        theta_bs = theta_put(S0, K, T, r, sigma, q)

    return {
        "theta_mc": theta_mc,
        "theta_bs": theta_bs,
        "error": abs(theta_mc - theta_bs),
    }


def mc_rho(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    num_simulations: int,
    option_type: str = "call",
    q: float = 0.0,
    bump_size: float = 0.01,
    seed: int | None = None,
) -> dict:
    """
    Estimate Rho (interest rate sensitivity) using finite differences.

    Rho = dV/dr ≈ [V(r+h) - V(r-h)] / (2h)

    Note: Rho is typically reported per 1% move in rates.
    """
    rng = np.random.default_rng(seed)

    z = rng.standard_normal(num_simulations)

    def price_at_rate(rate):
        """Price option at given interest rate."""
        exp_term = (rate - q - 0.5 * sigma**2) * T
        sqrt_term = sigma * math.sqrt(T)
        S_T = S0 * np.exp(exp_term + sqrt_term * z)

        if option_type == "call":
            payoffs = call_payoffs(S_T, K)
        else:
            payoffs = put_payoffs(S_T, K)

        discount_factor = math.exp(-rate * T)
        return discount_factor * np.mean(payoffs)

    r_down = r - bump_size
    r_up = r + bump_size

    price_down = price_at_rate(r_down)
    price_up = price_at_rate(r_up)

    rho_mc = (price_up - price_down) / (2.0 * bump_size)

    # Analytical benchmark
    if option_type == "call":
        rho_bs = rho_call(S0, K, T, r, sigma, q)
    else:
        rho_bs = rho_put(S0, K, T, r, sigma, q)

    return {
        "rho_mc": rho_mc,
        "rho_bs": rho_bs,
        "error": abs(rho_mc - rho_bs),
    }


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    print("Monte Carlo Greeks (with Common Random Numbers)")
    print("=" * 70)

    print("\nDelta:")
    delta_result = mc_delta(S0, K, T, r, sigma, 100_000, seed=42)
    print(f"  MC:  {delta_result['delta_mc']:>8.6f}")
    print(f"  BS:  {delta_result['delta_bs']:>8.6f}")
    print(f"  Error: {delta_result['error']:>8.6f}")

    print("\nGamma:")
    gamma_result = mc_gamma(S0, K, T, r, sigma, 100_000, seed=42)
    print(f"  MC:  {gamma_result['gamma_mc']:>8.6f}")
    print(f"  BS:  {gamma_result['gamma_bs']:>8.6f}")
    print(f"  Error: {gamma_result['error']:>8.6f}")

    print("\nVega:")
    vega_result = mc_vega(S0, K, T, r, sigma, 100_000, seed=42)
    print(f"  MC:  {vega_result['vega_mc']:>8.6f}")
    print(f"  BS:  {vega_result['vega_bs']:>8.6f}")
    print(f"  Error: {vega_result['error']:>8.6f}")

    print("\nTheta (daily):")
    theta_result = mc_theta(S0, K, T, r, sigma, 100_000, seed=42)
    print(f"  MC:  {theta_result['theta_mc']:>8.6f}")
    print(f"  BS:  {theta_result['theta_bs']:>8.6f}")
    print(f"  Error: {theta_result['error']:>8.6f}")

    print("\nRho:")
    rho_result = mc_rho(S0, K, T, r, sigma, 100_000, seed=42)
    print(f"  MC:  {rho_result['rho_mc']:>8.6f}")
    print(f"  BS:  {rho_result['rho_bs']:>8.6f}")
    print(f"  Error: {rho_result['error']:>8.6f}")

    print("\n" + "=" * 70)
