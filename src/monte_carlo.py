"""
monte_carlo.py

Purpose
-------
Price European options using Monte Carlo simulation.

The pricing model is built in several steps:

1. Simulate stock-price paths under the RISK-NEUTRAL measure.
2. Extract terminal prices ST.
3. Calculate option payoffs.
4. Average the simulated payoffs.
5. Discount the expected payoff back to today.

Risk-neutral GBM:
    S_{t+dt} = S_t * exp(
        (r - 0.5*sigma^2)dt
        + sigma*sqrt(dt)*Z
    )

European call:
    payoff = max(ST - K, 0)

European put:
    payoff = max(K - ST, 0)

Monte Carlo price:
    V0 ≈ exp(-rT) * (1/N) * sum(payoff_i)

Standard error:
    SE = sample_std(payoffs) / sqrt(N)

Approximate 95% confidence interval:
    price ± 1.96 * SE_discounted
"""

import math

import numpy as np

try:
    from gbm import simulate_gbm_paths
    from payoff import call_payoffs, put_payoffs
except ImportError:
    from .gbm import simulate_gbm_paths
    from .payoff import call_payoffs, put_payoffs


def price_european_option_mc(
    S0,
    K,
    T,
    r,
    sigma,
    steps,
    num_simulations,
    option_type="call",
    seed=None,
):
    """
    Price a European call or put using Monte Carlo simulation.

    Returns a dictionary so that we can inspect not only the price,
    but also the simulated terminal prices, payoffs, standard error,
    and confidence interval.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # IMPORTANT:
    # For pricing, the GBM drift is r, not the historical expected
    # return mu. This is risk-neutral pricing.
    paths = simulate_gbm_paths(
        S0=S0,
        mu=r,
        sigma=sigma,
        T=T,
        steps=steps,
        num_simulations=num_simulations,
        seed=seed,
    )

    terminal_prices = paths[:, -1]

    # Reuse the single payoff definition in payoff.py instead of
    # recomputing max(S - K, 0) / max(K - S, 0) here.
    if option_type == "call":
        payoffs = call_payoffs(terminal_prices, K)
    else:
        payoffs = put_payoffs(terminal_prices, K)

    discount_factor = math.exp(-r * T)

    expected_payoff = np.mean(payoffs)

    # Discount the expected future payoff to today.
    price = discount_factor * expected_payoff

    # Monte Carlo sampling error.
    payoff_std = np.std(payoffs, ddof=1)
    standard_error = discount_factor * payoff_std / math.sqrt(
        num_simulations
    )

    confidence_low = price - 1.96 * standard_error
    confidence_high = price + 1.96 * standard_error

    return {
        "price": price,
        "expected_payoff": expected_payoff,
        "standard_error": standard_error,
        "confidence_interval": (
            confidence_low,
            confidence_high,
        ),
        "terminal_prices": terminal_prices,
        "payoffs": payoffs,
        "paths": paths,
    }


if __name__ == "__main__":
    # European call example.
    result = price_european_option_mc(
        S0=100,
        K=110,
        T=1.0,
        r=0.05,
        sigma=0.20,
        steps=252,
        num_simulations=100_000,
        option_type="call",
        seed=42,
    )

    print("Monte Carlo call price:", result["price"])
    print("Expected payoff:", result["expected_payoff"])
    print("Standard error:", result["standard_error"])
    print("95% CI:", result["confidence_interval"])
