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
    from gbm import simulate_gbm_paths, simulate_gbm_terminal
    from payoff import call_payoffs, put_payoffs
except ImportError:
    from .gbm import simulate_gbm_paths, simulate_gbm_terminal
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
    q=0.0,
    seed=None,
):
    """
    Price a European call or put using Monte Carlo simulation.

    Returns a dictionary so that we can inspect not only the price,
    but also the simulated terminal prices, payoffs, standard error,
    and confidence interval.

    Parameters
    ----------
    q : float
        Dividend yield (default 0).
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # For pricing, the GBM drift is r (risk-neutral drift), not the
    # historical expected return mu.
    paths = simulate_gbm_paths(
        S0=S0,
        mu=r - q,
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


def price_european_option_mc_terminal(
    S0,
    K,
    T,
    r,
    sigma,
    num_simulations,
    option_type="call",
    q=0.0,
    seed=None,
):
    """
    Price a European call or put using direct terminal-price sampling.

    This is more efficient than price_european_option_mc() because it only
    samples terminal prices S_T directly, without storing full paths.

    Uses: S_T = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)

    Returns a dictionary with price, standard error, and confidence interval.
    No path data is returned since we don't store paths.

    Parameters
    ----------
    q : float
        Dividend yield (default 0).
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    terminal_prices = simulate_gbm_terminal(
        S0=S0,
        mu=r,
        sigma=sigma,
        T=T,
        num_simulations=num_simulations,
        q=q,
        seed=seed,
    )

    if option_type == "call":
        payoffs = call_payoffs(terminal_prices, K)
    else:
        payoffs = put_payoffs(terminal_prices, K)

    discount_factor = math.exp(-r * T)

    expected_payoff = np.mean(payoffs)
    price = discount_factor * expected_payoff

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
    }


if __name__ == "__main__":
    import time

    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    num_sims = 100_000

    # Full-path method (for comparison).
    print("Full-path Monte Carlo:")
    start = time.time()
    result_full = price_european_option_mc(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        steps=252,
        num_simulations=num_sims,
        option_type="call",
        seed=42,
    )
    time_full = time.time() - start
    print(f"  Price: ${result_full['price']:.4f}")
    print(f"  SE: ${result_full['standard_error']:.6f}")
    print(f"  Time: {time_full:.4f}s")

    # Terminal-sampling method (efficient).
    print("\nTerminal-sampling Monte Carlo:")
    start = time.time()
    result_terminal = price_european_option_mc_terminal(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        num_simulations=num_sims,
        option_type="call",
        seed=42,
    )
    time_terminal = time.time() - start
    print(f"  Price: ${result_terminal['price']:.4f}")
    print(f"  SE: ${result_terminal['standard_error']:.6f}")
    print(f"  Time: {time_terminal:.4f}s")

    print(f"\nSpeedup: {time_full/time_terminal:.2f}x")
