"""
payoff.py

Purpose
-------
Calculate option payoffs at expiration.

European call payoff:
    C_T = max(S_T - K, 0)

European put payoff:
    P_T = max(K - S_T, 0)

This file does NOT calculate the option price today.
It only converts terminal stock prices into terminal option payoffs.

Every other module in this project (monte_carlo.py, american_option.py)
imports its payoff logic from here instead of recomputing
max(S - K, 0) / max(K - S, 0) inline. That keeps the payoff definition
in exactly one place.
"""

import numpy as np


def call_payoff(S_T, K):
    """Return the payoff of a single European call at expiration."""
    return max(S_T - K, 0.0)


def put_payoff(S_T, K):
    """Return the payoff of a single European put at expiration."""
    return max(K - S_T, 0.0)


def call_payoffs(S_T, K):
    """
    Vectorized call payoffs for an array (or list) of terminal prices.

    Returns a numpy array so this can be used directly inside the
    Monte Carlo simulation loop without a separate implementation.
    """
    S_T = np.asarray(S_T, dtype=float)
    return np.maximum(S_T - K, 0.0)


def put_payoffs(S_T, K):
    """Vectorized put payoffs for an array (or list) of terminal prices."""
    S_T = np.asarray(S_T, dtype=float)
    return np.maximum(K - S_T, 0.0)


if __name__ == "__main__":
    # Small learning example.
    terminal_prices = [90, 100, 110, 120, 140]
    K = 110

    payoffs = call_payoffs(terminal_prices, K)

    print("Terminal prices:", terminal_prices)
    print("Strike:", K)
    print("Call payoffs:", payoffs)
