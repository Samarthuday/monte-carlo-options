"""
black_scholes.py

Purpose
-------
Analytical Black-Scholes prices for European call and put options.

We use this as a validation benchmark for the Monte Carlo model.

Black-Scholes assumptions here:
    - European option
    - No dividends (q = 0)
    - Constant risk-free rate
    - Constant volatility
    - Lognormal stock-price dynamics

Formulas
--------
d1 = [ln(S0/K) + (r + 0.5*sigma^2)T] / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

Call:
    C = S0*N(d1) - K*exp(-rT)*N(d2)

Put:
    P = K*exp(-rT)*N(-d2) - S0*N(-d1)
"""

import math


def standard_normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(S0, K, T, r, sigma):
    """Calculate the Black-Scholes d1 and d2 terms."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (
        math.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    return d1, d2


def black_scholes_call(S0, K, T, r, sigma):
    """Return the Black-Scholes European call price."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)

    return (
        S0 * standard_normal_cdf(d1)
        - K * math.exp(-r * T) * standard_normal_cdf(d2)
    )


def black_scholes_put(S0, K, T, r, sigma):
    """Return the Black-Scholes European put price."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)

    return (
        K * math.exp(-r * T) * standard_normal_cdf(-d2)
        - S0 * standard_normal_cdf(-d1)
    )


if __name__ == "__main__":
    S0 = 100
    K = 110
    T = 1.0
    r = 0.05
    sigma = 0.20

    print("Black-Scholes call:",
          black_scholes_call(S0, K, T, r, sigma))

    print("Black-Scholes put:",
          black_scholes_put(S0, K, T, r, sigma))
