"""
black_scholes.py

Purpose
-------
Analytical Black-Scholes prices for European call and put options.

We use this as a validation benchmark for the Monte Carlo model.

Black-Scholes assumptions:
    - European option
    - Dividend yield q
    - Constant risk-free rate r
    - Constant volatility sigma
    - Lognormal stock-price dynamics

Formulas with dividend yield:
d1 = [ln(S0/K) + (r - q + 0.5*sigma^2)T] / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)

Call:
    C = S0*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)

Put:
    P = K*e^(-rT)*N(-d2) - S0*e^(-qT)*N(-d1)

Greeks (rate of change of option price):
    Delta = ∂V/∂S
    Gamma = ∂²V/∂S²
    Vega = ∂V/∂σ
    Theta = -∂V/∂t (time decay)
    Rho = ∂V/∂r
"""

import math


def standard_normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def standard_normal_pdf(x):
    """Standard normal probability density function."""
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def d1_d2(S0, K, T, r, q, sigma):
    """Calculate the Black-Scholes d1 and d2 terms.

    Parameters
    ----------
    q : float
        Dividend yield (default 0 for non-dividend-paying stocks).
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    d1 = (
        math.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    return d1, d2


def black_scholes_call(S0, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes European call price."""
    d1, d2 = d1_d2(S0, K, T, r, q, sigma)

    return (
        S0 * math.exp(-q * T) * standard_normal_cdf(d1)
        - K * math.exp(-r * T) * standard_normal_cdf(d2)
    )


def black_scholes_put(S0, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes European put price."""
    d1, d2 = d1_d2(S0, K, T, r, q, sigma)

    return (
        K * math.exp(-r * T) * standard_normal_cdf(-d2)
        - S0 * math.exp(-q * T) * standard_normal_cdf(-d1)
    )


def delta_call(S0, K, T, r, sigma, q=0.0):
    """Delta of a European call: rate of change with respect to S0."""
    d1, _ = d1_d2(S0, K, T, r, q, sigma)
    return math.exp(-q * T) * standard_normal_cdf(d1)


def delta_put(S0, K, T, r, sigma, q=0.0):
    """Delta of a European put."""
    d1, _ = d1_d2(S0, K, T, r, q, sigma)
    return -math.exp(-q * T) * standard_normal_cdf(-d1)


def gamma(S0, K, T, r, sigma, q=0.0):
    """Gamma: second derivative with respect to S0.

    Same for calls and puts.
    """
    d1, _ = d1_d2(S0, K, T, r, q, sigma)
    return (
        math.exp(-q * T) * standard_normal_pdf(d1)
        / (S0 * sigma * math.sqrt(T))
    )


def vega(S0, K, T, r, sigma, q=0.0):
    """Vega: rate of change with respect to volatility sigma.

    Same for calls and puts. Returns change per 1% change in vol (not per 1.0).
    """
    d1, _ = d1_d2(S0, K, T, r, q, sigma)
    return S0 * math.exp(-q * T) * standard_normal_pdf(d1) * math.sqrt(T) / 100.0


def theta_call(S0, K, T, r, sigma, q=0.0):
    """Theta of a European call: time decay (negative of time derivative).

    Returns daily theta (divided by 365).
    """
    if T <= 0:
        return 0.0

    d1, d2 = d1_d2(S0, K, T, r, q, sigma)

    term1 = -S0 * math.exp(-q * T) * standard_normal_pdf(d1) * sigma / (
        2.0 * math.sqrt(T)
    )
    term2 = -r * K * math.exp(-r * T) * standard_normal_cdf(d2)
    term3 = q * S0 * math.exp(-q * T) * standard_normal_cdf(d1)

    return (term1 + term2 + term3) / 365.0


def theta_put(S0, K, T, r, sigma, q=0.0):
    """Theta of a European put."""
    if T <= 0:
        return 0.0

    d1, d2 = d1_d2(S0, K, T, r, q, sigma)

    term1 = -S0 * math.exp(-q * T) * standard_normal_pdf(d1) * sigma / (
        2.0 * math.sqrt(T)
    )
    term2 = r * K * math.exp(-r * T) * standard_normal_cdf(-d2)
    term3 = -q * S0 * math.exp(-q * T) * standard_normal_cdf(-d1)

    return (term1 + term2 + term3) / 365.0


def rho_call(S0, K, T, r, sigma, q=0.0):
    """Rho of a European call: rate of change with respect to interest rate r.

    Returns change per 1% change in rate (not per 1.0).
    """
    _, d2 = d1_d2(S0, K, T, r, q, sigma)
    return K * T * math.exp(-r * T) * standard_normal_cdf(d2) / 100.0


def rho_put(S0, K, T, r, sigma, q=0.0):
    """Rho of a European put."""
    _, d2 = d1_d2(S0, K, T, r, q, sigma)
    return -K * T * math.exp(-r * T) * standard_normal_cdf(-d2) / 100.0


if __name__ == "__main__":
    S0 = 100
    K = 110
    T = 1.0
    r = 0.05
    sigma = 0.20
    q = 0.02

    call_price = black_scholes_call(S0, K, T, r, sigma, q)
    put_price = black_scholes_put(S0, K, T, r, sigma, q)

    print(f"European Call:  ${call_price:.4f}")
    print(f"European Put:   ${put_price:.4f}")

    print("\nCall Greeks:")
    print(f"  Delta: {delta_call(S0, K, T, r, sigma, q):.6f}")
    print(f"  Gamma: {gamma(S0, K, T, r, sigma, q):.6f}")
    print(f"  Vega:  {vega(S0, K, T, r, sigma, q):.6f}")
    print(f"  Theta: {theta_call(S0, K, T, r, sigma, q):.6f} (daily)")
    print(f"  Rho:   {rho_call(S0, K, T, r, sigma, q):.6f}")

    print("\nPut Greeks:")
    print(f"  Delta: {delta_put(S0, K, T, r, sigma, q):.6f}")
    print(f"  Gamma: {gamma(S0, K, T, r, sigma, q):.6f}")
    print(f"  Vega:  {vega(S0, K, T, r, sigma, q):.6f}")
    print(f"  Theta: {theta_put(S0, K, T, r, sigma, q):.6f} (daily)")
    print(f"  Rho:   {rho_put(S0, K, T, r, sigma, q):.6f}")
