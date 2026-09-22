"""
implied_volatility.py

Purpose
-------
Solve for implied volatility from observed option prices.

Key Concept
-----------
While Black-Scholes gives price given parameters (forward problem):
    C = BS(S, K, T, r, q, σ)

Risk managers often need the inverse (backward problem):
    σ_implied = BS_inverse(S, K, T, r, q, C_market)

Implied volatility is the volatility that makes the theoretical price
equal to the market price. It's the market's estimate of realized volatility
over the option's lifetime, and is central to trading/risk management.

Numerical Methods
-----------------
The inverse is computed using root-finding algorithms:
1. Brent's method (robust, no derivative needed)
2. Newton-Raphson with vega (faster, uses derivative)

Both use vega (dC/dσ) for stability and efficiency.
"""

try:
    from black_scholes import black_scholes_call, black_scholes_put, vega
except ImportError:
    from .black_scholes import black_scholes_call, black_scholes_put, vega


def implied_volatility(
    S0,
    K,
    T,
    r,
    market_price,
    option_type="call",
    q=0.0,
    initial_guess=0.30,
    tol=1e-6,
    max_iter=100,
):
    """
    Compute implied volatility using Newton-Raphson with Vega.

    Parameters
    ----------
    S0, K, T, r, q : float
        Standard option parameters.
    market_price : float
        Observed option price in the market.
    option_type : str
        "call" or "put".
    initial_guess : float
        Starting volatility estimate (default 30%).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        Contains 'implied_vol', 'iterations', 'converged', and 'final_error'.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    sigma = initial_guess

    for iteration in range(max_iter):
        # Theoretical price at current sigma
        if option_type == "call":
            theo_price = black_scholes_call(S0, K, T, r, sigma, q)
        else:
            theo_price = black_scholes_put(S0, K, T, r, sigma, q)

        price_error = theo_price - market_price

        # Check convergence
        if abs(price_error) < tol:
            return {
                "implied_vol": sigma,
                "iterations": iteration + 1,
                "converged": True,
                "final_error": abs(price_error),
            }

        # Vega: derivative of price with respect to sigma
        vega_val = vega(S0, K, T, r, sigma, q)

        if abs(vega_val) < 1e-10:
            return {
                "implied_vol": sigma,
                "iterations": iteration + 1,
                "converged": False,
                "final_error": abs(price_error),
            }

        # Newton-Raphson update
        sigma = sigma - price_error / vega_val

        # Keep sigma positive and reasonable
        sigma = max(0.001, min(3.0, sigma))

    return {
        "implied_vol": sigma,
        "iterations": max_iter,
        "converged": False,
        "final_error": abs(price_error),
    }


def implied_volatility_brent(
    S0,
    K,
    T,
    r,
    market_price,
    option_type="call",
    q=0.0,
    vol_bounds=(0.001, 3.0),
    tol=1e-6,
):
    """
    Compute implied volatility using Brent's method.

    More robust than Newton-Raphson but slightly slower.
    Doesn't require vega calculation.

    Parameters
    ----------
    vol_bounds : tuple
        (min_vol, max_vol) for the root-finding bracket.

    Returns
    -------
    dict
        Contains 'implied_vol', 'iterations', 'converged'.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    def objective(sigma):
        """Price difference at given sigma."""
        if option_type == "call":
            theo_price = black_scholes_call(S0, K, T, r, sigma, q)
        else:
            theo_price = black_scholes_put(S0, K, T, r, sigma, q)
        return theo_price - market_price

    # Check that bounds bracket the solution
    f_low = objective(vol_bounds[0])
    f_high = objective(vol_bounds[1])

    if f_low * f_high > 0:
        # Solution not bracketed; try to find reasonable bounds
        if f_low > 0:
            vol_bounds = (vol_bounds[0] * 0.1, vol_bounds[0])
        else:
            vol_bounds = (vol_bounds[1], vol_bounds[1] * 10.0)

    # Brent's method implementation
    a, b = vol_bounds
    fa = objective(a)
    fb = objective(b)

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    iterations = 0
    max_iter = 100

    while abs(b - a) > tol and iterations < max_iter:
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Bounds checking
        if not (
            (3 * a + b) / 4 < s < b if mflag else (a < s < (3 * a + b) / 4)
        ):
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = objective(s)

        if abs(fs) < tol:
            return {
                "implied_vol": s,
                "iterations": iterations + 1,
                "converged": True,
            }

        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        iterations += 1

    return {
        "implied_vol": (a + b) / 2.0,
        "iterations": iterations,
        "converged": abs(objective((a + b) / 2.0)) < tol,
    }


if __name__ == "__main__":
    S0, K, T, r = 100, 110, 1.0, 0.05

    # Theoretical price at 20% volatility
    true_sigma = 0.20
    from black_scholes import black_scholes_call

    market_price = black_scholes_call(S0, K, T, r, true_sigma)

    print("Implied Volatility Recovery Test")
    print("=" * 70)
    print(f"\nTrue Volatility:    {true_sigma:.2%}")
    print(f"Theoretical Price:  ${market_price:.4f}")

    # Newton-Raphson
    print("\nNewton-Raphson Method:")
    result_nr = implied_volatility(S0, K, T, r, market_price)
    print(f"  Recovered σ:  {result_nr['implied_vol']:.2%}")
    print(f"  Iterations:   {result_nr['iterations']}")
    print(f"  Converged:    {result_nr['converged']}")
    print(f"  Error:        ${result_nr['final_error']:.2e}")

    # Brent's method
    print("\nBrent's Method:")
    result_brent = implied_volatility_brent(S0, K, T, r, market_price)
    print(f"  Recovered σ:  {result_brent['implied_vol']:.2%}")
    print(f"  Iterations:   {result_brent['iterations']}")
    print(f"  Converged:    {result_brent['converged']}")

    # Test with perturbed price
    print("\n" + "=" * 70)
    print("Recovery from Noisy Market Price (+0.05):")
    noisy_price = market_price + 0.05
    result_noisy = implied_volatility(S0, K, T, r, noisy_price)
    print(f"  Noisy Price:    ${noisy_price:.4f}")
    print(f"  Recovered σ:    {result_noisy['implied_vol']:.2%}")
    print(f"  True σ:         {true_sigma:.2%}")
    print(f"  Difference:     {abs(result_noisy['implied_vol'] - true_sigma):.2%}")
