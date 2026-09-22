"""
american_option.py

Purpose
-------
Price an American put using Longstaff-Schwartz Least-Squares
Monte Carlo (LSM).

Why an American put?
--------------------
An American option can be exercised before expiration.

For a non-dividend-paying stock, an American call does not have
an early-exercise advantage over the corresponding European call.
An American put is therefore a useful example for demonstrating
the early-exercise problem.

Core idea
---------
At each exercise date:

    1. Identify paths that are in the money.
    2. Calculate their immediate exercise value.
    3. Estimate continuation value using regression.
    4. Exercise when immediate value > estimated continuation value.
    5. Work backwards through time.

Regression basis families:
    - polynomial: powers of S [1, S, S^2, ...]
    - normalized: powers of S/K [1, S/K, (S/K)^2, ...]
    - laguerre: weighted Laguerre polynomials
"""

import math

import numpy as np
from numpy.polynomial.laguerre import lagval

try:
    from gbm import simulate_gbm_paths
    from payoff import put_payoffs
except ImportError:
    from .gbm import simulate_gbm_paths
    from .payoff import put_payoffs


def _build_regression_basis(x: np.ndarray, K: float, basis: str, degree: int) -> np.ndarray:
    """
    Build regression basis matrix for LSM continuation value estimation.

    Parameters
    ----------
    x : np.ndarray
        Stock prices at current time step (candidate paths).
    K : float
        Strike price.
    basis : str
        Type of basis: "polynomial", "normalized", or "laguerre".
    degree : int
        Degree of polynomial basis.

    Returns
    -------
    np.ndarray
        Design matrix of shape (len(x), degree + 1).
    """
    if basis == "polynomial":
        return np.column_stack([x**i for i in range(degree + 1)])
    elif basis == "normalized":
        normalized = x / K
        return np.column_stack([normalized**i for i in range(degree + 1)])
    elif basis == "laguerre":
        scaled = x / K
        weight = np.exp(-scaled / 2.0)
        laguerre_terms = [lagval(scaled, [1 if i == j else 0 for j in range(degree + 1)]) for i in range(degree + 1)]
        return np.column_stack([weight * laguerre_terms[i] for i in range(degree + 1)])
    else:
        raise ValueError(f"Unknown basis: {basis}. Choose 'polynomial', 'normalized', or 'laguerre'.")


def price_american_put_lsm(
    S0,
    K,
    T,
    r,
    sigma,
    steps=50,
    num_simulations=50_000,
    basis="polynomial",
    basis_degree=2,
    seed=None,
):
    """
    Price an American put using Longstaff-Schwartz Monte Carlo.

    Parameters
    ----------
    basis : str
        Type of basis functions for regression:
        - "polynomial": 1, S, S^2, ..., S^degree
        - "normalized": 1, S/K, (S/K)^2, ..., (S/K)^degree
        - "laguerre": Laguerre polynomials (advanced)
    basis_degree : int
        Degree of polynomial (2 for "polynomial" is standard).

    Returns
    -------
    price : float
        Estimated American put price.
    paths : np.ndarray
        Simulated risk-neutral stock-price paths.
    cashflows : np.ndarray
        Final discounted cashflows at time 0.
    """
    if steps < 2:
        raise ValueError("At least 2 time steps are required.")

    # Simulate under the risk-neutral measure.
    paths = simulate_gbm_paths(
        S0=S0,
        mu=r,
        sigma=sigma,
        T=T,
        steps=steps,
        num_simulations=num_simulations,
        seed=seed,
    )

    dt = T / steps
    discount_one_step = math.exp(-r * dt)

    # At maturity, the only possible payoff is the immediate payoff.
    # `cashflows[i]` always holds path i's best-known future cash flow,
    # expressed in the dollars of whatever time step the backward loop
    # has currently discounted to.
    cashflows = put_payoffs(paths[:, -1], K)

    # Track exercise boundary: S*(t) where exercise becomes optimal
    exercise_boundary = {}

    # Work backwards from the last exercise date to the first.
    for t in range(steps - 1, 0, -1):

        # Move all future cashflows one time step back.
        cashflows *= discount_one_step

        stock_prices = paths[:, t]

        immediate_value = put_payoffs(stock_prices, K)

        # Candidates for exercise are simply the paths that are in the
        # money at this date. There is no notion of a path being
        # "already exercised" and therefore excluded here: backward
        # induction must be free to revisit and override a later
        # (in-time) exercise decision with a better earlier one. The
        # `cashflows` array is simply overwritten whenever exercising
        # now beats the estimated continuation value, so no separate
        # "alive" bookkeeping is needed or correct.
        candidates = immediate_value > 0

        if np.count_nonzero(candidates) < 3:
            exercise_boundary[t * dt] = np.nan
            continue

        x = stock_prices[candidates]
        y = cashflows[candidates]

        # Build regression basis (polynomial, normalized, or Laguerre)
        X = _build_regression_basis(x, K, basis, basis_degree)

        # Least-squares regression:
        # continuation value ≈ X @ coefficients
        coefficients, *_ = np.linalg.lstsq(X, y, rcond=None)

        continuation_value = X @ coefficients

        # Exercise when immediate value is greater than
        # estimated continuation value.
        exercise = immediate_value[candidates] > continuation_value

        candidate_indices = np.where(candidates)[0]
        exercise_indices = candidate_indices[exercise]

        # Record exercise boundary: max stock price where exercise occurs
        if np.any(exercise):
            exercise_boundary[t * dt] = float(np.max(x[exercise]))
        else:
            exercise_boundary[t * dt] = np.nan

        # Replace the discounted future cashflow with the
        # immediate exercise payoff. This overrides whatever decision
        # (exercise or continuation) was recorded at a later date.
        cashflows[exercise_indices] = (
            immediate_value[exercise_indices]
        )

    # The loop above stops at t=1, so `cashflows` is currently
    # expressed in time-1 dollars. Discount once more to bring
    # everything back to today's (time-0) dollars.
    cashflows *= discount_one_step
    price = np.mean(cashflows)

    return price, paths, cashflows, exercise_boundary


if __name__ == "__main__":
    price, _, _, boundary = price_american_put_lsm(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.20,
        steps=50,
        num_simulations=50_000,
        seed=42,
    )

    print("American put price:", price)
    print(f"Exercise boundary times: {sorted(boundary.keys())}")
