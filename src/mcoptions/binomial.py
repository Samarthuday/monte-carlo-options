"""
binomial.py

Purpose
-------
Price an American put using the Cox-Ross-Rubinstein (CRR) binomial
tree.

Why this exists
----------------
Longstaff-Schwartz Monte Carlo (see `american_option.py`) is a
regression-based approximation: its accuracy depends on the choice
of basis functions, the number of paths, and the number of exercise
dates. A binomial tree, in contrast, converges to the true American
option price as the number of steps grows, with no regression
involved.

That makes it a useful *independent* benchmark for testing the LSM
implementation: `american_put.mean` alone can never catch a biased
exercise policy, because a wildly overpriced American put still
satisfies `American >= European`. Comparing against a converged
binomial price catches that.

CRR parameters:
    u  = exp(sigma * sqrt(dt))
    d  = 1 / u
    p  = (exp(r * dt) - d) / (u - d)
"""

import numpy as np


def price_american_put_binomial(S0, K, T, r, sigma, steps=2000):
    """
    Price an American put using a CRR binomial tree.

    Parameters mirror `price_american_put_lsm` in american_option.py
    so the two can be compared directly. `steps` here plays the same
    role as tree depth; a few thousand steps is enough to converge
    to within a cent or two for typical parameters.
    """
    if steps < 1:
        raise ValueError("At least 1 step is required.")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices, one per node at the last time step.
    j = np.arange(steps + 1)
    stock_prices = S0 * u ** (steps - j) * d ** j
    option_values = np.maximum(K - stock_prices, 0.0)

    # Work backwards through the tree. At each node, the option is
    # worth the greater of continuing (discounted expectation under
    # the risk-neutral probabilities) or exercising immediately.
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        stock_prices = S0 * u ** (i - j) * d ** j

        continuation_value = discount * (
            p * option_values[:-1] + (1 - p) * option_values[1:]
        )
        exercise_value = np.maximum(K - stock_prices, 0.0)

        option_values = np.maximum(continuation_value, exercise_value)

    return option_values[0]


if __name__ == "__main__":
    price = price_american_put_binomial(
        S0=100, K=100, T=1.0, r=0.05, sigma=0.20, steps=2000,
    )
    print("Binomial American put price:", price)
