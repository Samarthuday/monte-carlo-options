"""
returns.py

Purpose
-------
Calculate historical log returns and estimate volatility from a series
of historical stock prices.

This file is intentionally written in a learning-friendly way. The
calculations are shown explicitly rather than hidden inside a library
call so the mathematics is easy to follow.

Main formulas
-------------
Log return:
    r_t = ln(P_t / P_{t-1})

Sample variance:
    s^2 = sum((r_i - r_bar)^2) / (n - 1)

Standard deviation:
    s = sqrt(s^2)

Annualized volatility:
    sigma_annual = sigma_daily * sqrt(252)
"""

import math


def calculate_log_returns(prices):
    """Return the log return between each pair of consecutive prices."""
    returns = []

    for i in range(1, len(prices)):
        ret = math.log(prices[i] / prices[i - 1])
        returns.append(ret)

    return returns


def calculate_statistics(returns, trading_days=252):
    """
    Calculate mean return, sample variance, standard deviation,
    and annualized volatility.
    """
    if len(returns) < 2:
        raise ValueError("At least two returns are required.")

    mean_return = sum(returns) / len(returns)

    squared_deviations = []

    for ret in returns:
        deviation = ret - mean_return
        squared_deviation = deviation ** 2
        squared_deviations.append(squared_deviation)

    # n - 1: Bessel's correction for sample variance.
    variance = sum(squared_deviations) / (len(squared_deviations) - 1)

    standard_deviation = math.sqrt(variance)

    # Volatility scales with the square root of time.
    annualized_volatility = standard_deviation * math.sqrt(trading_days)

    return {
        "mean_return": mean_return,
        "variance": variance,
        "standard_deviation": standard_deviation,
        "annualized_volatility": annualized_volatility,
    }


if __name__ == "__main__":
    # Small educational example.
    # In a real project, these would eventually come from market data.
    prices = [100, 105, 103, 108, 106]

    returns = calculate_log_returns(prices)
    stats = calculate_statistics(returns)

    print("Prices:", prices)
    print("Log returns:", returns)
    print("Mean return:", stats["mean_return"])
    print("Variance:", stats["variance"])
    print("Standard deviation:", stats["standard_deviation"])
    print("Annualized volatility:", stats["annualized_volatility"])
