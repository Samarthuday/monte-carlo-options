import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from american_option import price_american_put_lsm
from black_scholes import black_scholes_call, black_scholes_put
from gbm import simulate_gbm_paths
from monte_carlo import price_european_option_mc
from payoff import call_payoff, call_payoffs, put_payoff, put_payoffs
from returns import calculate_log_returns, calculate_statistics

# --- payoff.py -------------------------------------------------------

def test_call_payoff():
    assert call_payoff(140, 110) == 30
    assert call_payoff(90, 110) == 0


def test_put_payoff():
    assert put_payoff(90, 110) == 20
    assert put_payoff(140, 110) == 0


def test_vectorized_payoffs_match_scalar():
    terminal_prices = [90, 100, 110, 120, 140]
    K = 110

    expected_calls = [call_payoff(s, K) for s in terminal_prices]
    expected_puts = [put_payoff(s, K) for s in terminal_prices]

    assert np.allclose(call_payoffs(terminal_prices, K), expected_calls)
    assert np.allclose(put_payoffs(terminal_prices, K), expected_puts)


# --- returns.py --------------------------------------------------------

def test_log_returns_and_statistics():
    prices = [100, 105, 103, 108, 106]
    returns = calculate_log_returns(prices)
    stats = calculate_statistics(returns)

    assert len(returns) == len(prices) - 1
    assert stats["variance"] > 0
    assert stats["annualized_volatility"] > stats["standard_deviation"]


# --- gbm.py --------------------------------------------------------------

def test_gbm_shape():
    paths = simulate_gbm_paths(
        S0=100,
        mu=0.08,
        sigma=0.20,
        T=1.0,
        steps=252,
        num_simulations=100,
        seed=42,
    )

    assert paths.shape == (100, 253)
    assert np.all(paths[:, 0] == 100)


# --- black_scholes.py ----------------------------------------------------

def test_black_scholes_call_positive():
    price = black_scholes_call(
        S0=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.20,
    )

    assert price > 0


def test_black_scholes_put_call_parity():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    call = black_scholes_call(S0, K, T, r, sigma)
    put = black_scholes_put(S0, K, T, r, sigma)

    # Put-call parity: C - P = S0 - K * exp(-rT)
    lhs = call - put
    rhs = S0 - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1e-8)


# --- monte_carlo.py --------------------------------------------------------

def test_monte_carlo_call_converges_to_black_scholes():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    mc = price_european_option_mc(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        steps=100, num_simulations=50_000,
        option_type="call", seed=42,
    )
    bs = black_scholes_call(S0, K, T, r, sigma)

    # Within a few standard errors of the analytical price.
    assert abs(mc["price"] - bs) < 5 * mc["standard_error"]


# --- american_option.py -----------------------------------------------

def test_american_put_at_least_european_put():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    american_price, _, _ = price_american_put_lsm(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        steps=50, num_simulations=20_000, seed=42,
    )
    european_price = black_scholes_put(S0, K, T, r, sigma)

    # The American put must be worth at least as much as the
    # European put, since early exercise is an optional extra right.
    assert american_price >= european_price - 0.05
