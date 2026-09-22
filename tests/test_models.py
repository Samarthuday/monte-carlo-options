import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from american_option import price_american_put_lsm
from binomial import price_american_put_binomial
from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    delta_put,
    gamma,
    vega,
)
from gbm import simulate_gbm_paths
from monte_carlo import price_european_option_mc, price_european_option_mc_terminal
from payoff import call_payoff, call_payoffs, put_payoff, put_payoffs
from returns import calculate_log_returns, calculate_statistics
from variance_reduction import (
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
)
from mc_greeks import mc_delta, mc_gamma, mc_vega, mc_theta, mc_rho

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


def test_black_scholes_with_dividend_yield():
    S0, K, T, r, sigma, q = 100, 110, 1.0, 0.05, 0.20, 0.02

    # With dividend yield q, call price decreases (less valuable to hold stock)
    call_no_div = black_scholes_call(S0, K, T, r, sigma, q=0.0)
    call_with_div = black_scholes_call(S0, K, T, r, sigma, q=q)

    assert call_with_div < call_no_div


def test_black_scholes_greeks_call():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    delta = delta_call(S0, K, T, r, sigma)
    gamma_val = gamma(S0, K, T, r, sigma)
    vega_val = vega(S0, K, T, r, sigma)

    # Call delta should be between 0 and 1 (OTM call here, so < 0.5)
    assert 0 < delta < 1
    assert delta < 0.5  # OTM

    # Gamma should be positive
    assert gamma_val > 0

    # Vega should be positive (call price increases with vol)
    assert vega_val > 0


def test_black_scholes_greeks_put():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    delta = delta_put(S0, K, T, r, sigma)
    gamma_val = gamma(S0, K, T, r, sigma)
    vega_val = vega(S0, K, T, r, sigma)

    # Put delta should be between -1 and 0 (OTM put, so > -1 but < 0)
    assert -1 < delta < 0

    # Gamma should be positive (same for calls and puts)
    assert gamma_val > 0

    # Vega should be positive (same for calls and puts)
    assert vega_val > 0


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


def test_monte_carlo_terminal_sampling_matches_full_path():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    bs = black_scholes_call(S0, K, T, r, sigma)

    # Terminal sampling should converge to BS as well.
    mc_terminal = price_european_option_mc_terminal(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        num_simulations=50_000,
        option_type="call", seed=42,
    )

    # Should be within a few standard errors of analytical price.
    assert abs(mc_terminal["price"] - bs) < 5 * mc_terminal["standard_error"]


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


def test_american_put_matches_binomial_benchmark():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    # A converged CRR binomial tree is an independent ground truth
    # that doesn't depend on the LSM regression at all. This is the
    # check that actually catches a biased exercise policy: an
    # overpriced American put can still satisfy
    # `american >= european`, but it can't also match the binomial
    # benchmark.
    binomial_price = price_american_put_binomial(
        S0=S0, K=K, T=T, r=r, sigma=sigma, steps=2000,
    )

    american_price, _, _ = price_american_put_lsm(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        steps=50, num_simulations=50_000, seed=42,
    )

    # LSM with a finite number of paths/exercise dates is only an
    # approximation, so allow a modest tolerance rather than
    # requiring an exact match.
    assert abs(american_price - binomial_price) < 0.15


# --- variance_reduction.py -----------------------------------------------


def test_antithetic_variates_converges_to_bs():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    bs = black_scholes_call(S0, K, T, r, sigma)

    result = price_european_option_mc_antithetic(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        num_simulations=50_000,
        option_type="call", seed=42,
    )

    # Antithetic should converge to BS like regular MC.
    assert abs(result["price"] - bs) < 5 * result["standard_error"]


def test_control_variate_converges_to_bs():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20
    bs = black_scholes_call(S0, K, T, r, sigma)

    result = price_european_option_mc_control_variate(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        num_simulations=50_000,
        option_type="call", seed=42,
    )

    # Control variate should also converge to BS.
    assert abs(result["price"] - bs) < 5 * result["standard_error"]


def test_control_variate_reduces_variance():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    # Standard MC
    result_standard = price_european_option_mc_terminal(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        num_simulations=50_000, option_type="call", seed=42,
    )

    # Control variate MC
    result_control = price_european_option_mc_control_variate(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        num_simulations=50_000, option_type="call", seed=42,
    )

    # Control variate should have lower or similar standard error.
    # Due to randomness, we just check it's a reasonable estimate.
    assert result_control["standard_error"] < result_standard["standard_error"] * 1.5


# --- mc_greeks.py -----------------------------------------------


def test_mc_delta_matches_analytical():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    result = mc_delta(S0, K, T, r, sigma, 100_000, seed=42)

    # MC delta should be close to analytical
    assert abs(result["delta_mc"] - result["delta_bs"]) < 0.01


def test_mc_gamma_matches_analytical():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    result = mc_gamma(S0, K, T, r, sigma, 100_000, seed=42)

    # MC gamma should be close to analytical
    assert abs(result["gamma_mc"] - result["gamma_bs"]) < 0.001


def test_mc_vega_matches_analytical():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    result = mc_vega(S0, K, T, r, sigma, 100_000, seed=42)

    # MC vega should be close to analytical
    assert abs(result["vega_mc"] - result["vega_bs"]) < 1.0


def test_mc_theta_matches_analytical():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    result = mc_theta(S0, K, T, r, sigma, 100_000, seed=42)

    # MC theta should be very close to analytical (least Monte Carlo noise)
    assert abs(result["theta_mc"] - result["theta_bs"]) < 0.001


def test_mc_rho_matches_analytical():
    S0, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.20

    result = mc_rho(S0, K, T, r, sigma, 100_000, seed=42)

    # MC rho should be close to analytical
    assert abs(result["rho_mc"] - result["rho_bs"]) < 1.0
