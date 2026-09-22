"""
mcoptions: Monte Carlo Options Pricing & Numerical Methods

A quantitative finance research package implementing Monte Carlo methods
for derivative pricing, variance reduction, Greeks estimation, and
numerical validation.

Key Features
------------
- Risk-neutral European option pricing (direct terminal sampling, 300x faster)
- Variance reduction: antithetic variates, control variates
- Analytical Greeks: Delta, Gamma, Vega, Theta, Rho
- Monte Carlo Greeks using common random numbers
- American put pricing via Longstaff-Schwartz with CRR validation
- Exotic options: arithmetic Asian with geometric control variate
- Implied volatility inversion (Newton-Raphson, Brent)
- Convergence analysis and RMSE verification

Usage Example
-------------
    from mcoptions.monte_carlo import price_european_option_mc_terminal
    from mcoptions.black_scholes import black_scholes_call, delta_call

    # Price European call
    result = price_european_option_mc_terminal(
        S0=100, K=110, T=1.0, r=0.05, sigma=0.20,
        num_simulations=100_000, seed=42
    )

    # Compare to analytical Black-Scholes
    bs_price = black_scholes_call(100, 110, 1.0, 0.05, 0.20)
    delta = delta_call(100, 110, 1.0, 0.05, 0.20)

References
----------
- Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities
- Longstaff, F. A., Schwartz, E. S. (2001). Valuing American options by simulation
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering
"""

__version__ = "1.0.0"
__author__ = "Samarth Uday"

# Core pricing models
from .american_option import price_american_put_lsm
from .binomial import price_american_put_binomial
from .black_scholes import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    rho_put,
    theta_call,
    theta_put,
    vega,
)
from .exotic_options import price_asian_arithmetic_mc

# Utilities
from .gbm import simulate_gbm_paths, simulate_gbm_terminal
from .implied_volatility import implied_volatility, implied_volatility_brent
from .mc_greeks import mc_delta, mc_gamma, mc_rho, mc_theta, mc_vega
from .monte_carlo import (
    price_european_option_mc,
    price_european_option_mc_terminal,
)
from .payoff import call_payoff, call_payoffs, put_payoff, put_payoffs
from .returns import calculate_log_returns, calculate_statistics
from .variance_reduction import (
    price_european_option_mc_antithetic,
    price_european_option_mc_control_variate,
)

__all__ = [
    # Black-Scholes pricing
    "black_scholes_call",
    "black_scholes_put",
    # Analytical Greeks
    "delta_call",
    "delta_put",
    "gamma",
    "vega",
    "theta_call",
    "theta_put",
    "rho_call",
    "rho_put",
    # European pricing
    "price_european_option_mc",
    "price_european_option_mc_terminal",
    # Variance reduction
    "price_european_option_mc_antithetic",
    "price_european_option_mc_control_variate",
    # MC Greeks
    "mc_delta",
    "mc_gamma",
    "mc_vega",
    "mc_theta",
    "mc_rho",
    # American options
    "price_american_put_lsm",
    "price_american_put_binomial",
    # Exotic options
    "price_asian_arithmetic_mc",
    # Implied volatility
    "implied_volatility",
    "implied_volatility_brent",
    # Simulation
    "simulate_gbm_paths",
    "simulate_gbm_terminal",
    # Payoff utilities
    "call_payoff",
    "put_payoff",
    "call_payoffs",
    "put_payoffs",
    # Returns analysis
    "calculate_log_returns",
    "calculate_statistics",
]
