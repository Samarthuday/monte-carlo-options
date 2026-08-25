# Monte Carlo Options Pricing

A quantitative finance project implementing Monte Carlo methods for option pricing, built incrementally from historical returns and volatility through Geometric Brownian Motion, European option pricing, validation, convergence analysis, and American option pricing using Longstaff-Schwartz Monte Carlo.

```text
Historical Prices
        ↓
Log Returns
        ↓
Historical Volatility
        ↓
Geometric Brownian Motion
        ↓
Simulated Price Paths
        ↓
Terminal Prices
        ↓
Option Payoffs
        ↓
Risk-Neutral Pricing
        ↓
Monte Carlo European Option Price
        ↓
Convergence / Confidence Intervals
        ↓
Black-Scholes Validation
        ↓
American Option Pricing
        ↓
Longstaff-Schwartz Early Exercise
```

---

## Project Structure

```text
monte-carlo-options/
├── learning/
│   └── notes.md
├── notebooks/
│   └── monte_carlo_options_analysis.ipynb
├── src/
│   ├── returns.py
│   ├── gbm.py
│   ├── payoff.py
│   ├── monte_carlo.py
│   ├── black_scholes.py
│   └── american_option.py
├── tests/
│   └── test_models.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## What Each File Does

| File | Purpose |
|---|---|
| `src/returns.py` | Calculates log returns, mean return, variance, standard deviation, and annualized volatility |
| `src/gbm.py` | Generates stock-price paths using GBM |
| `src/payoff.py` | Calculates call and put payoffs at expiration |
| `src/monte_carlo.py` | Prices European options using risk-neutral Monte Carlo |
| `src/black_scholes.py` | Provides analytical European option prices for validation |
| `src/american_option.py` | Prices an American put using Longstaff-Schwartz Monte Carlo |
| `notebooks/monte_carlo_options_analysis.ipynb` | Visual analysis, plots, convergence, and model comparison |
| `learning/notes.md` | Detailed mathematical learning notes |

---

## Core Formulas

### Log Return

\[
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)
\]

### Sample Variance

\[
s^2=\frac{\sum(r_i-\bar r)^2}{n-1}
\]

### Annualized Volatility

\[
\sigma_{\text{annual}}
=
\sigma_{\text{daily}}\sqrt{252}
\]

### Standard Normal Shock

\[
Z\sim N(0,1)
\]

### GBM

For general simulation:

\[
S_{t+dt}
=
S_t
\exp\left[
\left(\mu-\frac{1}{2}\sigma^2\right)dt
+
\sigma\sqrt{dt}Z
\right]
\]

For option pricing, the drift becomes the risk-free rate \(r\):

\[
S_{t+dt}
=
S_t
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)dt
+
\sigma\sqrt{dt}Z
\right]
\]

### European Call Payoff

\[
C_T=\max(S_T-K,0)
\]

### European Put Payoff

\[
P_T=\max(K-S_T,0)
\]

### Monte Carlo European Option Price

\[
V_0
\approx
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\text{Payoff}^{(i)}
\]

### Monte Carlo Standard Error

\[
SE
=
e^{-rT}
\frac{s_{\text{payoff}}}{\sqrt{N}}
\]

### 95% Confidence Interval

\[
V_0\pm1.96SE
\]

---

## European vs American Options

A European option can only be exercised at expiration.

An American option can be exercised at any time up to expiration.

For a European option, we only need the terminal stock price:

\[
S_T
\]

For an American option, we need to consider the possibility of exercise at multiple dates.

The project uses a European option first because it provides the foundation for the more difficult American-option problem.

The American-option implementation uses Longstaff-Schwartz Least-Squares Monte Carlo.

---

## Validation

The European Monte Carlo model is compared against the Black-Scholes analytical price.

The goal is not for the two prices to be exactly identical in every run.

Monte Carlo contains sampling error, so the estimate should approach the analytical value as the number of simulations increases.

---

## Visual Analysis

The notebook contains:

- Simulated GBM price paths
- Terminal stock-price distribution
- Call payoff distribution
- Call payoff as a function of terminal stock price
- Monte Carlo convergence
- Monte Carlo confidence intervals
- Monte Carlo vs Black-Scholes comparison
- Volatility sensitivity
- American put comparison

---

## Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the individual modules:

```bash
python3 src/returns.py
python3 src/gbm.py
python3 src/payoff.py
python3 src/monte_carlo.py
python3 src/black_scholes.py
python3 src/american_option.py
```

Run tests:

```bash
pytest
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/monte_carlo_options_analysis.ipynb
```

---

## Important Modeling Note

The historical expected return estimated in `returns.py` is useful for understanding historical data and for real-world GBM simulation.

However, when pricing options under the standard risk-neutral Monte Carlo framework, the stock-price drift is replaced by the risk-free rate \(r\).

Therefore:

```text
Historical analysis:
    estimate μ from returns

Option pricing:
    use r as the risk-neutral drift
```

This distinction is fundamental to the project.

---

## Final Objective

The final project demonstrates the full chain:

```text
Historical Data
    ↓
Statistical Estimation
    ↓
Volatility
    ↓
Stochastic Price Simulation
    ↓
Monte Carlo Payoffs
    ↓
Risk-Neutral Valuation
    ↓
European Option Pricing
    ↓
Analytical Validation
    ↓
Convergence Analysis
    ↓
American Option Pricing
    ↓
Early Exercise via Longstaff-Schwartz
```

The emphasis is on understanding the mathematics and implementation rather than treating the pricing model as a black box.
