# Monte Carlo Options Pricing

A quantitative finance project implementing Monte Carlo methods for option pricing — built incrementally, from historical returns and volatility through to simulated option pricing, with each concept understood mathematically before being implemented in code.

```text
Historical Prices → Log Returns → Volatility → Geometric Brownian Motion →
Simulated Price Paths → Option Payoffs → Monte Carlo Option Price
```

---

## Progress

- [x] Historical price data
- [x] Simple & log returns (and additivity of log returns)
- [x] Mean return, sample variance, Bessel's correction
- [x] Standard deviation, volatility, annualized volatility
- [x] Standard normal random shocks
- [x] Geometric Brownian Motion (GBM)
- [x] Single and multiple simulated price paths (vectorized with NumPy)
- [ ] Option payoff (calls & puts)
- [ ] Terminal stock price → Monte Carlo option pricing
- [ ] Discounting
- [ ] Convergence analysis with increasing simulations
- [ ] Validation against analytical pricing models
- [ ] American option pricing / early exercise

---

## Project Structure

```text
monte-carlo-options/
├── learning/
│   └── notes.md        # Detailed derivations, formulas, and observations
├── src/
│   ├── returns.py       # Log returns, mean, variance, std dev, annualized volatility
│   └── gbm.py            # Simulated price paths via Geometric Brownian Motion
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Mathematical Foundation

**Log return**

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

**Sample variance** (Bessel-corrected)

$$
s^2 = \frac{\sum (r_i - \bar r)^2}{n - 1}
$$

**Annualized volatility**

$$
\sigma_{\text{annual}} = \sigma_{\text{daily}}\sqrt{252}
$$

**Standard normal random shock**

$$
Z \sim N(0, 1)
$$

**Geometric Brownian Motion** (discrete form used in simulation)

$$
S_{t+dt} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \sqrt{dt}\, Z\right]
$$

| Symbol | Meaning |
|---|---|
| $S_t$ | Current stock price |
| $\mu$ | Expected annual return |
| $\sigma$ | Annual volatility |
| $dt$ | Time step |
| $Z$ | Standard normal random shock |

---

## Current Simulation

```python
S0 = 100          # initial stock price ($)
mu = 0.08         # expected annual return (8%)
sigma = 0.20      # annual volatility (20%)
dt = 1 / 252      # time step
steps = 252       # trading days simulated (1 year)
num_simulations = 3
```

The number of simulated paths will be increased substantially for the final pricing model.

---

## Technologies

- Python
- NumPy
- Monte Carlo simulation & probability theory
- Quantitative finance

---

## Long-Term Objective

Evolve this into a complete Monte Carlo options-pricing framework that can:

1. Estimate historical volatility
2. Simulate future stock-price paths
3. Calculate option payoffs
4. Price options via Monte Carlo simulation
5. Study convergence as simulation count increases
6. Compare Monte Carlo prices against analytical models
7. Extend to American option pricing

The focus throughout is understanding the quantitative reasoning behind the model — not treating it as a black box.