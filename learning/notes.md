# Monte Carlo Options — Learning Notes

## Project Structure

```text
monte-carlo-options/
├── learning/
│   └── notes.md
├── src/
│   ├── returns.py
│   └── gbm.py
├── README.md
├── requirements.txt
└── .gitignore
```

| File | Purpose |
|---|---|
| `src/returns.py` | Log returns, mean, variance, standard deviation, annualized volatility |
| `src/gbm.py` | Simulated stock-price paths via Geometric Brownian Motion (GBM) |
| `learning/notes.md` | Concepts, formulas, and observations from the project |

---

## 1. Project Goal

Build a Monte Carlo–based options pricing model.

```text
Historical Prices → Log Returns → Volatility → GBM → Simulated Price Paths → Option Payoffs → Monte Carlo Option Price
```

**Status:** completed through simulated price paths.

---

## 2. Returns

### 2.1 Simple Return

$$
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

Simple returns are **not symmetric**: a 10% gain (100 → 110) is not offset by a 10% loss — going from 110 back to 100 is actually a −9.09% return.

### 2.2 Log Return

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

```python
ret = math.log(prices[i] / prices[i - 1])
```

**Why log returns:** they are *additive* over time, since $\ln(a) + \ln(b) = \ln(ab)$:

$$
r_1 + r_2 = \ln\left(\frac{P_1}{P_0}\right) + \ln\left(\frac{P_2}{P_1}\right) = \ln\left(\frac{P_2}{P_0}\right)
$$

$$
\boxed{\text{Sum of log returns} = \text{Total log return}}
$$

Verified numerically in `src/returns.py` for `prices = [100, 105, 103, 108, 106]` (sum and total agree to floating-point precision, ≈ 0.05827).

---

## 3. Statistics of Returns

### 3.1 Mean Return

$$
\bar{r} = \frac{\sum r_i}{n}
$$

```python
mean_return = sum(returns) / len(returns)
```

### 3.2 Sample Variance & Bessel's Correction

$$
s^2 = \frac{\sum (r_i - \bar{r})^2}{n - 1}
$$

```python
squared_deviations = [(r - mean_return) ** 2 for r in returns]
variance = sum(squared_deviations) / (len(squared_deviations) - 1)
```

We divide by $n-1$ rather than $n$ because this is a **sample** variance estimated from observed data — this adjustment is **Bessel's correction**.

### 3.3 Standard Deviation & Volatility

$$
s = \sqrt{s^2}
$$

```python
standard_deviation = math.sqrt(variance)
```

Standard deviation of returns is used directly as **volatility**: more spread in returns → higher standard deviation → higher volatility.

### 3.4 Annualized Volatility

Assuming 252 trading days/year, and since variance scales linearly with time (so standard deviation scales with $\sqrt{t}$):

$$
\sigma_{\text{annual}} = \sigma_{\text{daily}} \sqrt{252}
$$

```python
annualized_volatility = standard_deviation * math.sqrt(252)
```

**Worked example:** daily $s \approx 0.0387 \Rightarrow \sigma_{\text{annual}} \approx 0.6147$ (≈ 61.47%).

---

## 4. Standard Normal Distribution ($Z$)

$$
Z \sim N(0, 1)
$$

```python
z = np.random.normal(0, 1)
```

- Mean 0, standard deviation 1 — but individual draws are **not** bounded to $[-1, 1]$ (values like −3.56 or 3.73 are valid).
- Also called a *standard normal random shock*.
- In GBM, $Z$ is the source of randomness that makes each simulated path different.

---

## 5. Geometric Brownian Motion (GBM)

$$
\text{Price Change} = \text{Expected Component} + \text{Random Component}
$$

**Continuous-time:**

$$
dS = \mu S\,dt + \sigma S\,dW
$$

**Discrete form used in simulation:**

$$
S_{t+dt} = S_t \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right) dt + \sigma \sqrt{dt}\, Z\right]
$$

| Symbol | Meaning |
|---|---|
| $S_t$ | current stock price |
| $\mu$ | expected annual return |
| $\sigma$ | annual volatility |
| $dt$ | time step |
| $Z$ | standard normal random shock |

The drift term $\left(\mu - \tfrac{1}{2}\sigma^2\right)dt$ is deterministic; $\sigma\sqrt{dt}\,Z$ is the random component that differentiates each path.

### Current parameters (`src/gbm.py`)

```python
S0 = 100
mu = 0.08
sigma = 0.20
dt = 1 / 252
steps = 252
num_simulations = 3
```

i.e. $100 initial price, 8% expected annual return, 20% annual volatility, daily steps for one year, 3 simulated paths.

### Building a path

Each step uses the previous price: $S_1$ uses $S_0$, $S_2$ uses $S_1$, and generally $S_t$ uses $S_{t-1}$:

```python
paths[:, t] = paths[:, t - 1] * np.exp(
    (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
)
```

A path is a chain: $S_0 \to S_1 \to S_2 \to \dots \to S_{252}$. Multiple paths (e.g. 3) all start from the same $S_0$ but diverge because each receives independent random shocks at every step.

---

## 6. Simulating Multiple Paths

### 6.1 The paths matrix

```python
paths = np.zeros((num_simulations, steps + 1))
```

- **Rows** = simulations/paths, **columns** = time points.
- `steps + 1` columns, since `steps` movements produce `steps + 1` prices (including $S_0$). For `steps = 252`, that's 253 columns.
- `paths[:, 0] = S0` initializes every path to the same starting price.

### 6.2 Generating shocks per step

Every path needs its own shock at each time step:

```python
z = np.random.normal(0, 1, num_simulations)   # one Z per simulation
```

New $Z$ values are drawn independently **every time step**, so each path evolves randomly across its full life (252 draws of `num_simulations` shocks each).

### 6.3 Full loop and vectorization

```python
for t in range(1, steps + 1):
    z = np.random.normal(0, 1, num_simulations)
    paths[:, t] = paths[:, t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    )
```

NumPy applies this element-wise across all simulations at once (**vectorization**) instead of looping over each path individually — with `steps = 252, num_simulations = 3`, `paths.shape == (3, 253)`.

### 6.4 Future price vs. price path

- **Future price** = one value at one future time point (e.g. $S_1 = 101.5$).
- **Price path** = the full sequence $S_0 \to S_1 \to \dots \to S_{252}$.
- **Monte Carlo simulation** = many simulated price paths together.

---

## 7. Formula Reference

| Concept | Formula |
|---|---|
| Simple return | $R_t = \dfrac{P_t - P_{t-1}}{P_{t-1}}$ |
| Log return | $r_t = \ln\left(\dfrac{P_t}{P_{t-1}}\right)$ |
| Mean return | $\bar{r} = \dfrac{\sum r_i}{n}$ |
| Sample variance | $s^2 = \dfrac{\sum(r_i - \bar r)^2}{n-1}$ |
| Standard deviation | $s = \sqrt{s^2}$ |
| Annualized volatility | $\sigma_{\text{annual}} = \sigma_{\text{daily}}\sqrt{252}$ |
| Standard normal shock | $Z \sim N(0,1)$ |
| GBM (discrete) | $S_{t+dt} = S_t \exp\left[\left(\mu - \dfrac{\sigma^2}{2}\right)dt + \sigma\sqrt{dt}\,Z\right]$ |

---

## 8. Code Overview

**`src/returns.py`**
```text
Historical Prices → Log Returns → Mean Return → Variance → Standard Deviation → Annualized Volatility
```

**`src/gbm.py`**
```text
Initial Price S0 → Generate Z ~ N(0,1) → Apply GBM → Calculate S1 → S1 drives S2 → S2 drives S3 → ... → Complete Price Paths
```

---

## 9. Next Step

Convert simulated stock prices into option payoffs. For a call option:

$$
\text{Payoff} = \max(S_T - K, 0)
$$

To be implemented once the payoff concept is fully understood.