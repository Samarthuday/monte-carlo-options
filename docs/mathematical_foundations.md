# Mathematical Foundations

Detailed mathematical derivations for the models and methods implemented in `mcoptions`.

---

## Table of Contents

1. [Geometric Brownian Motion](#geometric-brownian-motion)
2. [Risk-Neutral Valuation](#risk-neutral-valuation)
3. [Black–Scholes Formula](#black--scholes-formula)
4. [The Greeks](#the-greeks)
5. [Monte Carlo Pricing](#monte-carlo-pricing)
6. [Variance Reduction](#variance-reduction)
7. [American Options & Optimal Stopping](#american-options--optimal-stopping)

---

## Geometric Brownian Motion

The stock price follows a **geometric Brownian motion** under the real-world (historical) measure:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

where:
- $\mu$ = expected annual return (drift)
- $\sigma$ = annual volatility (diffusion)
- $W_t$ = standard Brownian motion

### Analytical Solution

Integrating the SDE yields:

$$S_T = S_0 \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right]$$

where $Z \sim N(0,1)$.

This allows **direct sampling** of terminal prices without discretization error.

---

## Risk-Neutral Valuation

Under the **risk-neutral measure**, the drift changes to the risk-free rate:

$$dS_t = (r - q)S_t dt + \sigma S_t dW_t$$

where:
- $r$ = risk-free rate
- $q$ = dividend yield (continuous)

### No-Arbitrage Principle

A call option's value equals the expected discounted payoff under the risk-neutral measure:

$$C_0 = e^{-rT} \mathbb{E}^Q[\max(S_T - K, 0)]$$

The risk-neutral expectation is equivalent to pricing under a **martingale measure** where $\mathbb{E}^Q[e^{-rT}S_T] = S_0 e^{-qT}$.

### Why Not Historical Volatility?

The historical expected return $\mu$ does **not** enter option prices. Instead:
- **Historical $\mu$**: Used for return forecasting, risk estimation
- **Historical $\sigma$**: Used for initial volatility estimate
- **Risk-neutral $r$**: Used for pricing (removes drift uncertainty)

---

## Black–Scholes Formula

For a **European call** on a dividend-paying stock:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where:

$$d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

$$N(\cdot) = \text{CDF of standard normal}$$

### European Put (Put-Call Parity)

$$P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)$$

**Put-Call Parity**:
$$C - P = S_0 e^{-qT} - K e^{-rT}$$

This relationship holds for both analytical and MC estimates.

---

## The Greeks

Greeks measure option value sensitivity to market parameters. They're essential for risk management.

### Delta: Stock Price Sensitivity

$$\Delta = \frac{\partial V}{\partial S}$$

Tells you how much the option value changes for a \$1 move in the stock.

**Call Delta**:
$$\Delta_C = e^{-qT} N(d_1)$$

**Put Delta**:
$$\Delta_P = -e^{-qT} N(-d_1)$$

Interpretation:
- Call Delta ∈ (0, 1) — upside exposure
- Put Delta ∈ (−1, 0) — downside exposure

### Gamma: Delta Sensitivity

$$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{e^{-qT} \phi(d_1)}{S_0 \sigma \sqrt{T}}$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ is the standard normal PDF.

Interpretation:
- High Gamma = Delta changes rapidly (higher risk if you rebalance infrequently)
- Gamma highest at-the-money (ATM)
- Always positive for both calls and puts

### Vega: Volatility Sensitivity

$$\mathcal{V} = \frac{\partial V}{\partial \sigma} = S_0 e^{-qT} \phi(d_1) \sqrt{T}$$

Interpretation:
- Positive for both calls and puts
- Highest ATM
- Useful for trading volatility

### Theta: Time Decay

$$\Theta = -\frac{\partial V}{\partial t}$$

For a **call**:
$$\Theta_C = -S_0 e^{-qT} \phi(d_1) \frac{\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2) + q S_0 e^{-qT} N(d_1)$$

Interpretation:
- Usually negative for long calls (time decay hurts you)
- May be positive for deep ITM puts (early exercise premium)

### Rho: Interest Rate Sensitivity

$$\rho = \frac{\partial V}{\partial r}$$

For a **call**:
$$\rho_C = K T e^{-rT} N(d_2)$$

Interpretation:
- Positive for calls (higher rates → higher call value)
- Negative for puts
- Most important for long-dated options

---

## Monte Carlo Pricing

### Basic Algorithm

1. **Simulate** $N$ stock price paths using GBM
2. **Calculate** payoff at maturity for each path
3. **Average** the payoffs across paths
4. **Discount** at the risk-free rate

$$V_0 \approx e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}(S_T^{(i)})$$

### Standard Error

If $Y_i = e^{-rT} \times \text{Payoff}_i$:

$$\hat{\sigma}^2 = \frac{1}{N-1} \sum_{i=1}^{N} (Y_i - \bar{Y})^2$$

$$SE = \frac{\hat{\sigma}}{\sqrt{N}}$$

The confidence interval widens as $1/\sqrt{N}$, so you need **4× more paths** for half the error.

### Why Terminal Sampling?

For European options, we only need $S_T$. The analytical solution is:

$$S_T = S_0 \exp\left[(r - q - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right]$$

**Direct approach** (1 random number per path):
- Generate $Z \sim N(0,1)$
- Compute $S_T$ directly
- Calculate payoff, discount

**Full-path approach** (252 random numbers per path if daily steps):
- Simulate 252 steps
- Extract final price
- Calculate payoff, discount

Both converge to the same answer, but direct sampling is **300× faster** with no approximation error.

---

## Variance Reduction

### Antithetic Variates

For each $Z \sim N(0,1)$, compute payoffs for both $Z$ and $-Z$:

$$\hat{V}_{\text{AV}} = \frac{1}{2N} \sum_{i=1}^{N} \left[ e^{-rT} \text{Payoff}(S_T^+) + e^{-rT} \text{Payoff}(S_T^-) \right]$$

where $S_T^{\pm} = S_0 \exp\left[(r - q - \tfrac{1}{2}\sigma^2)T \pm \sigma\sqrt{T}Z_i\right]$.

**Variance reduction**: ~40–50% (the $+Z$ and $-Z$ outcomes are negatively correlated).

### Control Variates

Choose a **control** $X$ with known expectation $\mu_X$, correlated with your payoff $Y$:

$$\hat{V}_{\text{CV}} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \left[ Y_i - \beta^*(X_i - \mu_X) \right]$$

where the **optimal coefficient** is:

$$\beta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

**Example**: For a call, use $X = e^{-rT}S_T$ (discounted spot price).

$$\mathbb{E}[X] = e^{-qT} S_0 \quad \text{(known analytically)}$$

Reduces variance by **40–90%** depending on correlation.

---

## American Options & Optimal Stopping

### The Optimal Exercise Problem

An American put can be exercised at any time $\tau \in [0, T]$:

$$V_{\text{American}} = \max_{\tau} \mathbb{E}^Q\left[e^{-r\tau} \text{Payoff}(S_\tau)\right]$$

### Longstaff–Schwartz Method

**Backward induction** through time steps $t_m = T, T - \Delta t, \ldots, \Delta t$:

At each step:
1. **Compute intrinsic value** (immediate exercise): $I_t = \max(K - S_t, 0)$
2. **Estimate continuation value** via regression:
   $$C_t(S_t) \approx \sum_j \beta_j \phi_j(S_t)$$
   where $\phi_j$ are basis functions (polynomials, Laguerre, etc.)
3. **Exercise rule**: Compare $I_t$ vs. $C_t$; choose max
4. **Discount** exercised cash flows back to $t=0$

### Regression Basis

Common choices:
- **Hermite**: $1, S, S^2$ (polynomial)
- **Normalized**: $1, S/K, (S/K)^2$ (rescaled)
- **Laguerre**: $e^{-S/2} L_k(S)$ (for jump processes)

Higher-order polynomials reduce approximation error but increase variance.

### Bias & Variance

- **LSM is downward-biased** (underestimates option value)
  - Approximates continuation value with finite basis
  - Misses some optimal exercise opportunities
- **CRR binomial is downward-biased** (different reason)
  - Finite lattice approximates continuous state space
- Both converge as parameters increase (more paths, finer grid)

---

## Numerical Stability

### Floating-Point Precision

For deep out-of-the-money options or extreme parameters, use **log-space** calculations:

$$\ln C = \ln S_0 + \ln N(d_1) + \text{lower-order terms}$$

### Convergence Diagnostics

- **Bias check**: Run 100 independent MC trials, compute empirical std
- **RMSE verification**: Plot log(RMSE) vs. log(N); slope should be −0.5
- **Sanity tests**: $C \geq \max(S_0 - Ke^{-rT}, 0)$, $C \leq S_0$

---

## References

- Black, F., Scholes, M. (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Longstaff, F. A., Schwartz, E. S. (2001). "Valuing American options by simulation." *Review of Financial Studies*.
- Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
