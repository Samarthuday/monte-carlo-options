# Monte Carlo Options — Learning Notes

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

# 1. Project Goal

Build a Monte Carlo-based options pricing model.

```text
Historical Prices
      ↓
Log Returns
      ↓
Volatility
      ↓
GBM
      ↓
Simulated Price Paths
      ↓
Terminal Prices ST
      ↓
Option Payoffs
      ↓
Risk-Neutral Pricing
      ↓
Monte Carlo Option Price
      ↓
Validation
      ↓
American Options
```

---

# 2. Returns

## 2.1 Simple Return

\[
R_t=\frac{P_t-P_{t-1}}{P_{t-1}}
\]

A 10% gain and a 10% loss are not symmetric.

---

## 2.2 Log Return

\[
r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)
\]

Implemented in `src/returns.py`.

Log returns are additive:

\[
\ln(P_1/P_0)+\ln(P_2/P_1)
=
\ln(P_2/P_0)
\]

Therefore:

\[
\boxed{\text{Sum of log returns}=\text{Total log return}}
\]

---

# 3. Return Statistics

## Mean

\[
\bar r=\frac{\sum r_i}{n}
\]

## Sample Variance

\[
s^2=
\frac{\sum(r_i-\bar r)^2}{n-1}
\]

The \(n-1\) denominator is Bessel's correction.

## Standard Deviation

\[
s=\sqrt{s^2}
\]

Standard deviation measures the spread of returns and is used as volatility.

## Annualized Volatility

Assuming 252 trading days:

\[
\boxed{
\sigma_{\text{annual}}
=
\sigma_{\text{daily}}\sqrt{252}
}
\]

---

# 4. Standard Normal Random Shock

\[
Z\sim N(0,1)
\]

In Python:

```python
z = np.random.normal(0, 1)
```

The theoretical mean is 0 and standard deviation is 1.

Individual values are not restricted to \([-1,1]\).

\(Z\) is the source of randomness in GBM.

---

# 5. Geometric Brownian Motion

Continuous form:

\[
dS=\mu Sdt+\sigma S\,dW
\]

Discrete simulation form:

\[
\boxed{
S_{t+dt}
=
S_t
\exp\left[
\left(\mu-\frac{1}{2}\sigma^2\right)dt
+
\sigma\sqrt{dt}Z
\right]
}
\]

The deterministic drift component is:

\[
\left(\mu-\frac{1}{2}\sigma^2\right)dt
\]

The random component is:

\[
\sigma\sqrt{dt}Z
\]

---

# 6. Price Paths

A single path is:

\[
S_0\rightarrow S_1\rightarrow S_2\rightarrow\cdots\rightarrow S_T
\]

Each price uses the previous price:

\[
S_1\text{ uses }S_0
\]

\[
S_2\text{ uses }S_1
\]

\[
S_t\text{ uses }S_{t-1}
\]

Multiple simulations are stored in a matrix:

```python
paths = np.zeros((num_simulations, steps + 1))
```

Rows = simulations.

Columns = time points.

`steps + 1` is required because:

```text
5 steps:
S0 → S1 → S2 → S3 → S4 → S5
```

contains 6 prices.

The first column is initialized with:

```python
paths[:, 0] = S0
```

The last column contains terminal prices:

```python
terminal_prices = paths[:, -1]
```

---

# 7. Call Options

A call option gives the buyer the right, but not the obligation, to buy the underlying stock at the strike price.

The strike price is:

\[
K
\]

The current stock price and strike price are different quantities.

Example:

```text
Current stock price = $100
Strike price        = $110
```

The contract gives the right to buy at $110.

The option premium is separate from the strike price.

---

# 8. Call Payoff

At expiration:

\[
S_T=\text{stock price at expiration}
\]

The European call payoff is:

\[
\boxed{
C_T=\max(S_T-K,0)
}
\]

If:

\[
S_T>K
\]

the call is in the money.

If:

\[
S_T=K
\]

it is at the money.

If:

\[
S_T<K
\]

it is out of the money.

Payoff is not the same as profit.

\[
\text{Profit}
=
\text{Payoff}
-
\text{Premium}
\]

---

# 9. Terminal Prices and Payoffs

The Monte Carlo connection is:

```text
GBM
 ↓
Many price paths
 ↓
Take the last column
 ↓
Terminal prices ST
 ↓
Call payoff max(ST-K,0)
 ↓
Many simulated payoffs
```

For example:

```text
ST = [90, 100, 110, 120, 140]
K  = 110

Payoffs = [0, 0, 0, 10, 30]
```

Implemented in `src/payoff.py`.

---

# 10. Expected Payoff

With \(N\) simulations:

\[
\boxed{
E[C_T]
\approx
\frac{1}{N}
\sum_{i=1}^{N}C_T^{(i)}
}
\]

The summation simply adds all simulated payoffs.

Dividing by \(N\) gives the average payoff.

This is a Monte Carlo estimate of the expected payoff at expiration.

---

# 11. Discounting

A future dollar is worth less than a dollar today because money can earn a return over time.

The continuous-compounding discount factor is:

\[
e^{-rT}
\]

where:

- \(r\) = risk-free rate
- \(T\) = time to expiration

Therefore:

\[
\boxed{
V_0=e^{-rT}E[C_T]
}
\]

---

# 12. Risk-Neutral Pricing

A crucial distinction:

Historical simulation may use an estimated expected return:

\[
\mu
\]

But standard derivative pricing uses the **risk-neutral measure**.

Under risk-neutral pricing, the drift becomes:

\[
\boxed{r}
\]

instead of the historical expected return \(\mu\).

Therefore the pricing GBM is:

\[
\boxed{
S_{t+dt}
=
S_t
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)dt
+
\sigma\sqrt{dt}Z
\right]
}
\]

This is implemented in `src/monte_carlo.py`.

---

# 13. Monte Carlo European Option Price

For a European call:

\[
C_T^{(i)}
=
\max(S_T^{(i)}-K,0)
\]

The Monte Carlo price is:

\[
\boxed{
C_0
\approx
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\max(S_T^{(i)}-K,0)
}
\]

The same structure applies to a put:

\[
\boxed{
P_0
\approx
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\max(K-S_T^{(i)},0)
}
\]

---

# 14. Monte Carlo Standard Error

Monte Carlo is an estimate and therefore contains sampling error.

If \(s_{\text{payoff}}\) is the sample standard deviation of simulated payoffs:

\[
\boxed{
SE
=
e^{-rT}
\frac{s_{\text{payoff}}}{\sqrt{N}}
}
\]

As \(N\) increases:

\[
SE\propto\frac{1}{\sqrt{N}}
\]

Therefore, reducing Monte Carlo error substantially requires many more simulations.

---

# 15. Confidence Interval

An approximate 95% confidence interval is:

\[
\boxed{
V_0\pm1.96SE
}
\]

The confidence interval describes Monte Carlo sampling uncertainty.

It does not mean that the underlying pricing model is necessarily correct.

---

# 16. Black-Scholes Validation

For European options, we can compare Monte Carlo against the analytical Black-Scholes model.

For a call:

\[
C=S_0N(d_1)-Ke^{-rT}N(d_2)
\]

where:

\[
d_1=
\frac{
\ln(S_0/K)+(r+\sigma^2/2)T
}{
\sigma\sqrt{T}
}
\]

\[
d_2=d_1-\sigma\sqrt{T}
\]

Black-Scholes provides a benchmark.

As the number of Monte Carlo simulations increases, the Monte Carlo estimate should generally move closer to the Black-Scholes price, subject to random sampling error.

Implemented in:

```text
src/black_scholes.py
```

---

# 17. European vs American Options

European:

```text
Exercise only at expiration
```

American:

```text
Exercise at any time up to expiration
```

For European options we only need:

\[
S_T
\]

For American options we must consider exercise decisions at multiple time points.

---

# 18. Why Use an American Put?

For a non-dividend-paying stock, an American call does not generally benefit from early exercise.

Therefore an American put is a useful first example for demonstrating early exercise.

The American put payoff is:

\[
\boxed{
\max(K-S_t,0)
}
\]

at any possible exercise time \(t\).

---

# 19. Longstaff-Schwartz Monte Carlo

American option pricing introduces a decision:

\[
\boxed{
\text{Exercise now OR continue holding?}
}
\]

Longstaff-Schwartz estimates the continuation value using regression.

At each exercise date:

1. Find paths that are in the money.
2. Calculate immediate exercise value.
3. Estimate continuation value using regression.
4. Exercise if immediate value is greater than continuation value.
5. Work backward through time.

The current educational implementation uses:

\[
1,\quad S,\quad S^2
\]

as regression basis functions.

Implemented in:

```text
src/american_option.py
```

---

# 20. Full Project Flow

```text
Historical Prices
        ↓
Log Returns
        ↓
Historical Volatility
        ↓
GBM
        ↓
Simulated Price Paths
        ↓
Terminal Prices
        ↓
Option Payoffs
        ↓
Expected Payoff
        ↓
Risk-Neutral Discounting
        ↓
Monte Carlo European Price
        ↓
Standard Error / Confidence Interval
        ↓
Black-Scholes Validation
        ↓
Convergence Analysis
        ↓
American Put
        ↓
Longstaff-Schwartz Early Exercise
```

---

# 21. Current Code References

`src/returns.py`

```text
Historical prices
→ Log returns
→ Mean
→ Variance
→ Standard deviation
→ Annualized volatility
```

`src/gbm.py`

```text
Initial price
→ Random shocks
→ GBM
→ Simulated price paths
```

`src/payoff.py`

```text
Terminal prices
→ Call / put payoff
```

`src/monte_carlo.py`

```text
Risk-neutral GBM
→ Terminal prices
→ Payoffs
→ Expected payoff
→ Discounting
→ European option price
```

`src/black_scholes.py`

```text
Inputs
→ d1, d2
→ Analytical European price
```

`src/american_option.py`

```text
Risk-neutral paths
→ Work backwards
→ Immediate exercise value
→ Continuation regression
→ Exercise decision
→ American put price
```

---

# 22. Important Modeling Distinction

Historical expected return:

\[
\mu
\]

is useful when describing or simulating real-world stock behavior.

Risk-neutral pricing uses:

\[
r
\]

as the drift.

Therefore:

```text
Historical / real-world simulation:
    drift = μ

Option pricing:
    drift = r
```

This distinction is fundamental.

---

# 23. Final Objective

The project demonstrates:

\[
\boxed{
\text{Statistics}
\rightarrow
\text{Stochastic Modeling}
\rightarrow
\text{Monte Carlo}
\rightarrow
\text{Derivative Pricing}
}
\]

and eventually:

\[
\boxed{
\text{European Pricing}
\rightarrow
\text{Validation}
\rightarrow
\text{American Early Exercise}
}
\]

The goal is to understand the mathematics behind every stage and implement the model rather than treating the final price as a black box.
