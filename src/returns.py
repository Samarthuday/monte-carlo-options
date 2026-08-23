"""
returns.py

Calculates historical log returns and annualized volatility from stock prices.

Steps:
1. Calculate log returns:
       r_t = ln(S_t / S_{t-1})

2. Calculate mean return:
       r̄ = sum(r_t) / n

3. Calculate sample variance:
       s² = sum((r_t - r̄)²) / (n - 1)

4. Calculate standard deviation:
       s = sqrt(s²)

5. Annualize volatility:
       σ_annual = σ_daily * sqrt(252)

252 is used as the approximate number of trading days in a year.
"""


import math

prices = [100, 105, 103, 108, 106]

returns = []

for i in range(1, len(prices)):
    ret = math.log(prices[i] / prices[i - 1])
    returns.append(ret)

mean_return = sum(returns) / len(returns)
print("Mean return:", mean_return)

squared_deviations = []
for ret in returns:
    deviation = ret - mean_return # ri - r̄
    squared_deviation = deviation ** 2 # (ri - r̄)²
    squared_deviations.append(squared_deviation) 

variance = sum(squared_deviations) / (len(squared_deviations) - 1) # n-1 is called Bessel's correction
standard_deviation = math.sqrt(variance)
annualized_volatility = standard_deviation * math.sqrt(252) # Assuming 252 trading days in a year

print("Variance:", variance)
print("Standard deviation:", standard_deviation)
print("Annualized volatility:", annualized_volatility)