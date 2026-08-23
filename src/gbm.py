"""
gbm.py

Generate stock-price paths using Geometric Brownian Motion (GBM).

GBM:
S(t+dt) = S(t) * exp(
    (mu - 0.5 * sigma^2) * dt
    + sigma * sqrt(dt) * Z
)

Z ~ N(0, 1)
"""

import numpy as np

# Model parameters
S0 = 100
mu = 0.08
sigma = 0.20

# Time step: 1 trading day
dt = 1 / 252

steps = 252
num_simulations = 3


# Create matrix:
# rows = simulations
# columns = time points
paths = np.zeros((num_simulations, steps + 1))

# All paths start at S0
paths[:, 0] = S0


# Generate the price paths
for t in range(1, steps + 1):

    # One random shock for each simulation
    z = np.random.normal(0, 1, num_simulations)

    # GBM: new price uses the previous price
    paths[:, t] = paths[:, t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * z
    )


# Display results
print("Shape:", paths.shape)
print("First 3 paths, first 5 time points:")
print(paths[:3, :5])