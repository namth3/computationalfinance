import numpy as np
import matplotlib.pyplot as plt

# Model parameters
S0 = 100      # initial stock price
K = 110       # strike price
T = 1.0       # time-to-maturity
r = 0.05      # risk-free rate
sigma = 0.2   # volatility
M = 50        # number of time steps
dt = T / M    # time interval
I = 250000    # number of paths
np.random.seed(0)  # fix the seed

# Simulation paths with geometric Brownian motion
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
            + sigma * np.sqrt(dt) * np.random.standard_normal(I))

# Case: Barrier option
barrier = 120
barrier_option = np.where(S > barrier, abs(S - K), S - K)
barrier_option_value = np.exp(-r * T) * np.sum(np.maximum(barrier_option[-1], 0)) / I

print(f"Barrier option value: {barrier_option_value:.2f}")

# Plot
plt.figure(figsize=(10,6))
plt.plot(S[:,:10], lw=1.5)  # plot the first 10 paths
plt.title('Simulated paths of Geometric Brownian Motion (first 10 paths)')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
