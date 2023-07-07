import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 1517.93  # Initial index level
T = 3.0  # Time horizon
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
M = 365 * 3  # Number of steps within each simulation
I = 10000  # Number of simulations
dt = T / M  # Time step

np.random.seed(0)  # for reproducibility

# Simulating I paths with M time steps
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    z = np.random.standard_normal(I)  # pseudorandom numbers
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    # geometric Brownian motion

# Adding Brownian Bridge for the knock-out event
KO_level = 0.625 * S0
KO_event = np.zeros(I, dtype=bool)
for t in range(1, M):
    if not np.any(KO_event):  # If no knock-out event yet
        u = np.random.uniform(0, 1, I)  # uniformly distributed random variable
        s1 = S[t - 1, ~KO_event]
        s2 = S[t, ~KO_event]
        tau = dt
        P = np.exp(-2 * (KO_level - s1) * (KO_level - s2) / (sigma ** 2 * tau))  # exit probability
        KO_event[~KO_event] = P > u  # if exit probability exceeds u, knock-out event happens

# Calculating the payoff for each path
redemption = np.zeros_like(S[-1])

# If knock-out event happens
redemption[KO_event] = S[-1, KO_event] / S0

# If no knock-out event
redemption[~KO_event] = 1 + np.abs(S[-1, ~KO_event] / S0 - 1)

# Calculating the Monte Carlo estimator
C0 = np.exp(-r * T) * np.mean(redemption)

print(f"Value of the Twin-Win Certificate: {C0:.3f}")

# Visualizing the paths
plt.figure(figsize=(10, 6))
plt.plot(S[:, :100])  # Plotting the first 100 paths
plt.title('Simulated Paths of the Underlying Index')
plt.ylabel('Index Level')
plt.xlabel('Time Steps')
plt.show()
