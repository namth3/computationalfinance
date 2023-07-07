import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes_option_price(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def monte_carlo_option_pricing(S, K, r, sigma, T, num_simulations, num_days):
    dt = T
    S_t = np.zeros((num_simulations, num_days))
    option_price_t = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        z = np.random.standard_normal(num_days)
        S_t[i, 0] = S
        option_price_t[i, 0] = max(S - K, 0)
        
        for j in range(1, num_days):
            S_t[i, j] = S_t[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[j])
            option_price_t[i, j] = max(S_t[i, j] - K, 0)
        
        # Update the plot dynamically
        plt.plot(np.arange(j+1), option_price_t[i, :j+1], color='b')
        plt.axhline(y=black_scholes_option_price(S, K, r, sigma, (j+1)*dt), color='r', linestyle='--')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Option Price')
        plt.title('Monte Carlo Simulation of Option Price')
        plt.pause(0.001)
    
    plt.show()

    option_price = np.exp(-r * T) * np.mean(option_price_t[:, -1])
    return option_price, option_price_t

# Option parameters
S0 = 100  # Initial stock price
K = 110  # Strike price
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
T = 1  # Time to expiration (in years)
num_simulations = 1000  # Number of Monte Carlo simulations
num_days = 252  # Number of trading days

# Activate interactive mode
plt.ion()

# Run Monte Carlo simulation
option_price, option_prices = monte_carlo_option_pricing(S0, K, r, sigma, T, num_simulations, num_days)
