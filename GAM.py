import numpy as np
from pygam import LinearGAM, s
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set a seed for reproducibility
np.random.seed(42)

# Simulate data with 1000 records
n = 1000
X = np.random.uniform(0, 10, n)  # Simulating predictor X
Y = np.random.uniform(0, 5, n)   # Simulating predictor Y
Z = 2 * np.sin(X) + 0.5 * Y + np.random.normal(0, 1, n)  # Simulating the outcome Z

# Create a GAM model
gam = LinearGAM(s(0) + s(1)).fit(np.column_stack((X, Y)), Z)

# Get basis functions used
basis_functions_X = gam.terms[0].basis
basis_functions_Y = gam.terms[1].basis
print("Basis functions used (GAM - X):", basis_functions_X)
print("Basis functions used (GAM - Y):", basis_functions_Y)

# Get smoothing functions
smooth_function = gam.predict(np.column_stack((X, Y)))
print("Smooth function (GAM):", smooth_function[:10])

# Model assessment (GAM)
Z_pred_gam = gam.predict(np.column_stack((X, Y)))
mse_gam = mean_squared_error(Z, Z_pred_gam)
r2_gam = r2_score(Z, Z_pred_gam)
print("Mean Squared Error (MSE) - GAM:", mse_gam)
print("R-squared (R^2) - GAM:", r2_gam)

# Fit a Linear Regression (OLS) for comparison
ols = LinearRegression()
X_ols = np.column_stack((X, Y))
start_time = time.time()
ols.fit(X_ols, Z)
ols_runtime = time.time() - start_time

# Model assessment (OLS)
Z_pred_ols = ols.predict(X_ols)
mse_ols = mean_squared_error(Z, Z_pred_ols)
r2_ols = r2_score(Z, Z_pred_ols)
print("Mean Squared Error (MSE) - OLS:", mse_ols)
print("R-squared (R^2) - OLS:", r2_ols)
print("Runtime - OLS:", ols_runtime)
