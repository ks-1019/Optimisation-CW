from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

file_path_t = Path(__file__).resolve().parents[2] / "data" / "trainingI.dat"
file_path_v = Path(__file__).resolve().parents[2] / "data" / "validationI.dat"
training_data = np.loadtxt(file_path_t)
validation_data = np.loadtxt(file_path_v)

x_train, b_train = training_data[:, 0], training_data[:, 1]
x_val, V_val = validation_data[:, 0], validation_data[:, 1]

# Design matrix
def legendre_design_matrix(x, degree):
    X = np.ones((len(x), degree+1))
    X[:, 1] = x
    for k in range(1, degree):
        X[:, k+1] = ((2*k + 1) * x * X[:, k] - (k * X[:, k-1])) / (k+1)
    return X

A_train = legendre_design_matrix(x_train, 8)
# Solve LLS with training data
theta, residuals, _, _ = np.linalg.lstsq(A_train, b_train, rcond=None)

# Validation and MSE
A_val = legendre_design_matrix(x_val, 8)
V_theta = A_val @ theta
val_mse = np.mean((V_theta - V_val)**2)
print(f"Validation MSE: {val_mse:.6e}")
print(f"Sum of residuals: {residuals}")
print(f"Vector of residuals: {A_train @ theta - b_train}")

x_fit = np.linspace(min(x_train.min(), x_val.min()), max(x_train.max(), x_val.max()), 300)
V_fit = legendre_design_matrix(x_fit, 8) @ theta

# 3. Plot training data, least-squares fit, and validation curve
plt.figure(figsize=(8, 5))
plt.scatter(x_train, b_train, color="steelblue", alpha=0.75, label="Training data")
plt.plot(x_fit, V_fit, color="crimson", linewidth=2, label="Least-squares fit (n=8)")

# Optional: overlay clean validation data (sorted to avoid zig-zag lines)
idx = np.argsort(x_val)
plt.plot(x_val[idx], V_val[idx], "k--", linewidth=1.5, label="Validation data (true V(x))")

plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Legendre Least-Squares Approximation")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()