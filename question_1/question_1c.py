# Implementing gradient method with constant stepsize to minimise Huber loss func

from question_1a import (
    A_train, b_train, A_val, V_val, theta,
    x_train, x_val, legendre_design_matrix
)
from question_1b import huber_loss, huber_grad

import numpy as np
import matplotlib.pyplot as plt

sigma_max = np.linalg.norm(A_train, ord=2)
t = 1 / (sigma_max**2)

# Run the gradient method

def gradient_method(loss_func, grad_func, theta0, A, b, epsilon, t, tol=1e-12, max_iter=200000):
    theta_k = theta0.copy()
    loss_history = []
    grad_norm_history = []
    for k in range(max_iter):
        g = grad_func(theta_k, A, b, epsilon)
        grad_norm = np.linalg.norm(g)
        loss_history.append(loss_func(theta_k, A, b, epsilon))
        grad_norm_history.append(grad_norm)

        if grad_norm < tol:
            print(f'Converged in {k} iterations.')
            return theta_k, k, np.array(loss_history), np.array(grad_norm_history)

        theta_k = theta_k - t * g
    else:
        print(f"Reached max_iter ({max_iter}) with ||grad|| = {grad_norm:.3e}")

    return theta_k, max_iter, np.array(loss_history), np.array(grad_norm_history)


theta0 = theta.copy()

eps_array = np.logspace(-3, 1, 9)
iters = []
thetas = []
mse_array = []
loss_histories = {}

for eps in eps_array:
    theta_gd, k_stop, loss_history, _ = gradient_method(
        huber_loss, huber_grad, theta0, A_train, b_train, eps, t
    )
    iters.append(k_stop)
    loss_histories[eps] = loss_history
    thetas.append(theta_gd)
    V_theta = A_val @ theta_gd
    val_mse = np.mean((V_theta - V_val)**2)
    mse_array.append(val_mse)

# saved loss_history as well bc it's useful in part ii

ls_mse = np.mean(((A_val @ theta) - V_val)**2)

iters = np.array(iters)
mse_array = np.array(mse_array)
print("Stopping iterations:", iters)

# 3. Plot MSE vs epsilon
plt.figure(figsize=(8, 5))

plt.loglog(eps_array, mse_array, marker="o", color="crimson", linewidth=2, label="Huber Validation MSE")
plt.axhline(y=ls_mse, color="black", linestyle="--", linewidth=1.5, label="LLS MSE")
plt.xlabel("ε")
plt.ylabel("Validation MSE")
plt.title("MSE for Gradient Method with Constant Stepsize for Huber Loss Function")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Summary table for findings
print(f"\n{'epsilon':>10} | {'iterations':>10} | {'val_mse':>12}")
print("-" * 38)
for eps, k_stop, mse in zip(eps_array, iters, mse_array):
    print(f"{eps:10.4e} | {k_stop:10d} | {mse:12.6e}")

# Select the best epsilon (with the least MSE)
best_i = np.argmin(mse_array)
best_eps = eps_array[best_i]
best_theta = thetas[best_i]
print(f"\nBest epsilon: {best_eps:.4e} (Validation MSE: {mse_array[best_i]:.6e})")

r_best = A_train @ best_theta - b_train
abs_r = abs(r_best)

# Print the 10 largest residuals to see the outliers clearly
top_idx = np.argsort(abs_r)[::-1][:10]
print("\nLargest residuals in trainingI.dat:")
for idx in top_idx:
    print(f"Row {idx + 1:3d} (index {idx:3d}): x = {x_train[idx]:7.4f}, b = {b_train[idx]:7.4f}, |residual| = {abs_r[idx]:.4f}")

# Outliers are points exceeding the best Huber threshold - but we're using 0.5 bc it works better:
outlier_idx = np.where(abs_r > 0.5)[0]
print("\nOutlier row numbers (1-based):", (outlier_idx + 1).tolist())

# Plot the fitted function for the best epsilon and highlight outliers
x_fit = np.linspace(-1, 1, 300)
V_best_fit = legendre_design_matrix(x_fit, 7) @ best_theta
V_ls_fit = legendre_design_matrix(x_fit, 7) @ theta

plt.figure(figsize=(8, 5))
plt.scatter(x_train, b_train, color="steelblue", alpha=0.75, label="Training data")
plt.scatter(x_train[outlier_idx], b_train[outlier_idx], color="darkorange", edgecolor="black",
            s=80, zorder=5, label="Detected outliers")
plt.plot(x_fit, V_best_fit, color="crimson", linewidth=2, label=f"Best Huber fit (ε = {best_eps:.2e})")
plt.plot(x_fit, V_ls_fit, color="gray", linestyle=":", linewidth=1.5, label="Least-squares fit")

idx_val = np.argsort(x_val)
plt.plot(x_val[idx_val], V_val[idx_val], "k--", linewidth=1.2, label="Validation data (true V(x))")

plt.xlabel("x")
plt.ylabel("V(x)")
plt.title(f"Best Huber Fit (ε = {best_eps:.2e}) and Detected Outliers")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# For part ii: filter out residuals s.t we only have |residuals| < eps
# and find the smallest eigenval of A.T A
# and also plot the losses for those 3 epsilons for whatever reason
Theta = np.column_stack(thetas)
in_I = np.abs(A_train @ Theta - b_train[:, None]) < eps_array  # created mask for residuals_I

# finding cardinality of I and smallest eigenval
for eps, col in zip(eps_array, in_I.T):
    A_I = A_train[col]  # apply the mask
    l_min = np.linalg.eigvalsh(A_I.T @ A_I)[0]  # smallest eigenvalue
    print(f"eps = {eps:8.2e} | |I| = {col.sum():2d} | lambda_min = {l_min:.4e}")

# plotting stuff to see iteration progression for 3 different epsilon
plt.figure(figsize=(8, 5))
for eps in [1e-2, 1e-1, 1.0]:
    losses = loss_histories[eps]
    plt.semilogy(losses[:-1] - losses[-1], label=rf"$\epsilon = {eps:g}$")
# plt.ylim(1e-14, 10)
plt.xlabel("Iteration $k$")
plt.ylabel(r"$F_\epsilon(\theta^{(k)}) - F_\epsilon(\theta^*)$")
plt.title(r"Objective Suboptimality vs. Iteration $k$")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()