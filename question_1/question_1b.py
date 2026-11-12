# Numeric continuation of the last part of 1b)

from question_1a import A_train, b_train, theta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def huber_loss(theta, A, b, epsilon):
    r = A @ theta - b
    abs_r = abs(r)
    losses = np.where(abs_r <= epsilon, 0.5 * r**2, epsilon * (abs_r) - 0.5 * epsilon)
    return np.sum(losses)

def huber_grad(theta, A, b, epsilon):
    r = A @ theta - b
    abs_r = np.abs(r)
    L_prime = np.where(abs_r <= epsilon, r, epsilon * np.sign(r))
    return A.T @ L_prime

r = A_train @ theta - b_train
epsilon = np.linalg.norm(r, ord=np.inf) - 1e-3

res = minimize(
    huber_loss,
    x0=theta,
    args=(A_train, b_train, epsilon),
    jac=huber_grad,
    method="BFGS",
    options={"gtol": 1e-12}
)
theta_huber = res.x

# Reporting numerical checks comparing theta_huber and theta_LS
print(f"Residuals exceeding epsilon      : {np.sum(np.abs(r) > epsilon)}")
print(f"||grad F_eps(theta_LS)||_2       : {np.linalg.norm(huber_grad(theta, A_train, b_train, epsilon)):.6e}")
print(f"||theta_huber - theta_LS||_2     : {np.linalg.norm(theta_huber - theta):.6e}")
print(f"F_eps(theta_LS)                  : {huber_loss(theta, A_train, b_train, epsilon):.6f}")
print(f"F_eps(theta_huber)               : {huber_loss(theta_huber, A_train, b_train, epsilon):.6f}")