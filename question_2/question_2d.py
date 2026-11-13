from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Place this file at question_2/question_2d.py in the coursework repository.
data_dir = Path(__file__).resolve().parents[1] / "data"
anchors = np.loadtxt(data_dir / "anchorsII.dat").reshape(3, 2)
ranges = np.loadtxt(data_dir / "rangesII.dat")
checkpoints = np.loadtxt(data_dir / "checkpointsII.dat").reshape(-1, 3)
T = len(ranges)
assert ranges.shape == (T, 3)

# The coursework labels times t = 1, ..., T; NumPy uses 0, ..., T-1.
check_t = checkpoints[:, 0].astype(int) - 1
assert np.all((check_t >= 0) & (check_t < T))
check_p = checkpoints[:, 1:]


def objective_and_gradient(P, lam):
    """G_lambda(P) and its gradient; P[t] is the 2D position at time t."""
    displacement = P[:, None, :] - anchors[None, :, :]  # T x 3 x 2
    residual = np.sum(displacement**2, axis=2) - ranges**2  # T x 3
    value = 0.25 * np.sum(residual**2)
    grad = np.sum(residual[:, :, None] * displacement, axis=1)

    differences = P[1:] - P[:-1]  # D P
    value += 0.5 * lam * np.sum(differences**2)
    grad[:-1] -= lam * differences
    grad[1:] += lam * differences  # lambda D^T D P
    return value, grad


def checkpoint_error(P):
    return np.sqrt(np.mean(np.sum((P[check_t] - check_p)**2, axis=1)))


def gradient_descent(P0, lam, step, tol=1e-6, max_iter=100000):
    P = P0.copy()
    for k in range(max_iter + 1):
        value, grad = objective_and_gradient(P, lam)
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= tol or k == max_iter:
            return P, k, value, grad_norm, grad_norm <= tol
        P -= step * grad


# Part II.c: compute the static estimates used to initialise Part II.d.
# Linearised least-squares initialisation at every time t: C p = d_t.
C = 2 * (anchors[1:] - anchors[0])
d = (ranges[:, [0]]**2 - ranges[:, 1:]**2
     + np.sum(anchors[1:]**2, axis=1) - np.sum(anchors[0]**2))
P_static = np.linalg.lstsq(C, d.T, rcond=None)[0].T

# Independent static gradient steps with the stepsize prescribed in II.c.
static_tol = 1e-8
for k in range(100001):
    displacement = P_static[:, None, :] - anchors[None, :, :]
    residual = np.sum(displacement**2, axis=2) - ranges**2
    static_grad = np.sum(residual[:, :, None] * displacement, axis=1)
    if np.max(np.linalg.norm(static_grad, axis=1)) <= static_tol or k == 100000:
        break
    P_static -= 1e-3 * static_grad
print(f"Static estimates: {k} steps; max gradient norm = "
      f"{np.max(np.linalg.norm(static_grad, axis=1)):.3e}")

# Part II.d(ii): choose lambda using checkpoint accuracy.
lambdas = [1e-1, 1, 10, 1e2, 1e3]
results = {}
print(f"{'lambda':>8} {'iterations':>12} {'G_lambda':>16} "
      f"{'E_ck':>12} {'|gradient|':>12} {'converged':>10}")
for lam in lambdas:
    P, count, value, grad_norm, converged = gradient_descent(
        P_static, lam, 1 / (1000 + 4 * lam)
    )
    results[lam] = P
    print(f"{lam:8g} {count:12d} {value:16.8g} "
          f"{checkpoint_error(P):12.6g} {grad_norm:12.3e} {str(converged):>10}")

best_lam = min(lambdas, key=lambda lam: checkpoint_error(results[lam]))
P_best = results[best_lam]
print(f"Selected lambda = {best_lam:g}; E_ck = {checkpoint_error(P_best):.6g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].semilogx(lambdas, [checkpoint_error(results[lam]) for lam in lambdas], "o-")
axes[0].set(xlabel=r"$\lambda$", ylabel=r"$E_{\rm ck}$", title="Checkpoint error")
for lam, P in results.items():
    axes[1].plot(P[:, 0], P[:, 1], label=rf"$\lambda={lam:g}$")
axes[1].plot(anchors[:, 0], anchors[:, 1], "k^", label="Anchors")
axes[1].plot(check_p[:, 0], check_p[:, 1], "kx", label="Checkpoints")
axes[1].set(xlabel=r"$p_1$", ylabel=r"$p_2$", title="Tracked trajectories")
axes[1].set_aspect("equal", adjustable="datalim")
for ax in axes:
    ax.grid(True, alpha=0.3)
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()

# Part II.d(iii): repeat at the selected lambda, starting at the mirror image.
P_mirror_start = P_best.copy()
P_mirror_start[:, 1] *= -1
P_mirror, count, mirror_value, grad_norm, converged = gradient_descent(
    P_mirror_start, best_lam, 1 / (1000 + 4 * best_lam)
)
best_value, _ = objective_and_gradient(P_best, best_lam)
print(f"Original: G = {best_value:.8g}; E_ck = {checkpoint_error(P_best):.8g}")
print(f"Mirrored: G = {mirror_value:.8g}; E_ck = {checkpoint_error(P_mirror):.8g}; "
      f"iterations = {count}; |gradient| = {grad_norm:.3e}; converged = {converged}")

plt.figure(figsize=(7, 6))
plt.plot(P_best[:, 0], P_best[:, 1], label="Original initialisation")
plt.plot(P_mirror[:, 0], P_mirror[:, 1], label="Mirrored initialisation")
plt.plot(anchors[:, 0], anchors[:, 1], "k^", label="Anchors")
plt.plot(check_p[:, 0], check_p[:, 1], "kx", label="Checkpoints")
plt.xlabel(r"$p_1$")
plt.ylabel(r"$p_2$")
plt.title(rf"Tracking with $\lambda={best_lam:g}$")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
