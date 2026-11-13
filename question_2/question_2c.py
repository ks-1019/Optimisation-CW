# Evaluating p_LLS + implementing gradient method with constant stepsize

from question_2b import S, R

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

C = 2 * (S[1:] - S[0])                                        
s_sq = np.sum(S**2, axis=1)                                   
D = (R[:, [0]]**2 - R[:, 1:]**2) + (s_sq[1:] - s_sq[0])       
p_LLS = np.linalg.solve(C, D.T).T

# checkpoints
file_path_c = Path(__file__).resolve().parents[1] / "data" / "checkpointsII.dat"
ck = np.loadtxt(file_path_c)
ck_idx = ck[:, 0].astype(int) - 1  # indices ending in 0 so 9, 19, 29, ....
ck_pos = ck[:, 1:3]

alpha = 1e-3
def srls(p, S, R):  # this is g_t(p), but doing it for all t at once
    diff = p[:, None, :] - S[None, :, :]          
    sq_dist = np.sum(diff**2, axis=2)             # this is ||p_t - s_i||_2^2
    return 0.25 * np.sum((sq_dist - R**2)**2, axis=1)  # g_t(p_t) for all t

def srls_grad(p, S, R):
    # From II.b: grad g_t(p) = sum_i (||p - s_i||^2 - r_{t,i}^2)(p - s_i)
    diff = p[:, None, :] - S[None, :, :]          
    sq_dist = np.sum(diff**2, axis=2)             
    coeff = (sq_dist - R**2)[:, :, None]          
    return np.sum(coeff * diff, axis=1)    
       
# Run the gradient method


alpha = 1e-3
tol = 1e-12
max_iter = 200000

p_k = p_LLS.copy()
srls_history = []
grad_norm_history = []
for k in range(max_iter):
    g = srls_grad(p_k, S, R)
    grad_norm = np.max(np.linalg.norm(g, axis=1))
    srls_history.append(srls(p_k, S, R))
    grad_norm_history.append(grad_norm)

    if grad_norm < tol:
        print(f'Converged in {k} iterations.')
        break
    p_k = p_k - alpha * g
else:
    print(f"Reached max_iter ({max_iter}) with ||grad|| = {grad_norm:.3e}")


def checkpoint_error(P_est):
    return np.sqrt(np.mean(np.sum((P_est[ck_idx] - ck_pos)**2, axis=1))) # the checkpoint func


kappa = np.linalg.cond(C.T @ C)
print(kappa)
print(f"E_ck (p_LLS): {checkpoint_error(p_LLS):.4f}")
print(f"E_ck (p_GD) : {checkpoint_error(p_k):.4f}")

# 4. Identify time instants (1-based t) where p_GD jumps across the coastline
# Fit coastline line through the 3 anchors and check signed perpendicular distance
line_dir = S[-1] - S[0]
normal = np.array([-line_dir[1], line_dir[0]])
signed_side = (p_k - S[0]) @ normal
ck_side = np.sign((ck_pos[0] - S[0]) @ normal)

jumped_t = np.where(np.sign(signed_side) != ck_side)[0] + 1
print("Time instants t (1-based) on the opposite side of coastline:", jumped_t.tolist())

# 5. Plot both trajectories, anchors, and checkpoints
plt.figure(figsize=(9, 6))
plt.plot(p_LLS[:, 0], p_LLS[:, 1], "o--", color="gray", alpha=0.65, markersize=4, label=r"$p_{LLS,t}$")
plt.plot(p_k[:, 0], p_k[:, 1], "s-", color="crimson", linewidth=1.5, markersize=4, label=r"Gradient method ($p_{GD,t}$)")
plt.scatter(S[:, 0], S[:, 1], marker="^", s=140, color="navy", zorder=6, label="Anchors $s_i$")
plt.scatter(ck_pos[:, 0], ck_pos[:, 1], marker="*", s=180, color="gold", edgecolor="black", zorder=7, label="Checkpoints")

# Label checkpoint times and any jumped points
for idx, pos in zip(ck_idx, ck_pos):
    plt.annotate(f"t={idx+1}", pos, textcoords="offset points", xytext=(6, 6), fontsize=8)
for t_jump in jumped_t:
    plt.annotate(f"t={t_jump}", p_k[t_jump - 1], textcoords="offset points", xytext=(6, -10), color="crimson", fontsize=8)

plt.xlabel("$p_1$")
plt.ylabel("$p_2$")
plt.title("Static Position Estimates: Linearised ($p_{LLS,t}$) vs. Gradient Method ($p_{GD,t}$)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")
plt.tight_layout()
plt.show()