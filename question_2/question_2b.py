import numpy as np
from pathlib import Path

file_path_a = Path(__file__).resolve().parents[1] / "data" / "anchorsII.dat"
file_path_r = Path(__file__).resolve().parents[1] / "data" / "rangesII.dat"
S = np.loadtxt("file_path_a")  # 3x2, rows s_i
R = np.loadtxt("file_path_r")  # 60x3
r = R[0]  # t = 1


def hess(p, r, S):
    H = np.zeros((2, 2))
    for i in range(3):
        d = p - S[i]
        H += 2*np.outer(d, d) + (d @ d - r[i]**2) * np.eye(2)
    return H


for k in range(3):
    H = hess(S[k], r, S)
    print(k+1, np.linalg.eigvalsh(H))