import numpy as np
import pandas as pd

# cargar CSV
df = pd.read_csv("../digits/knn_results.csv")

# lista de N_totales a analizar
N_VALUES = [200, 500, 1000, 1500, 1797]

for N in N_VALUES:
    # filtrar por n_total
    sub = df[df["n_total"] == N].copy()
    sub = sub.sort_values("p")   # asegurarnos de que p va en orden
    
    p = sub["p"].to_numpy(dtype=float)
    T_exp = sub["time_total"].to_numpy(dtype=float)
    
    # normalización
    T1 = T_exp[0]
    T_norm_exp = T_exp / T1
    
    # y y matriz X
    y = T_norm_exp - 1.0/p
    X = np.column_stack((p, np.log2(p)))
    
    # mínimos cuadrados
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    lam_hat, mu_hat = theta
    
    print(f"N_total={N}: lambda={lam_hat:.6e}, mu={mu_hat:.6e}")
