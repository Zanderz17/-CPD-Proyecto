import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PAR = "../digits/knn_results.csv"
CSV_SEQ = "../../sec/knn_sequential.csv"

# Cargar datos
dfp = pd.read_csv(CSV_PAR)
dfs = pd.read_csv(CSV_SEQ)

N_VALUES = sorted(dfp["n_total"].unique())

for N in N_VALUES:
    sub_p = dfp[dfp["n_total"] == N].copy().sort_values("p")
    sub_s = dfs[dfs["n_total"] == N].copy()

    p = sub_p["p"].to_numpy(float)
    T_exp = sub_p["time_compute"].to_numpy(float)

    ntr = sub_p["n_tr"].iloc[0]
    nte = sub_p["n_test"].iloc[0]

    # Construimos:
    # T(p) = a*(ntr*nte)/p + b*p + c*log2(p)
    X = np.column_stack((
        (ntr * nte) / p,
        p,
        np.log2(p)
    ))

    # Ajustar parámetros a,b,c
    theta, _, _, _ = np.linalg.lstsq(X, T_exp, rcond=None)
    a, b, c = theta

    print(f"N={N}: α={a:.3e}  β={b:.3e}  γ={c:.3e}")

    # Evaluar modelo
    T_teor = X @ theta

    # ---- Gráfica ----
    plt.figure(figsize=(10,6))
    plt.plot(p, T_exp, marker="o", label="Experimental")
    plt.plot(p, T_teor, marker="s", label="Teórico ajustado")

    plt.xlabel("p")
    plt.ylabel("Tiempo de cómputo (s)")
    plt.title(f"Comparación teórico vs experimental (N={N})")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"teo_vs_exp_N{N}.png", dpi=300)
    plt.close()
