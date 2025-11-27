import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Archivos ===
CSV_PAR = "../digits/knn_results.csv"
CSV_SEQ = "../../sec/knn_sequential.csv"

# === Cargar data ===
dfp = pd.read_csv(CSV_PAR)
dfs = pd.read_csv(CSV_SEQ)

# Crear diccionario n_total -> T_sec
seq_times = dfs.set_index("n_total")["time_seq"].to_dict()

# Valores disponibles en ambos CSV
N_VALUES = sorted(set(dfp["n_total"]).intersection(seq_times.keys()))
print("N disponibles:", N_VALUES)

# Procesos para búsqueda del mínimo teórico
P = np.arange(1, 300)

resumen = []

for N in N_VALUES:
    sub = dfp[dfp["n_total"] == N].copy().sort_values("p")

    p = sub["p"].to_numpy(float)
    T_exp = sub["time_compute"].to_numpy(float)

    ntr = sub["n_tr"].iloc[0]
    nte = sub["n_test"].iloc[0]

    # Modelo absoluto:
    # T(p) = a*(ntr*nte)/p + b*p + c*log2(p)
    X = np.column_stack([
        (ntr * nte) / p,
        p,
        np.log2(p)
    ])

    theta, _, _, _ = np.linalg.lstsq(X, T_exp, rcond=None)
    a, b, c = theta

    print(f"\nN={N}: α={a:.3e}, β={b:.3e}, γ={c:.3e}")

    # Función modelo para búsqueda del p óptimo
    def T_model(pp):
        return a*(ntr*nte)/pp + b*pp + c*np.log2(pp)

    Tvals = T_model(P)
    p_opt = P[np.argmin(Tvals)]
    T_opt = np.min(Tvals)

    print(f" → p_opt teórico = {p_opt}, T_opt = {T_opt:.4f} s")

    resumen.append((N, a, b, c, p_opt, T_opt))

    # ---- Gráfica del modelo ----
    plt.figure(figsize=(9,6))
    plt.plot(P, Tvals)
    plt.axvline(p_opt, color='red', linestyle='--', label=f"p_opt={p_opt}")
    plt.title(f"Modelo teórico absoluto - N={N}")
    plt.xlabel("Procesos p")
    plt.ylabel("Tiempo modelo (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"p_optimo_teorico_N{N}.png", dpi=300)
    plt.close()

print("\n================ TABLA RESUMEN ================")
print("N_total   alpha     beta     gamma     p_opt   T_opt(s)")
for row in resumen:
    print(f"{row[0]:5d}   {row[1]:.3e}  {row[2]:.3e}  {row[3]:.3e}   {row[4]:4d}   {row[5]:.4f}")

print("\nListo ✔ Gráficas p_optimo_teorico_N*.png generadas.")
