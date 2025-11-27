import matplotlib
matplotlib.use("Agg")  # necesario para SSH sin display

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== ARCHIVOS ====
PAR_CSV = "../digits/knn_results.csv"
SEQ_CSV = "../../sec/knn_sequential.csv"


# ==== 1. LEER CSVs ====
df_par = pd.read_csv(PAR_CSV)
df_seq = pd.read_csv(SEQ_CSV)

# Diccionario: n_total -> tiempo secuencial
seq_times = df_seq.set_index("n_total")["time_seq"].to_dict()

# Tamaños que existan en ambas tablas
problem_sizes = sorted(set(df_par["n_total"]).intersection(seq_times.keys()))

plt.figure(figsize=(10, 6))

# ==== 2. EFICIENCIA PARA CADA n_total ====
for n in problem_sizes:
    sub = df_par[df_par["n_total"] == n].copy()
    sub = sub.sort_values("p")

    p = sub["p"].to_numpy(int)
    T_par = sub["time_total"].to_numpy(float)
    T_seq_n = seq_times[n]

    # Speedup real
    S = T_seq_n / T_par

    # Eficiencia
    E = S / p

    plt.plot(p, E, marker="o", label=f"n={n}")

# ==== 3. CONFIG. DE GRÁFICA ====
plt.xlabel("Número de procesos p")
plt.ylabel("Eficiencia E(p,n) = S(p,n) / p")
plt.title("Eficiencia paralela vs número de procesos")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(title="Tamaño del problema")
plt.tight_layout()

OUTPUT = "efficiency_multi_n.png"
plt.savefig(OUTPUT, dpi=220)
plt.close()

print(f"Gráfica guardada como: {OUTPUT}")
