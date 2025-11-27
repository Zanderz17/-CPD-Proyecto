import matplotlib
matplotlib.use("Agg")  # necesario en SSH (sin display)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== ARCHIVOS ====
PAR_CSV = "../digits/knn_results.csv"
SEQ_CSV = "../../sec/knn_sequential.csv"

# ==== 1. LEER CSVs ====
df_par = pd.read_csv(PAR_CSV)
df_seq = pd.read_csv(SEQ_CSV)

# Esperamos columnas:
# df_par: p,n_total,n_tr,n_test,accuracy,time_total,time_compute,time_comm
# df_seq: n_total,n_tr,n_test,accuracy,time_seq

# Diccionario: n_total -> T_seq(n)
seq_times = df_seq.set_index("n_total")["time_seq"].to_dict()

# Intersección de tamaños que existen en ambos CSV
problem_sizes = sorted(set(df_par["n_total"]).intersection(seq_times.keys()))

print("Tamaños de problema considerados:", problem_sizes)

plt.figure(figsize=(10, 6))

# ==== 2. SPEEDUP PARA CADA n_total ====
for n in problem_sizes:
    sub = df_par[df_par["n_total"] == n].copy()
    sub = sub.sort_values("p")

    p = sub["p"].to_numpy(int)
    T_par = sub["time_total"].to_numpy(float)
    T_seq_n = seq_times[n]

    speedup = T_seq_n / T_par

    plt.plot(p, speedup, marker="o", label=f"n={n}")

# ==== 3. LÍNEA DE SPEEDUP IDEAL S = p ====
p_unique = sorted(df_par["p"].unique())
plt.plot(p_unique, p_unique, "--", color="gray", label="Speedup ideal S=p")

# ==== 4. FORMATO DE GRÁFICA ====
plt.xlabel("Número de procesos p")
plt.ylabel("Speedup S(p,n) = T_seq(n) / T_par(p,n)")
plt.title("Speedup real vs número de procesos para distintos tamaños de problema")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(title="Tamaño del problema")
plt.tight_layout()

OUTPUT = "speedup_real_multi_n.png"
plt.savefig(OUTPUT, dpi=220)
plt.close()

print(f"Gráfica guardada como: {OUTPUT}")
