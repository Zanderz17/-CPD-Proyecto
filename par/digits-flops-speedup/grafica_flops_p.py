import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ruta del CSV
PAR_CSV = "../digits/knn_results.csv"

# Dimensión del dataset digits (8x8 imágenes)
D = 64
FLOPS_PER_DISTANCE = 3 * D  # resta + mult + suma por componente

# Cargar
df = pd.read_csv(PAR_CSV)

# Distintos tamaños N encontrados
N_VALUES = sorted(df["n_total"].unique())
print("N encontrados:", N_VALUES)

# === GRAFICAR TODAS LAS CURVAS EN UNA SOLA FIGURA ===
plt.figure(figsize=(12,8))

for N in N_VALUES:
    sub = df[df["n_total"] == N].copy().sort_values("p")

    p = sub["p"].to_numpy(float)
    T_comp = sub["time_compute"].to_numpy(float)

    ntr = sub["n_tr"].iloc[0]
    nte = sub["n_test"].iloc[0]

    # FLOPs totales de entrenamiento
    flops_total = ntr * nte * FLOPS_PER_DISTANCE

    # FLOPs/s
    flops_per_sec = flops_total / T_comp

    plt.plot(p, flops_per_sec, marker='o', label=f"N={N}")

plt.xlabel("Número de procesos p")
plt.ylabel("FLOPs por segundo")
plt.title("FLOPs/s vs p para distintos tamaños de dataset (KNN paralelo)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Guardar imagen (modo SSH)
output_file = "flops_vs_p_all_N.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"[OK] Gráfico guardado como: {output_file}")
