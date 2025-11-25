import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN
# ================================
CSV_FILE = "knn_results.csv"   # <-- Cambia si tu archivo se llama distinto
N_VALUES = [200, 500, 1000, 1500, 1797]

# ================================
# MODELO TEÓRICO NORMALIZADO
# ================================
def T_teor_norm(p, lam, mu):
    """
    Modelo teórico normalizado:
    T_norm(p) = 1/p + lambda * p + mu * log2(p)
    """
    return 1/p + lam * p + mu * np.log2(p)


# ================================
# CARGAR DATOS
# ================================
df = pd.read_csv(CSV_FILE)

# Diccionarios donde guardaremos cada λ y μ
lambda_params = {}
mu_params = {}

print("\n===== AJUSTE DE CONSTANTES λ y μ =====")
for N in N_VALUES:

    # Filtrar por tamaño del dataset
    sub = df[df["n_total"] == N].copy().sort_values("p")

    p = sub["p"].to_numpy(float)
    T_exp = sub["time_compute"].to_numpy(float)  # Solo computación

    # Normalización experimental con T(1)
    T1 = T_exp[0]
    T_norm_exp = T_exp / T1

    # Construir variables para el ajuste lineal:
    # y = T_norm - 1/p
    y = T_norm_exp - 1/p

    # X = [p, log2(p)]
    X = np.column_stack((p, np.log2(p)))

    # Resolver mínimos cuadrados
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    lam, mu = theta

    # Guardar resultados
    lambda_params[N] = lam
    mu_params[N] = mu

    print(f"N={N}:  lambda={lam:.6e}   mu={mu:.6e}")


# ================================
# GRAFIQUE: TEÓRICO VS EXPERIMENTAL
# ================================
print("\n===== GENERANDO GRÁFICAS =====")

for N in N_VALUES:

    sub = df[df["n_total"] == N].copy().sort_values("p")

    p = sub["p"].to_numpy(float)
    T_exp = sub["time_compute"].to_numpy(float)
    T_norm_exp = T_exp / T_exp[0]

    lam = lambda_params[N]
    mu = mu_params[N]

    # Evaluar modelo teórico ajustado
    T_norm_teor = T_teor_norm(p, lam, mu)

    # --- GRAFICAR ---
    plt.figure(figsize=(10,6))
    plt.plot(p, T_norm_exp, marker='o', label="Experimental normalizado")
    plt.plot(p, T_norm_teor, marker='s', label="Teórico ajustado")

    plt.xlabel("Número de procesos p")
    plt.ylabel("Tiempo normalizado")
    plt.title(f"Comparación Teórico vs Experimental (N={N})")
    plt.grid(True)
    plt.legend()

    filename = f"teorico_vs_experimental_N{N}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Gráfico guardado como: {filename}")


print("\n===== PROCESO COMPLETADO =====")
