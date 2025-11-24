import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("knn_results.csv")

# --------- 1. TIEMPO TOTAL VS P PARA CADA N ---------
plt.figure()

for n in sorted(df["n_total"].unique()):
    sub = df[df["n_total"] == n]
    plt.plot(sub["p"], sub["time_total"], marker="o", label=f"n={n}")

plt.xlabel("Número de procesos (p)")
plt.ylabel("Tiempo total (s)")
plt.title("Tiempo total vs procesos para distintos tamaños de datos")
plt.legend()
plt.grid(True)

plt.savefig("tiempo_total_vs_p.png")   # <---- Guarda imagen
plt.close()


# --------- 2. TIEMPO DE CÓMPUTO VS P ---------
plt.figure()

for n in sorted(df["n_total"].unique()):
    sub = df[df["n_total"] == n]
    plt.plot(sub["p"], sub["time_compute"], marker="o", label=f"n={n}")

plt.xlabel("Número de procesos (p)")
plt.ylabel("Tiempo de cómputo (s)")
plt.title("Tiempo de cómputo vs procesos")
plt.legend()
plt.grid(True)

plt.savefig("tiempo_compute_vs_p.png")
plt.close()

# --------- 3. TIEMPO DE COMUNICACIÓN VS P ---------
plt.figure()

for n in sorted(df["n_total"].unique()):
    sub = df[df["n_total"] == n]
    plt.plot(sub["p"], sub["time_comm"], marker="o", label=f"n={n}")

plt.xlabel("Número de procesos (p)")
plt.ylabel("Tiempo de comunicación (s)")
plt.title("Tiempo de comunicación vs procesos")
plt.legend()
plt.grid(True)

plt.savefig("tiempo_comm_vs_p.png")
plt.close()

print("Gráficas generadas:")
print(" - tiempo_total_vs_p.png")
print(" - tiempo_compute_vs_p.png")
print(" - tiempo_comm_vs_p.png")
