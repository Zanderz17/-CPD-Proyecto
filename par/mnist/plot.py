import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("knn_mnist_results.csv")

plt.figure()

for n in sorted(df["n_total"].unique()):
    sub = df[df["n_total"] == n]
    plt.plot(sub["p"], sub["time_total"], marker="o", label=f"n={n}")

    # marcar mínimo
    fila_min = sub.loc[sub["time_total"].idxmin()]
    p_opt = fila_min["p"]
    t_min = fila_min["time_total"]
    plt.scatter(p_opt, t_min, marker="x", s=80)

plt.xlabel("Número de procesos (p)")
plt.ylabel("Tiempo total (s)")
plt.title("Tiempo total vs procesos para distintos tamaños de datos (MNIST)")
plt.legend()
plt.grid(True)

plt.savefig("mnist_tiempo_total_vs_p_minimos.png")
plt.close()

print("Gráfica guardada en mnist_tiempo_total_vs_p_minimos.png")
