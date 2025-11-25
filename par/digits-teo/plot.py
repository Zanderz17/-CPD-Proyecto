import numpy as np
import matplotlib.pyplot as plt

lam = 8.9e-5
mu  = 4.9e-3

# modelo
def T_norm(p):
    return 1/p + lam*p + mu*np.log2(p)

# rango extendido de procesos
P = np.arange(1, 1800)

# evaluar
Tvals = T_norm(P)

# encontrar mínimo
p_opt = P[np.argmin(Tvals)]
T_opt = np.min(Tvals)

print("P óptimo teórico:", p_opt)
print("T_norm mínimo:", T_opt)

# graficar
plt.figure(figsize=(10,6))
plt.plot(P, Tvals)
plt.axvline(p_opt, color='red', linestyle='--', label=f"p óptimo = {p_opt}")
plt.xlabel("Número de procesos p")
plt.ylabel("T_norm(p)")
plt.title("Modelo teórico normalizado (búsqueda de p óptimo)")
plt.grid(True)
plt.legend()

plt.savefig("p_optimo_teorico.png", dpi=300)
print("Gráfico guardado como p_optimo_teorico.png")
