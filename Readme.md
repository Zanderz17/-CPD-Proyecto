#  NN Paralelo con MPI 

Este proyecto implementa y evalúa una versión paralela del algoritmo
*k*-Nearest Neighbors utilizando `mpi4py` en Python. El trabajo forma parte del
curso Computación Paralela y Distribuida (UTEC).

📄 Informe completo → `Informe.pdf`  
📊 Resultados experimentales → carpeta `par/digits/`

---

## 🎯 Objetivo

Acelerar el proceso de clasificación mediante paralelización del cálculo de
distancias entre muestras. Se mide:

- Tiempo total, cómputo y comunicación
- Speedup, eficiencia y FLOPs/s
- Validación del modelo con un ajuste teórico basado en el DAG del algoritmo
- Precisión del modelo (accuracy) con distintos valores de `p` y `N`

---

## 🧠 Dataset utilizado

- Dataset: `digits` — scikit-learn
- Tamaños probados: `N = 200, 500, 1000, 1500, 1797`
- Dimensión de cada imagen: `d = 64`
- Vecinos: `k = 3`

> Se hicieron pruebas preliminares con `MNIST`, pero la mejora no se mantuvo en alto número de procesos, por lo que el análisis final se centró en `digits`.

---

## ⚙️ Ejecución

### ▶️ Versión secuencial
```bash
python3 sec/knn_digits_sec.py
```

### 🚀 Versión paralela (MPI)

```bash
mpiexec --oversubscribe -n 32 \
    python3 par/digits/knn_digits_mpi_timed.py \
    --n_samples 1797 --k 3 --m 7 --csv knn_results.csv
```

### Parámetros

| Flag        | Descripción |
|------------|-------------|
| `--n_samples` | tamaño del dataset |
| `--k`         | número de vecinos |
| `--m`         | repeticiones para promediar tiempos |
| `--csv`       | guarda resultados |

---

### 📌 Conclusiones principales

- La operación dominante del algoritmo (**cálculo de distancias**) es altamente paralelizable.
- Para **N = 1797**, el tiempo mínimo se logra con **p ≈ 32 procesos**.
- El **accuracy** del modelo no se ve afectado por la paralelización.
- La eficiencia disminuye con valores altos de `p` debido a:
  - overhead de comunicación MPI,
  - sobredistribución del trabajo (muy pocos datos por proceso).
