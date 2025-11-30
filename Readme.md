#  KNN Paralelo con MPI 

Este proyecto implementa y evalúa una versión paralela del algoritmo
*k*-Nearest Neighbors utilizando `mpi4py` en Python. El trabajo forma parte del
curso Computación Paralela y Distribuida (UTEC).

📄 Informe completo → `Informe.pdf`  
📊 Resultados experimentales → carpeta `par/digits/`

---

## Objetivo

Acelerar el proceso de clasificación mediante paralelización del cálculo de
distancias entre muestras. Se mide:

- Tiempo total, cómputo y comunicación
- Speedup, eficiencia y FLOPs/s
- Validación del modelo con un ajuste teórico basado en el DAG del algoritmo
- Precisión del modelo (accuracy) con distintos valores de `p` y `N`

---

## Dataset utilizado

- Dataset: `digits` — scikit-learn
- Tamaños probados: `N = 200, 500, 1000, 1500, 1797`
- Dimensión de cada imagen: `d = 64`
- Vecinos: `k = 3`

> Se hicieron pruebas preliminares con `MNIST`, pero la mejora no se mantuvo en alto número de procesos, por lo que el análisis final se centró en `digits`.

---

## Ejecución

### Versión secuencial
```bash
python3 sec/knn_digits_sec.py
```

### Versión paralela (MPI)

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

### Conclusiones principales

- El cálculo de distancias de KNN es altamente paralelizable y se logró una mejora significativa en rendimiento para tamaños de datos medianos y grandes.
- Para el dataset completo `digits` (N = 1797), el mejor tiempo de ejecución se obtuvo alrededor de **p ≈ 32 procesos**, con speedup cercano a 12x y buena eficiencia.
- El **accuracy** se mantuvo constante para todos los valores de `p`, lo que demuestra que la paralelización no altera el resultado del clasificador respecto a la versión secuencial.
- La eficiencia y el rendimiento disminuyen a partir de **p ≥ 64**, debido al incremento del costo de comunicación y la menor carga de trabajo por proceso.
- Para tamaños pequeños (N = 200, 500), la paralelización **no es recomendable**, ya que la sobrecarga supera al cómputo útil.
- El algoritmo muestra **buena escalabilidad** siempre que el tamaño del problema crezca proporcionalmente al número de procesos (aprox. N = Θ(p)).

