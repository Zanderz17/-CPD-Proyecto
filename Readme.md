# KNN paralelo 


## 📌 Resumen ejecutivo
Este informe resume la paralelización del algoritmo KNN sobre los datasets digits y MNIST, la metodología experimental, la derivación teórica de FLOPs, la normalización con datos experimentales, métricas (tiempos totales, cómputo, comunicación), speedup, FLOPs/s y conclusiones.  

**Datos experimentales empleados:** `par/digits/knn_results.csv` (experimentos para N = 200, 500, 1000, 1500, 1797; p variando hasta 80).  

**Conclusión principal:** la operación de cálculo de distancias es altamente paralelizable. Sin embargo, el rendimiento real se ve limitado por la comunicación y el overhead de serialización en mpi4py/Python. Para N grande (ej. N=1797) el tiempo mínimo observado se alcanza alrededor de **p=32**; la eficiencia disminuye al aumentar p debido a sobrecostes de comunicación.

---

## 📑 Índice  
1. Objetivo y alcance  
2. Implementación y decisiones de paralelización  
3. Conteo de FLOPs y modelo teórico  
4. Protocolo experimental y métricas  
5. Resultados experimentales  
6. Speedup, eficiencia y p óptimo  
7. FLOPs/s y discusión  
8. Comparación con Amdahl  
9. Conclusiones y recomendaciones  
10. Anexos: comandos para reproducir  

---

## 1. 🧠 Objetivo y alcance
- **Objetivo:** paralelizar KNN (clasificación supervisada) con mpi4py; medir tiempos (total, cómputo, comunicación), comparar con la versión secuencial y validar con un modelo teórico basado en FLOPs.
- **Alcance:** código del repositorio (`sec/`, `par/`) y resultados guardados en CSV/PNG en `par/`.

---

## 2. 🧩 Implementación y decisiones de paralelización

**Scripts principales:**
| Tipo | Script |
|------|-------|
| Secuencial | `sec/knn_digits_sec.py` |
| MPI (Digits) | `par/digits/knn_digits_mpi.py` / `knn_digits_mpi_timed.py` |

**Estrategias MPI probadas:**
| Variante | Descripción | Uso |
|----------|-------------|-----|
| 1 | `broadcast(train)` + `scatter(test)` | Más usada — cada proceso tiene TODO el train y su porción del test |
| 2 | `scatter(train)` + `broadcast(test)` | Útil para combinar vecinos locales y luego hacer gather |

**Primitivas:** `comm.bcast`, `comm.scatter`, `comm.gather`, `MPI.Barrier`, `MPI.Wtime`.  
**Medición:** tiempos separados por región usando `MPI.Wtime()`.

---

## 3. 📏 Conteo de FLOPs y modelo teórico

### FLOPs por distancia euclidiana (dimensión d):
FLOP_pair ≈ 3 · d (restas + multiplicaciones + sumas + sqrt)

shell
Copiar código

### Total para la parte paralelizable:
FLOP_total ≈ 3 · d · n_train · n_test
con: n_train ≈ 0.8·N , n_test ≈ 0.2·N

shell
Copiar código

### Modelo de tiempo:
compute_time_teo(p) = FLOP_total / (p · peak_flops_por_nodo)
comm_time_teo(p) ≈ α(p−1) + β · message_size(p)

markdown
Copiar código

**Peak estimado empíricamente desde p=1:**
peak ≈ FLOP_total / compute_time(p=1)

yaml
Copiar código

---

## 4. 🧪 Protocolo experimental y métricas

- **Repeticiones:** `m=7` por punto en CSV.  
- **Semilla fija** para reproducibilidad.  
- **Barridos realizados:**
  - **Strong scaling:** N fijo, p ∈ {1,2,4,8,16,32,64,80}
  - **Weak scaling:** distintos N (scripts incluidos)

**Métricas registradas:**
| p | n_total | time_total | time_compute | time_comm | accuracy |

---

## 5. 📊 Resultados experimentales (resumen)

### 🔹 N = 200 → FLOP_total = 1,228,800  
| p | total | compute | comm | accuracy |
|---|-------|---------|------|---------|
| 1 | 0.04579 | 0.04519 | 0.00060 | 0.95 |
| 16 | 0.00616 | 0.00434 | 0.00201 | 0.95 |

📌 **p óptimo ~16**

---

### 🔹 N = 1000 → FLOP_total = 30,720,000  
| p | total | compute | comm | accuracy |
|---|-------|---------|------|---------|
| 1 | 1.11606 | 1.11329 | 0.00276 | 0.995 |
| 16 | 0.12011 | 0.10794 | 0.02687 | 0.995 |

---

### 🔹 N = 1797 → FLOP_total ≈ 99M  
| p | total | compute | comm | accuracy |
|---|-------|---------|------|---------|
| 1 | 3.92186 | 3.91640 | 0.00545 | ~0.983 |
| 32 | 0.28376 | 0.23172 | 0.11069 | ~0.983 |

📌 **Tiempo mínimo ≈ p=32, eficiencia mayor en p=16**

---

## 6. ⚙️ Speedup, eficiencia y p óptimo

- **Speedup:** `S(p) = T1 / Tp`  
- **Eficiencia:** `E(p) = S(p) / p`  

### Tendencias:
- N pequeño → el speedup cae rápido por comunicación.  
- N grande → p óptimo se desplaza hacia valores altos (más cómputo para paralelizar).

| N | p óptimo (tiempo) | p óptimo (eficiencia) |
|---|-------------------|------------------------|
| 200 | 16 | 8 |
| 1000 | 16 | 8 |
| 1797 | **32** | **16** |

---

## 7. ⚡ FLOPs/s y rendimiento

FLOPs/s ≈ FLOP_total / time_compute

- Valores modestos: ~25–50 MFLOP/s (p=1)
- No representan el peak del hardware: Python + mpi4py añaden mucho overhead.
- Útiles solo para comparar implementaciones, NO para medir hardware real.

---

## 8. 📉 Comparación con Amdahl

- fracción paralelizable f ≈ 0.99 (para N grande).  
- Amdahl predice más speedup del observado.
- **Causas de discrepancia:**
  - Serialización (pickling) en mpi4py.
  - Overhead de Python.
  - `np.array_split` produce desbalance.
  - Contención de caché/memoria.

---

## 9. Conclusiones y recomendaciones

### ✔ Conclusiones
- KNN **sí escala**, pero solo hasta cierto p.
- Para N=1797, **p≈32 minimiza tiempo total**, pero eficiencia cae.
- El cuello principal es **comunicación + serialización**.
