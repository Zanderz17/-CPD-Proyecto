#!/bin/bash

CSV_NAME="knn_results.csv"
rm -f "$CSV_NAME"   # borrar resultados anteriores

# Número de repeticiones por cada (p, n_total)
M=7   

# Conjuntos de n (n_total) y de procesos p
N_VALUES=(200 500 1000 1500 1797)   # 1797 ~ tamaño completo de load_digits
P_VALUES=(1 2 4 8 16 32 64 80)

for n in "${N_VALUES[@]}"; do
  for p in "${P_VALUES[@]}"; do
    echo ">>> Ejecutando con p=$p, n_total=$n, m=$M"
    mpiexec --oversubscribe -n "$p" python3 knn_digits_mpi_timed.py \
      --n_samples "$n" \
      --k 3 \
      --m "$M" \
      --csv "$CSV_NAME"
  done
done

echo "Experimentos terminados. Resultados en $CSV_NAME"
