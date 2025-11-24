#!/bin/bash

module load gnu12/12.4.0
module load openmpi4/4.1.6
module load py3-mpi4py/3.1.3

CSV_NAME="knn_mnist_results.csv"
rm -f "$CSV_NAME"

# Tamaños de datos reales (subconjuntos de MNIST, máx 70000)
N_VALUES=(5000 10000 20000 40000 60000)
# Procesos (puedes ajustar según lo que soporte tu entorno)
P_VALUES=(1 2 4 8 16 32 64)

for n in "${N_VALUES[@]}"; do
  for p in "${P_VALUES[@]}"; do
    echo ">>> Ejecutando con p=$p, n_total=$n"
    mpiexec --oversubscribe -n "$p" python3 knn_mnist_mpi_timed.py \
      --n_samples "$n" \
      --k 3 \
      --csv "$CSV_NAME"
  done
done

echo "Experimentos terminados. Resultados en $CSV_NAME"
