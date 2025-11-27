from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import argparse
import csv
import os

# ---------- Funciones auxiliares ----------

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def local_knn_neighbors_batch(X_test, X_train_local, y_train_local, k):
    """
    Para cada punto de test, calcula las distancias sólo al chunk local de entrenamiento
    y retorna los k vecinos locales (distancia, etiqueta).
    Output: lista de largo n_test, cada elemento es una lista de k tuplas (dist, label).
    """
    neighbors_all = []

    for x in X_test:
        distances = [euclidean_distance(x, t) for t in X_train_local]
        # índices de los k vecinos locales más cercanos
        k_idx = np.argsort(distances)[:k]
        local_neighbors = [(distances[i], y_train_local[i]) for i in k_idx]
        neighbors_all.append(local_neighbors)

    return neighbors_all

# ---------- MPI init ----------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Argumentos ----------

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Número total de muestras a usar del dataset (train+test).")
    parser.add_argument("--k", type=int, default=3,
                        help="Número de vecinos K en KNN.")
    parser.add_argument("--csv", type=str, default="knn_results.csv",
                        help="Nombre del archivo CSV de salida.")
    parser.add_argument("--m", type=int, default=1,
                        help="Número de repeticiones para promediar tiempos.")
    args = parser.parse_args()
else:
    args = None

# Broadcast de los argumentos a todos
args = comm.bcast(args, root=0)

# ---------- Carga y partición de datos (solo rank 0) ----------

if rank == 0:
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Limitar número de muestras si se pidió n_samples
    if args.n_samples is not None:
        n_total = min(args.n_samples, len(X))
        X = X[:n_total]
        y = y[:n_total]
    else:
        n_total = len(X)

    # Split train / test (igual que en el secuencial)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Partir el TRAIN entre procesos (cada uno recibe n_tr/p filas)
    X_train_chunks = np.array_split(X_train, size)
    y_train_chunks = np.array_split(y_train, size)

    k = args.k
else:
    X_train_chunks = None
    y_train_chunks = None
    X_test = None
    y_test = None
    k = None
    n_total = None

# ---------- Scatter del TRAIN (cada proceso recibe su chunk local) ----------

X_train_local = comm.scatter(X_train_chunks, root=0)
y_train_local = comm.scatter(y_train_chunks, root=0)

# ---------- Broadcast del TEST y parámetros (todos ven todo el test) ----------

X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)
k = comm.bcast(k, root=0)
n_total = comm.bcast(n_total, root=0)

n_tr_local = len(X_train_local)
n_tr_global = comm.allreduce(n_tr_local, op=MPI.SUM)
n_test = len(X_test)

m = args.m  # número de repeticiones

# ---------- Acumuladores para promedios (solo los usará rank 0) ----------

total_time_sum = 0.0
compute_time_sum = 0.0
comm_time_sum = 0.0
accuracy_sum = 0.0

# ---------- Repetir experimento m veces ----------

for rep in range(m):
    # Medición de tiempos
    comm.Barrier()
    start_total = MPI.Wtime()

    # Cómputo local: distancias, sort, k-nearest
    compute_start = MPI.Wtime()
    neighbors_local = local_knn_neighbors_batch(X_test, X_train_local, y_train_local, k)
    compute_end = MPI.Wtime()

    # Comunicación: gather de vecinos locales hacia root
    comm_start = MPI.Wtime()
    neighbors_all = comm.gather(neighbors_local, root=0)
    comm_end = MPI.Wtime()

    comm.Barrier()
    end_total = MPI.Wtime()

    # Tiempos locales
    local_compute_time = compute_end - compute_start
    local_comm_time = comm_end - comm_start
    local_total_time = end_total - start_total

    # Reducir al máximo (proceso más lento) para tener tiempo paralelo representativo
    compute_time = comm.reduce(local_compute_time, op=MPI.MAX, root=0)
    comm_time = comm.reduce(local_comm_time, op=MPI.MAX, root=0)
    total_time = comm.reduce(local_total_time, op=MPI.MAX, root=0)

    # ---------- Rank 0: apply majority + accuracy y acumular ----------

    if rank == 0:
        # neighbors_all es lista de tamaño 'size'
        # neighbors_all[proc][i] -> lista de k vecinos locales del proceso 'proc' para el test i
        # Construimos los vecinos globales por cada test
        global_neighbors = [[] for _ in range(n_test)]

        for proc in range(size):
            chunk = neighbors_all[proc]  # lista de largo n_test
            for i in range(n_test):
                global_neighbors[i].extend(chunk[i])

        # Ahora cada global_neighbors[i] tiene p*k vecinos (dist,label)
        y_pred = []
        for neighs in global_neighbors:
            neighs_sorted = sorted(neighs, key=lambda x: x[0])
            k_best = neighs_sorted[:k]
            labels = [lab for (_, lab) in k_best]
            pred = Counter(labels).most_common(1)[0][0]
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        accuracy = float(np.mean(y_pred == y_test))

        # Acumular para promedio
        total_time_sum += total_time
        compute_time_sum += compute_time
        comm_time_sum += comm_time
        accuracy_sum += accuracy

# ---------- Rank 0: promedios y CSV ----------

if rank == 0:
    avg_total_time = total_time_sum / m
    avg_compute_time = compute_time_sum / m
    avg_comm_time = comm_time_sum / m
    avg_accuracy = accuracy_sum / m

    print(f"(Promedio de {m} corridas)")
    print(f"p={size}, n_total={n_total}, n_tr={n_tr_global}, n_test={n_test}")
    print(f"Accuracy promedio: {avg_accuracy:.4f}")
    print(f"Total time prom:   {avg_total_time:.6f} sec")
    print(f"Compute time prom: {avg_compute_time:.6f} sec")
    print(f"Comm time prom:    {avg_comm_time:.6f} sec")

    # Guardar en CSV
    csv_file = args.csv
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "p", "n_total", "n_tr", "n_test",
                "accuracy", "time_total", "time_compute", "time_comm", "m"
            ])
        writer.writerow([
            size, n_total, n_tr_global, n_test,
            avg_accuracy, avg_total_time, avg_compute_time, avg_comm_time, m
        ])
