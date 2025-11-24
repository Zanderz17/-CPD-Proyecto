from mpi4py import MPI
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import argparse
import csv
import os

# ---------- Funciones auxiliares ----------

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def knn_predict_batch(X_test_chunk, X_train, y_train, k):
    preds = []
    for x in X_test_chunk:
        preds.append(knn_predict(x, X_train, y_train, k))
    return np.array(preds)

# ---------- MPI init ----------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Argumentos ----------

if rank == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Número total de muestras a usar del dataset MNIST (train+test). Máx ~70000.")
    parser.add_argument("--k", type=int, default=3,
                        help="Número de vecinos K en KNN.")
    parser.add_argument("--csv", type=str, default="knn_mnist_results.csv",
                        help="Nombre del archivo CSV de salida.")
    args = parser.parse_args()
else:
    args = None

# Broadcast de los argumentos a todos
args = comm.bcast(args, root=0)

# ---------- Carga de datos (solo rank 0) ----------

if rank == 0:
    print("Rank 0: cargando MNIST (fetch_openml)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)

    # Limitar número de muestras si se pidió n_samples
    if args.n_samples is not None:
        n_total = min(args.n_samples, X.shape[0])
        X = X[:n_total]
        y = y[:n_total]
    else:
        n_total = X.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    k = args.k

    # Partir los datos de test entre los procesos
    X_test_chunks = np.array_split(X_test, size)
    y_test_chunks = np.array_split(y_test, size)
else:
    X_train = None
    y_train = None
    X_test_chunks = None
    y_test_chunks = None
    k = None
    n_total = None
    X_test = None
    y_test = None

# ---------- Broadcast de train y parámetros ----------

X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)
n_total = comm.bcast(n_total, root=0)

# ---------- Scatter de test ----------

X_test_local = comm.scatter(X_test_chunks, root=0)
y_test_local = comm.scatter(y_test_chunks, root=0)

# Para info final (solo root necesita el tamaño de test global)
if rank == 0:
    n_test = len(X_test)
else:
    n_test = None

n_test = comm.bcast(n_test, root=0)

# ---------- Medición de tiempos ----------

comm.Barrier()
start_total = MPI.Wtime()

# Comunicación 1 (ya ocurrió bcast+scatter; aquí solo dejamos estructura)
comm1_start = MPI.Wtime()
comm1_end = MPI.Wtime()

# Cómputo
compute_start = MPI.Wtime()
y_pred_local = knn_predict_batch(X_test_local, X_train, y_train, k)
compute_end = MPI.Wtime()

# Comunicación 2: gather
comm2_start = MPI.Wtime()
y_pred_all = comm.gather(y_pred_local, root=0)
y_test_all = comm.gather(y_test_local, root=0)
comm2_end = MPI.Wtime()

comm.Barrier()
end_total = MPI.Wtime()

# Tiempos locales
local_compute_time = compute_end - compute_start
local_comm_time = (comm1_end - comm1_start) + (comm2_end - comm2_start)
local_total_time = end_total - start_total

# Reducir al máximo (el proceso más lento)
compute_time = comm.reduce(local_compute_time, op=MPI.MAX, root=0)
comm_time = comm.reduce(local_comm_time, op=MPI.MAX, root=0)
total_time = comm.reduce(local_total_time, op=MPI.MAX, root=0)

# ---------- Rank 0: accuracy y escribir CSV ----------

if rank == 0:
    y_pred_all = np.concatenate(y_pred_all)
    y_test_all = np.concatenate(y_test_all)

    accuracy = float(np.mean(y_pred_all == y_test_all))

    print(f"p={size}, n_total={n_total}, n_test={n_test}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total time:      {total_time:.6f} sec")
    print(f"Compute time:    {compute_time:.6f} sec")
    print(f"Comm time:       {comm_time:.6f} sec")

    csv_file = args.csv
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "p", "n_total", "n_test",
                "accuracy", "time_total", "time_compute", "time_comm"
            ])
        writer.writerow([
            size, n_total, n_test,
            accuracy, total_time, compute_time, comm_time
        ])
