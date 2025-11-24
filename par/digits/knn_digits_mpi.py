from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# ---------- Funciones auxiliares ----------

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    # Igual que en tu versión secuencial
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

def knn_predict_batch(X_test_chunk, X_train, y_train, k):
    """
    Aplica knn_predict a todas las muestras de un trozo de X_test.
    """
    preds = []
    for x in X_test_chunk:
        preds.append(knn_predict(x, X_train, y_train, k))
    return np.array(preds)

# ---------- Inicialización MPI ----------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Rank 0 carga datos y los prepara ----------

if rank == 0:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    k = 3  # número de vecinos

    # Dividir el conjunto de test en 'size' pedazos (uno por proceso)
    X_test_chunks = np.array_split(X_test, size)
    y_test_chunks = np.array_split(y_test, size)
else:
    # En los otros procesos inicializamos variables vacías;
    # se rellenarán con bcast/scatter
    X_train = None
    y_train = None
    k = None
    X_test_chunks = None
    y_test_chunks = None

# ---------- Broadcast de datos comunes ----------

# Todos los procesos necesitan tener TODO el train y el valor de k
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)

# ---------- Scatter de los datos de test ----------

# Cada proceso recibe un trozo distinto del test
X_test_local = comm.scatter(X_test_chunks, root=0)
y_test_local = comm.scatter(y_test_chunks, root=0)

# ---------- Cálculo paralelo ----------

comm.Barrier()  # sincronizar antes de medir tiempo
start_time = MPI.Wtime()

y_pred_local = knn_predict_batch(X_test_local, X_train, y_train, k)

comm.Barrier()  # sincronizar después del cálculo
end_time = MPI.Wtime()

# ---------- Recolección de resultados ----------

# Juntamos predicciones y etiquetas verdaderas en el rank 0
y_pred_all = comm.gather(y_pred_local, root=0)
y_test_all = comm.gather(y_test_local, root=0)

if rank == 0:
    # Concatenar las listas de arrays en un solo array
    y_pred_all = np.concatenate(y_pred_all)
    y_test_all = np.concatenate(y_test_all)

    accuracy = np.mean(y_pred_all == y_test_all)
    exec_time = end_time - start_time

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Execution time (parallel, p={size}): {exec_time:.4f} sec")
