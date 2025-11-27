import matplotlib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import time
import csv
import os

# ==== CONFIGURACIÓN ====
CSV_OUT = "knn_sequential.csv"
k = 3
test_size = 0.2
REPS = 7

problem_sizes = [200, 500, 1000, 1500, 1797]

digits = load_digits()
X = digits.data
y = digits.target

# Crear CSV con cabecera
with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_total", "n_tr", "n_test", "avg_accuracy", "time_seq"])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

# ==== Medición con promedio ====
for n_total in problem_sizes:

    n_total = min(n_total, len(X))
    X_sub = X[:n_total]
    y_sub = y[:n_total]

    times = []
    accs = []

    for r in range(REPS):
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=test_size, random_state=None
        )

        start = time.time()
        y_pred = [knn_predict(x, X_train, y_train, k) for x in X_test]
        end = time.time()

        acc = float(np.mean(y_pred == y_test))
        t = end - start

        times.append(t)
        accs.append(acc)

        print(f"[{r+1}/{REPS}] N={n_total} | t={t:.4f}s | acc={acc:.4f}")

    avg_time = np.mean(times)
    avg_acc = np.mean(accs)

    # Guardar solo los promedios
    with open(CSV_OUT, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            n_total,
            len(X_train),
            len(X_test),
            avg_acc,
            avg_time
        ])

    print(f" → PROMEDIO N={n_total}: t={avg_time:.4f}s | acc={avg_acc:.4f}\n")

print(f"\nGuardado en: {CSV_OUT}")
