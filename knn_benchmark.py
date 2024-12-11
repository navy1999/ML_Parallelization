import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import time

def generate_dataset(n_samples, n_features):
    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.8), random_state=42)

def time_knn(num_threads, X, y, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=num_threads)
        knn.fit(X, y)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_knn_benchmark(dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"KNN: Processing dataset with {n_samples} samples and {n_features} features...")
        X, y = generate_dataset(n_samples, n_features)

        thread_execution_times = [
            (num_threads, time_knn(num_threads, X, y))
            for num_threads in thread_counts
        ]

        single_thread_time = thread_execution_times[0][1]
        speedup = [
            (num_threads, single_thread_time / exec_time if exec_time > 0 else 0)
            for num_threads, exec_time in thread_execution_times
        ]

        results[(n_samples, n_features)] = {
            "execution_times": thread_execution_times,
            "speedups": speedup,
        }
    return results
