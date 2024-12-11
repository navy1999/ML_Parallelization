import numpy as np
from sklearn.datasets import make_classification
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
import time

def knn_predict(X_train, y_train, X_test, k):
    distances = cdist(X_test, X_train, metric='euclidean')
    neighbors = np.argsort(distances, axis=1)[:, :k]
    predictions = [np.bincount(y_train[neighbor]).argmax() for neighbor in neighbors]
    return np.array(predictions)

def parallel_knn(X_train, y_train, X_test, k, num_threads):
    def process_chunk(chunk):
        return knn_predict(X_train, y_train, chunk, k)

    chunk_size = len(X_test) // num_threads
    chunks = [X_test[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_chunk, chunks))

    return np.concatenate(results)

def generate_dataset(n_samples, n_features):
    return make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features * 0.8), random_state=42)

def time_knn(num_threads, X_train, y_train, X_test, k, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        parallel_knn(X_train, y_train, X_test, k, num_threads)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_knn_benchmark(dataset_configs, thread_counts):
    results = {}
    k = 5
    for n_samples, n_features in dataset_configs:
        print(f"KNN: Processing dataset with {n_samples} samples and {n_features} features...")
        X, y = generate_dataset(n_samples, n_features)
        X_train, X_test = X[:int(0.8*n_samples)], X[int(0.8*n_samples):]
        y_train = y[:int(0.8*n_samples)]

        thread_execution_times = [
            (num_threads, time_knn(num_threads, X_train, y_train, X_test, k))
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
