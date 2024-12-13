import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.preprocessing import StandardScaler

def generate_dataset(n_samples, n_features):
    return np.random.rand(n_samples, n_features)

def compute_covariance(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

def eigen_decomposition(cov_matrix):
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

def time_pca(num_threads, X_std, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        cov_matrix = compute_covariance(X_std)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            eig_vals, eig_vecs = executor.submit(eigen_decomposition, cov_matrix).result()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_pca_benchmark(dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"PCA: Processing dataset with {n_samples} samples and {n_features} features...")
        X = generate_dataset(n_samples, n_features)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        thread_execution_times = [
            (num_threads, time_pca(num_threads, X_std))
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
