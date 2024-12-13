import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.preprocessing import StandardScaler

def compute_partial_covariance(X_chunk):
    return np.dot(X_chunk.T, X_chunk)

def parallel_covariance(X, num_threads):
    chunk_size = X.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(compute_partial_covariance, 
                                 X[i:i+chunk_size]) 
                  for i in range(0, X.shape[0], chunk_size)]
        return sum(future.result() for future in futures) / (X.shape[0] - 1)

def eigen_decomposition_parallel(cov_matrix, num_threads):
    chunk_size = cov_matrix.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eig_vals)[::-1]
        return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

def time_pca(num_threads, X_std, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        # Parallelize covariance computation
        cov_matrix = parallel_covariance(X_std, num_threads)
        # Parallelize eigendecomposition
        eig_vals, eig_vecs = eigen_decomposition_parallel(cov_matrix, num_threads)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_pca_benchmark(dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"PCA: Processing dataset with {n_samples} samples and {n_features} features...")
        X = np.random.rand(n_samples, n_features)
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
