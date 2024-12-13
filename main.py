import numpy as np
import matplotlib.pyplot as plt
import argparse
from pca_benchmark import run_pca_benchmark
from random_forest_benchmark import run_random_forest_benchmark
from neural_network_benchmark import run_neural_network_benchmark
from knn_benchmark import run_knn_benchmark

def parse_args():
    parser = argparse.ArgumentParser(description="ML Parallelization Benchmark")
    parser.add_argument("--algorithms", nargs="+", default=["pca", "rf", "nn", "knn"],
                      choices=["pca", "rf", "nn", "knn"],
                      help="Algorithms to benchmark (pca, rf, nn, knn)")
    parser.add_argument("--dataset_configs", nargs="+", 
                      type=lambda x: tuple(map(int, x.split('x'))),
                      default=[(1000, 50), (5000, 500), (10000, 5000)],
                      help="Dataset configurations in format 'samples x features' (e.g., 1000x50)")
    parser.add_argument("--thread_counts", nargs="+", type=int, 
                      default=[1, 2, 4, 8, 12, 16, 24, 32, 40, 64],
                      help="Thread counts to benchmark")
    return parser.parse_args()

def plot_speedup(results, title):
    plt.figure(figsize=(12, 8))
    for (n_samples, n_features), data in results.items():
        threads, speedups = zip(*data["speedups"])
        plt.plot(threads, speedups, marker='o', label=f'{n_samples}x{n_features}')
    
    plt.title(f"{title} - Speedup vs Thread Count")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.legend(title="Dataset Size (samples x features)")
    plt.grid(True)
    plt.savefig(f"{title.lower().replace(' ', '_')}_speedup.png")
    plt.close()

def calculate_metrics(thread_execution_times):
    single_thread_time = thread_execution_times[0][1]
    metrics = []
    for num_threads, exec_time in thread_execution_times:
        speedup = single_thread_time / exec_time if exec_time > 0 else 0
        efficiency = speedup / num_threads if num_threads > 0 else 0
        scalability = speedup / np.log2(num_threads) if num_threads > 1 else 1
        parallel_overhead = (num_threads * exec_time - single_thread_time) / single_thread_time
        metrics.append({
            "num_threads": num_threads,
            "execution_time": exec_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "scalability": scalability,
            "parallel_overhead": parallel_overhead
        })
    return metrics

def plot_metrics(results, title):
    metrics = ["speedup", "efficiency", "scalability", "parallel_overhead"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title} - Performance Metrics", fontsize=16)
    
    for (n_samples, n_features), data in results.items():
        label = f'{n_samples}x{n_features}'
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            x = [d["num_threads"] for d in data]
            y = [d[metric] for d in data]
            ax.plot(x, y, marker='o', label=label)
            ax.set_title(metric.capitalize())
            ax.set_xlabel("Number of Threads")
            ax.set_ylabel(metric.capitalize())
            ax.legend(title="Dataset Size (samples x features)")
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_metrics.png")
    plt.close()

if __name__ == "__main__":
    args = parse_args()

    for algo in args.algorithms:
        if algo == "pca":
            results = run_pca_benchmark(args.dataset_configs, args.thread_counts)
            plot_speedup(results, "PCA")
            metrics_results = {k: calculate_metrics(v["execution_times"]) for k, v in results.items()}
            plot_metrics(metrics_results, "PCA")
        elif algo == "rf":
            results = run_random_forest_benchmark(args.dataset_configs, args.thread_counts)
            plot_speedup(results, "Random Forest")
            metrics_results = {k: calculate_metrics(v["execution_times"]) for k, v in results.items()}
            plot_metrics(metrics_results, "Random Forest")
        elif algo == "nn":
            results = run_neural_network_benchmark(args.dataset_configs, args.thread_counts)
            plot_speedup(results, "Neural Network")
            metrics_results = {k: calculate_metrics(v["execution_times"]) for k, v in results.items()}
            plot_metrics(metrics_results, "Neural Network")
        elif algo == "knn":
            results = run_knn_benchmark(args.dataset_configs, args.thread_counts)
            plot_speedup(results, "K-Nearest Neighbors")
            metrics_results = {k: calculate_metrics(v["execution_times"]) for k, v in results.items()}
            plot_metrics(metrics_results, "K-Nearest Neighbors")

    print("Benchmarking complete. Results saved as PNG files.")
