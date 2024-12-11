import argparse
import matplotlib.pyplot as plt
from pca_benchmark import run_pca_benchmark
from random_forest_benchmark import run_random_forest_benchmark
from neural_network_benchmark import run_neural_network_benchmark
from knn_benchmark import run_knn_benchmark

def plot_results(results, title):
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

def parse_args():
    parser = argparse.ArgumentParser(description="ML Parallelization Benchmark")
    parser.add_argument("--algorithms", nargs="+", default=["pca", "rf", "nn", "knn"],
                        choices=["pca", "rf", "nn", "knn"],
                        help="Algorithms to benchmark (pca, rf, nn, knn)")
    parser.add_argument("--dataset_configs", nargs="+", type=lambda x: tuple(map(int, x.split('x'))),
                        default=[(1000, 10), (1000, 100), (10000, 10), (10000, 100)],
                        help="Dataset configurations in the format 'samples x features' (e.g., 1000x10)")
    parser.add_argument("--thread_counts", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="Thread counts to benchmark")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Run benchmarks
    for algo in args.algorithms:
        if algo == "pca":
            results = run_pca_benchmark(args.dataset_configs, args.thread_counts)
            plot_results(results, "PCA")
        elif algo == "rf":
            results = run_random_forest_benchmark(args.dataset_configs, args.thread_counts)
            plot_results(results, "Random Forest")
        elif algo == "nn":
            results = run_neural_network_benchmark(args.dataset_configs, args.thread_counts)
            plot_results(results, "Neural Network")
        elif algo == "knn":
            results = run_knn_benchmark(args.dataset_configs, args.thread_counts)
            plot_results(results, "K-Nearest Neighbors")

    print("Benchmarking complete. Results saved as PNG files.")
