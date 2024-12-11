import numpy as np
import pandas as pd
import time
import multiprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor

def generate_dataset(n_samples, n_features):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=int(n_features * 0.7), 
        n_redundant=int(n_features * 0.2), 
        random_state=42
    )
    return X, y

def train_random_forest(X_train, y_train, n_estimators, random_state):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def parallel_random_forest(X_train, y_train, n_estimators, num_threads):
    estimators_per_thread = n_estimators // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(train_random_forest, X_train, y_train, estimators_per_thread, i) 
                   for i in range(num_threads)]
        sub_forests = [future.result() for future in futures]
    
    # Combine sub-forests
    combined_forest = RandomForestClassifier(n_estimators=n_estimators)
    combined_forest.estimators_ = [tree for forest in sub_forests for tree in forest.estimators_]
    return combined_forest

def time_random_forest(num_threads, X_train, y_train, X_test, y_test, n_estimators=100, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        rf = parallel_random_forest(X_train, y_train, n_estimators, num_threads)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_random_forest_benchmark(dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"Random Forest: Processing dataset with {n_samples} samples and {n_features} features...")
        X, y = generate_dataset(n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        thread_execution_times = [
            (num_threads, time_random_forest(num_threads, X_train, y_train, X_test, y_test))
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
