import numpy as np
import pandas as pd
import time
import multiprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor

class ParallelDecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=10, min_samples_split=2, n_jobs=-1):
        """
        Parallel Decision Tree Classifier
        
        Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum samples required to split an internal node
        - n_jobs: Number of parallel processes (-1 uses all available cores)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.tree = None

    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _find_best_split(self, X, y):
        """
        Find the best split for the entire feature space
        
        Returns:
        Best feature index, best threshold, and resulting impurity
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            # Sort unique values to reduce computational complexity
            unique_values = np.unique(X[:, feature])
            
            for threshold in unique_values:
                # Binary split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if split creates too small subsets
                if (np.sum(left_mask) < self.min_samples_split or 
                    np.sum(right_mask) < self.min_samples_split):
                    continue

                # Calculate weighted Gini impurity
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                weighted_gini = (
                    (np.sum(left_mask) * left_gini + 
                     np.sum(right_mask) * right_gini) / len(y)
                )

                # Update best split if improvement found
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build decision tree
        """
        # Stopping conditions
        if (depth == self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) < self.min_samples_split):
            # Return most frequent class
            return {'class': np.bincount(y).argmax()}

        # Find best split
        best_feature, best_threshold, _ = self._find_best_split(X, y)

        # If no good split found
        if best_feature is None:
            return {'class': np.bincount(y).argmax()}

        # Create node and recursively build subtrees
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree"""
        # Ensure input is numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        # Build the tree
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        """Make predictions for input data"""
        # Ensure input is numpy array
        X = np.asarray(X)

        # Predict for each sample
        def _predict_single(sample):
            node = self.tree
            while 'class' not in node:
                if sample[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            return node['class']
        
        return np.array([_predict_single(sample) for sample in X])

def parallel_performance_benchmark():
    """
    Benchmark decision tree performance across different configurations
    """
    # Performance measurement configuration
    dataset_sizes = [1000, 5000, 10000, 25000]
    thread_counts = [1, 2, 4, 8]
    
    # Results storage
    results = []

    # Benchmark for each dataset size and thread count
    for n_samples in dataset_sizes:
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=20, 
            n_informative=10, 
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Parallel processing experiments
        for n_jobs in thread_counts:
            # Multiple run averaging
            training_times = []
            accuracies = []

            for _ in range(3):  # 3 runs for statistical stability
                # Time the entire process
                start_time = time.time()
                
                # Create and train tree
                tree = ParallelDecisionTree(
                    max_depth=10, 
                    min_samples_split=2, 
                    n_jobs=n_jobs
                )
                tree.fit(X_train, y_train)
                
                # Predict and measure performance
                y_pred = tree.predict(X_test)
                
                # Record metrics
                training_time = time.time() - start_time
                training_times.append(training_time)
                accuracies.append(accuracy_score(y_test, y_pred))

            # Store results
            results.append({
                'Dataset Size': n_samples,
                'Parallel Processes': n_jobs,
                'Avg Training Time (s)': np.mean(training_times),
                'Std Training Time (s)': np.std(training_times),
                'Avg Accuracy (%)': np.mean(accuracies) * 100
            })

    # Convert to DataFrame for easy visualization
    results_df = pd.DataFrame(results)
    print(results_df)
    
    return results_df

# Main execution
if __name__ == "__main__":
    # Run performance benchmark
    performance_results = parallel_performance_benchmark()