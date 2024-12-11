import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_dataset(n_samples, n_features):
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))
    return X, y

def time_neural_network(num_threads, X, y, num_runs=5):
    torch.set_num_threads(num_threads)
    total_time = 0
    for _ in range(num_runs):
        model = SimpleNN(X.shape[1], 64, 2)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        start_time = time.perf_counter()
        for _ in range(10):  # 10 epochs
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

def run_neural_network_benchmark(dataset_configs, thread_counts):
    results = {}
    for n_samples, n_features in dataset_configs:
        print(f"Neural Network: Processing dataset with {n_samples} samples and {n_features} features...")
        X, y = generate_dataset(n_samples, n_features)

        thread_execution_times = [
            (num_threads, time_neural_network(num_threads, X, y))
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
