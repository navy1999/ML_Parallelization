import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.flattened_size = 64 * input_size * input_size
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_dataset(num_samples, num_features):
    X = torch.randn(num_samples, 1, num_features, num_features)
    y = torch.randint(0, 10, (num_samples,))
    return X, y

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def time_neural_network(num_threads, X, y, hidden_size=128, num_epochs=2, batch_size=32, num_runs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[2]
    output_size = 10
    
    total_time = 0
    for _ in range(num_runs):
        model = SimpleNN(input_size, hidden_size, output_size)
        model = nn.DataParallel(model, device_ids=list(range(num_threads)))
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start_time = time.perf_counter()
        for epoch in range(num_epochs):
            train_model(model, dataloader, criterion, optimizer, device)
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
