import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, FashionMNIST, CIFAR10
import numpy as np
import random
import time
import os
import csv

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def partition_data_iid(dataset, num_clients):
    data_size = len(dataset)
    shard_size = data_size // num_clients
    all_indices = list(range(data_size))
    random.shuffle(all_indices)
    client_indices = [all_indices[i*shard_size:(i+1)*shard_size] for i in range(num_clients)]
    return client_indices


def partition_data_quantity_skew_iid(dataset, num_clients, alpha=0.5):
    # Dirichlet for quantity skew but iid distribution
    data_size = len(dataset)
    proportions = np.random.dirichlet([alpha] * num_clients)
    proportions = (proportions / proportions.sum())
    sizes = (proportions * data_size).astype(int)
    # adjust rounding
    sizes[-1] = data_size - sizes[:-1].sum()
    all_indices = list(range(data_size))
    random.shuffle(all_indices)
    client_indices = []
    idx = 0
    for sz in sizes:
        client_indices.append(all_indices[idx:idx+sz])
        idx += sz
    return client_indices


def partition_data_class_noniid(dataset, num_clients, classes_per_client=2):
    # each client gets a few classes, equal quantity
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    class_indices = {i: np.where(targets == i)[0].tolist() for i in range(num_classes)}
    client_indices = []
    for i in range(num_clients):
        chosen = np.random.choice(range(num_classes), classes_per_client, replace=False)
        inds = []
        per_class = len(dataset) // (num_clients * classes_per_client)
        for c in chosen:
            chosen_inds = random.sample(class_indices[c], per_class)
            inds.extend(chosen_inds)
        client_indices.append(inds)
    return client_indices

# -----------------------
# Client Simulation
# -----------------------
class Client:
    def __init__(self, cid, dataset, indices, model_fn, device, compute_speed, comm_speed):
        self.cid = cid
        self.device = device
        self.compute_speed = compute_speed  # e.g., samples/sec
        self.comm_speed = comm_speed      # e.g., bytes/sec
        self.model_fn = model_fn
        subset = Subset(dataset, indices)
        self.loader = DataLoader(subset, batch_size=32, shuffle=True)

    def train(self, global_params, epochs=1, lr=0.01):
        # load global model
        model = self.model_fn()
        model.load_state_dict(global_params)
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # simulate compute time
        num_samples = len(self.loader.dataset)
        compute_time = num_samples / self.compute_speed

        # actual local training (fast)
        for _ in range(epochs):
            for data, target in self.loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # simulate communication time (model size ~ params * 4 bytes)
        param_size = sum(p.numel() for p in model.parameters()) * 4
        comm_time = param_size / self.comm_speed

        # return updated weights and time cost
        return model.state_dict(), compute_time + comm_time

# -----------------------
# Federated Simulation
# -----------------------
class FederatedSimulation:
    def __init__(self, dataset_name, dataset, model_fn, device):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.model_fn = model_fn
        self.device = device

    def run(self, client_indices, selection_probs, num_rounds=50, epochs=1):
        # initialize global model
        global_model = self.model_fn().to(self.device)
        global_params = global_model.state_dict()

        clients = []
        for cid, inds in enumerate(client_indices):
            # assign random speeds
            compute_speed = random.uniform(50, 200)  # samples/sec
            comm_speed = random.uniform(1e5, 5e5)   # bytes/sec
            clients.append(Client(cid, self.dataset, inds,
                                   lambda: self.model_fn(), self.device,
                                   compute_speed, comm_speed))

        history = {'round': [], 'time': [], 'loss': []}
        total_time = 0.0

        for r in range(1, num_rounds+1):
            # select clients
            selected_ids = np.random.choice(len(clients), size=int(len(clients)*0.1),
                                            replace=False, p=selection_probs)
            updates = []
            times = []
            # local training
            for cid in selected_ids:
                weights, tcost = clients[cid].train(global_params, epochs)
                updates.append(weights)
                times.append(tcost)
            # aggregate
            new_params = {}
            for k in global_params.keys():
                new_params[k] = sum([u[k] for u in updates]) / len(updates)
            global_params = new_params

            # evaluate global loss
            global_model.load_state_dict(global_params)
            global_model.eval()
            loader = DataLoader(self.dataset, batch_size=128)
            criterion = nn.CrossEntropyLoss()
            loss = 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = global_model(data)
                    loss += criterion(output, target).item() * data.size(0)
            loss /= len(self.dataset)

            # record
            round_time = sum(times)
            total_time += round_time
            history['round'].append(r)
            history['time'].append(total_time)
            history['loss'].append(loss)

            print(f"Dataset {self.dataset_name}, Round {r}, Time {total_time:.2f}s, Loss {loss:.4f}")

        return history

# -----------------------
# Experiment Driver
# -----------------------
def main(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        'EMNIST': EMNIST('../data/emnist', split='balanced', download=True,
                         transform=transforms.ToTensor()),
        'FashionMNIST': FashionMNIST('../data', download=True,
                                     transform=transforms.ToTensor()),
        'CIFAR10': CIFAR10('../data', download=True,
                           transform=transforms.ToTensor())
    }

    client_counts = [10, 20, 50]
    skew_types = ['iid', 'quantity_skew', 'class_noniid']

    for name, dataset in datasets.items():
        for n_clients in client_counts:
            # partition data
            idx_iid = partition_data_iid(dataset, n_clients)
            idx_qty = partition_data_quantity_skew_iid(dataset, n_clients)
            idx_cls = partition_data_class_noniid(dataset, n_clients)
            partitions = {'iid': idx_iid, 'quantity_skew': idx_qty, 'class_noniid': idx_cls}

            for skew in skew_types:
                indices = partitions[skew]
                # base: equal prob
                probs_equal = np.ones(n_clients) / n_clients
                # weighted by capability: assign after init
                # here simulate weights by random speeds
                speeds = np.random.rand(n_clients)
                probs_speed = speeds / speeds.sum()
                # weighted by data quantity
                qty = np.array([len(idx) for idx in indices], dtype=float)
                probs_qty = qty / qty.sum()

                sim = FederatedSimulation(name, dataset, 
                                          lambda: SimpleCNN(dataset[0][0].shape[0],
                                                            len(dataset.classes) if hasattr(dataset, 'classes') else 10),
                                          device)

                experiments = {
                    'equal': probs_equal,
                    'speed_weighted': probs_speed,
                    'quantity_weighted': probs_qty
                }

                for exp_name, probs in experiments.items():
                    history = sim.run(indices, probs, num_rounds=50)
                    # save
                    fname = f"{name}_{n_clients}_{skew}_{exp_name}.csv"
                    with open(os.path.join(results_dir, fname), 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['round', 'time', 'loss'])
                        for r, t, l in zip(history['round'], history['time'], history['loss']):
                            writer.writerow([r, t, l])

if __name__ == "__main__":
    main()
