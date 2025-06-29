import os
import csv
import argparse
from main import FederatedSimulation, SimpleCNN, EMNIST, FashionMNIST, CIFAR10  # assume your classes are in federated.py
import torchvision.transforms as transforms
import torch

def run_and_save(name, dataset, partitions, strategies, dynamic, results_dir):
    mode = 'dynamic' if dynamic else 'static'
    for dist_name, client_indices in partitions.items():
        for strat in strategies:
            sim = FederatedSimulation(
                name, dataset,
                lambda: SimpleCNN(dataset[0][0].shape[0], len(dataset.classes)),
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            history = sim.run(client_indices, strat, dynamic_resources=dynamic)
            fname = f"{name}_{dist_name}_{strat}_{mode}.csv"
            path = os.path.join(results_dir, fname)
            os.makedirs(results_dir, exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['round', 'time', 'loss', 'acc', 'jfi', 'auc', 'roc'])
                for r, t, l, a, j, uu, rr in zip(
                        history['round'], history['time'], history['loss'],
                        history['acc'], history['jfi'], history['auc'], history['roc']):
                    writer.writerow([r, t, l, a, j, uu, rr])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_tradeoff')
    parser.add_argument('--dynamic', action='store_true', help='vary resources each round')
    args = parser.parse_args()

    # datasets
    datasets = {
        'EMNIST': EMNIST('../data/emnist', split='balanced', download=True, transform=transforms.ToTensor()),
        'FashionMNIST': FashionMNIST('../data', download=True, transform=transforms.ToTensor()),
        'CIFAR10': CIFAR10('../data', download=True, transform=transforms.ToTensor())
    }
    strategies = ['random', 'comp_greedy', 'comm_greedy', 'rbff', 'rbcsf']

    # 1) vary number of clients, fixed 40% participation
    client_counts = [10, 20, 30, 40, 50]
    partitions_clients = {}
    for n in client_counts:
        # 40% participation ⇒ but partition_data functions don't care about that,
        # we tag this experiment by name only.
        # we just need the indices for n clients.
        all_parts = {
            f"{n}clients_iid": FederatedSimulation.partition_data_iid(datasets['EMNIST'], n),
            f"{n}clients_qty": FederatedSimulation.partition_data_quantity_skew_iid(datasets['EMNIST'], n),
            f"{n}clients_class": FederatedSimulation.partition_data_class_noniid(datasets['EMNIST'], n),
            # repeat for other datasets below before calling run_and_save
        }
        partitions_clients[n] = all_parts

    # Actually run for each dataset
    for name, ds in datasets.items():
        # build partitions for this dataset
        parts_by_n = {}
        for n in client_counts:
            parts_by_n[f"{n}cli_iid"] = FederatedSimulation.partition_data_iid(ds, n)
            parts_by_n[f"{n}cli_qty"] = FederatedSimulation.partition_data_quantity_skew_iid(ds, n)
            parts_by_n[f"{n}cli_class"] = FederatedSimulation.partition_data_class_noniid(ds, n)
        run_and_save(name, ds, parts_by_n, strategies, args.dynamic, os.path.join(args.results_dir, 'by_clients'))

    # 2) vary participation rate, fixed 50 clients
    participation_rates = list(range(10, 101, 10))
    for name, ds in datasets.items():
        parts_by_rate = {}
        n = 50
        # first get a 50-client partition once
        base_iid = FederatedSimulation.partition_data_iid(ds, n)
        base_qty = FederatedSimulation.partition_data_quantity_skew_iid(ds, n)
        base_class = FederatedSimulation.partition_data_class_noniid(ds, n)
        for p in participation_rates:
            # tag these partitions by rate
            parts_by_rate[f"{p}pct_iid"]   = base_iid
            parts_by_rate[f"{p}pct_qty"]   = base_qty
            parts_by_rate[f"{p}pct_class"] = base_class
        run_and_save(name, ds, parts_by_rate, strategies, args.dynamic, os.path.join(args.results_dir, 'by_rate'))

if __name__ == "__main__":
    main()
