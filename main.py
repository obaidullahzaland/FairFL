import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, FashionMNIST, CIFAR10
import numpy as np
import random
import os
import csv
import argparse
from sklearn.metrics import roc_auc_score


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
    indices = list(range(data_size))
    random.shuffle(indices)
    return [indices[i*shard_size:(i+1)*shard_size] for i in range(num_clients)]


def partition_data_quantity_skew_iid(dataset, num_clients, alpha=0.5):
    n = len(dataset)
    props = np.random.dirichlet([alpha]*num_clients)
    # allocate counts with at least 1 each
    base = np.floor(props * (n - num_clients)).astype(int)
    sizes = base + 1
    excess = n - sizes.sum()
    for i in np.argsort(props)[-excess:]:
        sizes[i] += 1

    indices = list(range(n))
    random.shuffle(indices)
    client_indices, idx = [], 0
    for sz in sizes:
        client_indices.append(indices[idx:idx+sz])
        idx += sz
    return client_indices


def partition_data_class_noniid(dataset, num_clients, classes_per_client=2):
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_classes = [np.random.choice(num_classes, classes_per_client, replace=False).tolist()
                      for _ in range(num_clients)]
    class2clients = {c: [] for c in range(num_classes)}
    for cid, cl in enumerate(client_classes):
        for c in cl:
            class2clients[c].append(cid)
    client_indices = [[] for _ in range(num_clients)]
    for c, cids in class2clients.items():
        inds = np.where(targets==c)[0].tolist()
        random.shuffle(inds)
        k = len(cids)
        if k==0: continue
        base = len(inds)//k; rem = len(inds)%k; st=0
        for i,cid in enumerate(cids):
            size = base + (1 if i<rem else 0)
            client_indices[cid].extend(inds[st:st+size]); st+=size
    return client_indices


class Client:
    def __init__(self, cid, dataset, indices, model_fn, device, comp_speed, comm_speed):
        self.cid = cid
        self.device = device
        self.comp_speed = comp_speed
        self.comm_speed = comm_speed
        self.model_fn = model_fn
        subset = Subset(dataset, indices)
        self.loader = DataLoader(subset, batch_size=32, shuffle=True)

    def train(self, global_params, epochs=1, lr=0.01):
        model = self.model_fn()
        model.load_state_dict(global_params)
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        compute_time = len(self.loader.dataset)/self.comp_speed
        for _ in range(epochs):
            for x,y in self.loader:
                x,y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        param_size = sum(p.numel() for p in model.parameters())*4
        comm_time = param_size/self.comm_speed
        return model.state_dict(), compute_time + comm_time


class FederatedSimulation:
    def __init__(self, name, dataset, model_fn, device):
        self.name = name
        self.dataset = dataset
        self.model_fn = model_fn
        self.device = device

    def run(self, client_indices, strategy, num_rounds=50, epochs=1, dynamic_resources=False):
        num_clients = len(client_indices)
        global_model = self.model_fn().to(self.device)
        global_params = global_model.state_dict()

        reputations = np.zeros(num_clients)
        sel_counts = np.zeros(num_clients, dtype=int)
        clients = []
        for cid, inds in enumerate(client_indices):
            comp = random.uniform(50,200)
            comm = random.uniform(1e5,5e5)
            reputations[cid] = comp + comm
            clients.append(Client(cid, self.dataset, inds, lambda: self.model_fn(), self.device, comp, comm))

        history = {'round': [], 'time': [], 'loss': [], 'acc': [], 'jfi': [], 'auc': [], 'roc': []}
        total_time = 0.0

        for r in range(1, num_rounds+1):
            if dynamic_resources:
                for cid, client in enumerate(clients):
                    comp = random.uniform(50,200)
                    comm = random.uniform(1e5,5e5)
                    client.comp_speed = comp
                    client.comm_speed = comm
                    reputations[cid] = comp + comm

            n_sel = int(num_clients * 0.4)
            if strategy == 'random':
                sel = np.random.choice(num_clients, n_sel, replace=False)
            elif strategy == 'comp_greedy':
                sel = np.argsort([c.comp_speed for c in clients])[::-1][:n_sel]
            elif strategy == 'comm_greedy':
                sel = np.argsort([c.comm_speed for c in clients])[::-1][:n_sel]
            elif strategy == 'rbff':
                scores = reputations / (1 + sel_counts)
                sel = np.argsort(scores)[::-1][:n_sel]
            elif strategy == 'rbcsf':
                # Scale penalty so selection penalties are on same order as reputations
                max_rep = reputations.max()
                penalty_per_pick = max_rep / float(num_rounds)
                scores = reputations - penalty_per_pick * sel_counts
                sel = np.argsort(scores)[::-1][:n_sel]
            else:
                raise ValueError(f"Unknown strategy {strategy}")

            updates, times = [], []
            for cid in sel:
                w, t = clients[cid].train(global_params, epochs)
                updates.append(w)
                times.append(t)
                sel_counts[cid] += 1

            # aggregate
            global_params = {k: sum(u[k] for u in updates) / len(updates) for k in global_params}

            # evaluate
            global_model.load_state_dict(global_params)
            global_model.eval()
            loader = DataLoader(self.dataset, batch_size=128)
            criterion = nn.CrossEntropyLoss()
            total_loss, correct = 0.0, 0
            all_true, all_probs = [], []
            with torch.no_grad():
                for x,y in loader:
                    x,y = x.to(self.device), y.to(self.device)
                    out = global_model(x)
                    loss = criterion(out, y)
                    total_loss += loss.item() * x.size(0)
                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    probs = F.softmax(out, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_true.extend(y.cpu().numpy())

            avg_loss = total_loss / len(self.dataset)
            acc = correct / len(self.dataset)
            # compute ROC-AUC scores (macro and micro)
            auc_macro = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
            auc_micro = roc_auc_score(all_true, all_probs, multi_class='ovr', average='micro')

            total_time += sum(times)
            s = sel_counts
            jfi = (s.sum()**2) / (num_clients * (s**2).sum())

            history['round'].append(r)
            history['time'].append(total_time)
            history['loss'].append(avg_loss)
            history['acc'].append(acc)
            history['jfi'].append(jfi)
            history['auc'].append(auc_macro)
            history['roc'].append(auc_micro)

            print(f"{self.name} R{r} [{strategy} - {'dynamic' if dynamic_resources else 'static'}]: "
                  f"t={total_time:.2f}s, loss={avg_loss:.4f}, acc={acc:.4f}, jfi={jfi:.4f}, "
                  f"auc={auc_macro:.4f}, roc={auc_micro:.4f}")

        return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='newresults')
    parser.add_argument('--dynamic_resources', action='store_true',
                        help='If set, computation and communication speeds change each round')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    datasets = {
        'EMNIST': EMNIST(os.path.join('..','data','emnist'), split='balanced', download=True,
                         transform=transforms.ToTensor()),
        'FashionMNIST': FashionMNIST(os.path.join('..','data'), download=True,
                                     transform=transforms.ToTensor()),
        'CIFAR10': CIFAR10(os.path.join('..','data'), download=True,
                           transform=transforms.ToTensor())
    }
    client_counts = [10, 20, 50]
    splits = {
        'iid': partition_data_iid,
        'quantity_skew': partition_data_quantity_skew_iid,
        'class_noniid': partition_data_class_noniid
    }
    strategies = ['random', 'comp_greedy', 'comm_greedy', 'rbff', 'rbcsf']

    for name, dataset in datasets.items():
        for n in client_counts:
            partitions = {k: fn(dataset, n) for k, fn in splits.items()}
            for dist, indices in partitions.items():
                for strat in strategies:
                    sim = FederatedSimulation(name, dataset,
                                              lambda: SimpleCNN(dataset[0][0].shape[0], len(dataset.classes)),
                                              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    history = sim.run(indices, strat, dynamic_resources=args.dynamic_resources)
                    mode = 'dynamic' if args.dynamic_resources else 'static'
                    fname = f"{name}_{n}_{dist}_{strat}_{mode}.csv"
                    path = os.path.join(args.results_dir, fname)
                    with open(path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['round', 'time', 'loss', 'acc', 'jfi', 'auc', 'roc'])
                        for r, t, l, a, j, uu, rr in zip(
                                history['round'], history['time'], history['loss'],
                                history['acc'], history['jfi'], history['auc'], history['roc']):
                            writer.writerow([r, t, l, a, j, uu, rr])
    print("All experiments completed.")

if __name__ == "__main__":
    main()
