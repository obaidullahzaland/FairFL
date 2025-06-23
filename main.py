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

# -----------------------
# Data Partitioning
# -----------------------
def partition_data_iid(dataset, num_clients):
    data_size = len(dataset)
    shard_size = data_size // num_clients
    indices = list(range(data_size))
    random.shuffle(indices)
    return [indices[i*shard_size:(i+1)*shard_size] for i in range(num_clients)]


def partition_data_quantity_skew_iid(dataset, num_clients, alpha=0.5):
    data_size = len(dataset)
    props = np.random.dirichlet([alpha]*num_clients)
    sizes = (props/props.sum() * data_size).astype(int)
    sizes[-1] = data_size - sizes[:-1].sum()
    indices = list(range(data_size))
    random.shuffle(indices)
    client_indices, idx = [], 0
    for sz in sizes:
        client_indices.append(indices[idx:idx+sz]); idx += sz
    return client_indices


def partition_data_class_noniid(dataset, num_clients, classes_per_client=2):
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    # assign classes
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

# -----------------------
# Client Simulation
# -----------------------
class Client:
    def __init__(self, cid, dataset, indices, model_fn, device, comp_speed, comm_speed):
        self.cid = cid; self.device = device
        self.comp_speed, self.comm_speed = comp_speed, comm_speed
        self.model_fn = model_fn
        subset = Subset(dataset, indices)
        self.loader = DataLoader(subset, batch_size=32, shuffle=True)

    def train(self, global_params, epochs=1, lr=0.01):
        model = self.model_fn(); model.load_state_dict(global_params)
        model.to(self.device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        compute_time = len(self.loader.dataset)/self.comp_speed
        for _ in range(epochs):
            for x,y in self.loader:
                x,y = x.to(self.device), y.to(self.device)
                opt.zero_grad(); out=model(x); loss=crit(out,y)
                loss.backward(); opt.step()
        param_size = sum(p.numel() for p in model.parameters())*4
        comm_time = param_size/self.comm_speed
        return model.state_dict(), compute_time+comm_time

# -----------------------
# Federated Simulation
# -----------------------
class FederatedSimulation:
    def __init__(self, name, dataset, model_fn, device):
        self.name, self.dataset = name, dataset
        self.model_fn, self.device = model_fn, device

    def run(self, client_indices, strategy, num_rounds=50, epochs=1):
        num_clients = len(client_indices)
        # init global
        global_model = self.model_fn().to(self.device)
        global_params = global_model.state_dict()
        # init clients
        reputations = np.zeros(num_clients)
        sel_counts = np.zeros(num_clients)
        clients=[]
        for cid,inds in enumerate(client_indices):
            comp=random.uniform(50,200); comm=random.uniform(1e5,5e5)
            reputations[cid]=comp+comm
            clients.append(Client(cid, self.dataset, inds,
                                  lambda: self.model_fn(), self.device,
                                  comp, comm))
        history={'round':[],'time':[],'loss':[],'acc':[]}
        total_time=0.0
        for r in range(1,num_rounds+1):
            # selection
            if strategy=='random':
                sel = np.random.choice(num_clients,5,replace=False)
            elif strategy=='comp_greedy':
                sel = np.argsort([-c.comp_speed for c in clients])[:5]
            elif strategy=='comm_greedy':
                sel = np.argsort([-c.comm_speed for c in clients])[:5]
            elif strategy=='rbff':
                scores = reputations/(1+sel_counts)
                sel = np.argsort(-scores)[:5]
            elif strategy=='rbcsf':
                scores = reputations - sel_counts
                sel = np.argsort(-scores)[:5]
            else:
                raise ValueError(f"Unknown strategy {strategy}")
            # train
            updates, times = [], []
            for cid in sel:
                w,t=clients[cid].train(global_params, epochs)
                updates.append(w); times.append(t)
                sel_counts[cid]+=1
            # aggregate
            newp={k:sum(u[k] for u in updates)/len(updates) for k in global_params}
            global_params=newp
            # eval
            global_model.load_state_dict(global_params)
            global_model.eval()
            loader=DataLoader(self.dataset,batch_size=128)
            crit=nn.CrossEntropyLoss()
            loss,sum_correct=0,0
            with torch.no_grad():
                for x,y in loader:
                    x,y=x.to(self.device),y.to(self.device)
                    out=global_model(x); loss+=crit(out,y).item()*x.size(0)
                    sum_correct+=(out.argmax(1)==y).sum().item()
            loss/=len(self.dataset)
            acc=sum_correct/len(self.dataset)
            total_time+=sum(times)
            history['round'].append(r)
            history['time'].append(total_time)
            history['loss'].append(loss)
            history['acc'].append(acc)
            print(f"{self.name} R{r} [{strategy}]: t={total_time:.2f}s, loss={loss:.4f}, acc={acc:.4f}")
        # JFI
        s=sel_counts; jfi=(s.sum()**2)/(num_clients*(s**2).sum())
        return history, global_params, total_time, jfi

# -----------------------
# Main Experiment Driver
# -----------------------
def main(results_dir="results"):
    os.makedirs(results_dir,exist_ok=True)
    log_path=os.path.join(results_dir,'log.csv')
    if not os.path.exists(log_path):
        with open(log_path,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['dataset','n_clients','distribution','strategy','final_loss','final_acc','total_time','JFI'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        'EMNIST': EMNIST('../data/emnist', split='balanced', download=True, transform=transforms.ToTensor()),
        'FashionMNIST': FashionMNIST('../data', download=True, transform=transforms.ToTensor()),
        'CIFAR10': CIFAR10('../data', download=True, transform=transforms.ToTensor())
    }
    client_counts=[10,20,50]
    splits={'iid':partition_data_iid,'quantity_skew':partition_data_quantity_skew_iid,'class_noniid':partition_data_class_noniid}
    strategies=['random','comp_greedy','comm_greedy','rbff','rbcsf']
    for name,dset in datasets.items():
        for n in client_counts:
            idxs={k:fn(dset,n) for k,fn in splits.items()}
            for dist,indices in idxs.items():
                for strat in strategies:
                    sim=FederatedSimulation(name,dset,lambda:SimpleCNN(dset[0][0].shape[0],len(dset.classes)),device)
                    hist,params,t,jfi = sim.run(indices,strat)
                    final_loss,final_acc=hist['loss'][-1],hist['acc'][-1]
                    with open(log_path,'a',newline='') as f:
                        csv.writer(f).writerow([name,n,dist,strat,final_loss,final_acc,t,jfi])
    print("All experiments completed.")

if __name__=="__main__":
    main()
