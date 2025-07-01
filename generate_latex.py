#!/usr/bin/env python3
import os
import pandas as pd
from glob import glob

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'newresults'
DATASET    = 'FashionMNIST'
CLIENTS    = 50

# note: 'non_iid' here replaces your old 'class_noniid'
DISTRIBUTIONS = ['iid', 'non_iid', 'quantity_skew']
DIST_LABELS   = {
    'iid': 'IID',
    'non_iid': 'Class non-IID',
    'quantity_skew': 'Quantity non-IID'
}
STRATEGIES    = ['random', 'comp_greedy', 'comm_greedy', 'rbff', 'rbcsf']
MODES         = ['static', 'dynamic']

METRICS = [
    ('acc', lambda x: x * 100, r'Acc $\uparrow$'),
    ('time', lambda x: x,      r'Time $\downarrow$'),
    ('jfi', lambda x: x,       r'JFI $\uparrow$'),
    ('auc', lambda x: x,       r'AUC $\uparrow$'),
    ('roc', lambda x: x,       r'ROC $\uparrow$'),
]
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load all CSVs into one DataFrame with annotations
records = []
for fp in glob(os.path.join(RESULTS_DIR, '*.csv')):
    name = os.path.basename(fp)[:-4]
    try:
        dataset, clients_str, dist, strat, mode = name.split('_')
    except ValueError:
        print(f"Skipping `{name}.csv`: filename must be Dataset_Clients_Distribution_Method_Mode.csv")
        continue

    if dataset not in ['FashionMNIST', 'EMNIST', 'CIFAR10']:
        print(f"Skipping `{name}.csv`: unknown dataset `{dataset}`")
        continue
    if dist not in DISTRIBUTIONS:
        print(f"Skipping `{name}.csv`: unknown distribution `{dist}`")
        continue
    if strat not in STRATEGIES:
        print(f"Skipping `{name}.csv`: unknown strategy `{strat}`")
        continue
    if mode not in MODES:
        print(f"Skipping `{name}.csv`: unknown mode `{mode}`")
        continue

    try:
        clients = int(clients_str)
    except ValueError:
        print(f"Skipping `{name}.csv`: invalid client count `{clients_str}`")
        continue

    df = pd.read_csv(fp)
    df['dataset']      = dataset
    df['clients']      = clients
    df['distribution'] = dist
    df['strategy']     = strat
    df['mode']         = mode
    records.append(df)

if not records:
    raise RuntimeError(f"No valid CSV files found in {RESULTS_DIR}")

all_df = pd.concat(records, ignore_index=True)

# 2) Filter to FashionMNIST & 50 clients
df = all_df.query("dataset == @DATASET and clients == @CLIENTS")

# 3) Pull final‐round metrics
results = {}
for mode in MODES:
    results[mode] = {}
    for strat in STRATEGIES:
        results[mode][strat] = {}
        for dist in DISTRIBUTIONS:
            sub = df.query("mode == @mode and strategy == @strat and distribution == @dist")
            if sub.empty:
                vals = {m[0]: None for m in METRICS}
            else:
                last = sub.loc[sub['round'].idxmax()]
                vals = {m[0]: m[1](last[m[0]]) for m in METRICS}
            results[mode][strat][dist] = vals

# 4) Build LaTeX table as before...
#    (same code to assemble header, cmidrules, body, etc.)
#    Just reuse the block from the prior script.

# 5) Print it all
#    print( ... )  # same as before
