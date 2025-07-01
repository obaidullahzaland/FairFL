#!/usr/bin/env python3
import os
import pandas as pd
from glob import glob

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'newresults'
DATASET    = 'FashionMNIST'
CLIENTS    = 50

DISTRIBUTIONS = ['iid', 'class_noniid', 'quantity_skew']
DIST_LABELS   = {
    'iid': 'IID',
    'class_noniid': 'Class non-IID',
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
    parts = name.split('_')
    mode, strat, dist = parts[-1], parts[-2], parts[-3]
    clients = int(parts[-4])
    dataset = '_'.join(parts[:-4])
    df = pd.read_csv(fp)
    df['mode']         = mode
    df['strategy']     = strat
    df['distribution'] = dist
    df['dataset']      = dataset
    df['clients']      = clients
    records.append(df)
if not records:
    raise RuntimeError(f"No CSVs found in {RESULTS_DIR}")
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

# 4) Build LaTeX
# Header
lines = []
lines.append(r"\begin{table*}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \caption{Performance of different methods under various data distributions on the FashionMNIST dataset with 50 clients.}")
lines.append(r"  \label{tab:results}")
lines.append(r"  \begin{tabularx}{\textwidth}{m{1.5cm} m{2cm} *{15}{X}}")
lines.append(r"    \toprule")
# First header row
hdr = r"    Resource & Method "
for d in DISTRIBUTIONS:
    hdr += f"& \\multicolumn{{5}}{{c}}{{{DIST_LABELS[d]}}} "
hdr += r"\\"
lines.append(hdr)
# cmidrules
lines.append(r"    \cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}")
# Second header row
sec = r"      &  "
for _ in DISTRIBUTIONS:
    sec += " & ".join(m[2] for m in METRICS) + "  \\\\  "
lines.append(sec)
lines.append(r"    \midrule")

# Body
for mode in MODES:
    lines.append(f"    \\multirow{{{len(STRATEGIES)}}}{{*}}{{{mode.capitalize()}}}")
    for strat in STRATEGIES:
        row = f"      & {strat.replace('_','-').capitalize():<12}"
        for dist in DISTRIBUTIONS:
            vals = results[mode][strat][dist]
            for key, _, _ in METRICS:
                if vals[key] is None:
                    row += " & ---"
                else:
                    fmt = "{:.2f}" if key=='acc' else "{:.3f}"
                    row += " & " + fmt.format(vals[key])
        row += r" \\"
        lines.append(row)
    lines.append(r"    \midrule")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabularx}")
lines.append(r"\end{table*}")

# 5) Print it all
print("\n".join(lines))
