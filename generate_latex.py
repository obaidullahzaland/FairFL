#!/usr/bin/env python3
import os
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = 'newresults'
DATASET      = 'CIFAR10'
CLIENTS      = 50

DISTRIBUTIONS = ['iid', 'class_noniid', 'quantity_skew']
METHODS       = ['random', 'rbff', 'rbcsf', 'comm_greedy', 'comp_greedy']
MODES         = ['static', 'dynamic']

METRICS = [
    ('acc',  lambda x: x * 100, r'Acc $\uparrow$'),
    ('time', lambda x: x,       r'Time $\downarrow$'),
    ('jfi',  lambda x: x,       r'JFI $\uparrow$'),
    ('auc',  lambda x: x,       r'AUC $\uparrow$'),
    ('roc',  lambda x: x,       r'ROC $\uparrow$'),
]
# ────────────────────────────────────────────────────────────────────────────────

# 1) Gather “final‐round” rows for every existing file
records = []
for dist in DISTRIBUTIONS:
    for meth in METHODS:
        for mode in MODES:
            fname = f"{DATASET}_{CLIENTS}_{dist}_{meth}_{mode}.csv"
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                print(f"– missing {fname}, skipping")
                continue

            df = pd.read_csv(fpath)
            # pick final round
            last = df.loc[df['round'].idxmax()]
            # annotate
            rec = {
                'dataset': DATASET,
                'clients': CLIENTS,
                'distribution': dist,
                'strategy': meth,
                'mode': mode,
            }
            # compute every metric
            for key, func, _ in METRICS:
                rec[key] = func(last[key])
            records.append(rec)

if not records:
    raise RuntimeError("No valid CSVs found for EMNIST with 50 clients!")

results_df = pd.DataFrame(records)

# 2) Pivot into nested dict for easy table‐building
results = {}
for mode in MODES:
    results[mode] = {}
    for meth in METHODS:
        results[mode][meth] = {}
        for dist in DISTRIBUTIONS:
            sub = results_df.query(
                "mode==@mode and strategy==@meth and distribution==@dist"
            )
            if sub.empty:
                vals = {m[0]: None for m in METRICS}
            else:
                row = sub.iloc[0]
                vals = {m[0]: row[m[0]] for m in METRICS}
            results[mode][meth][dist] = vals

# 3) Assemble LaTeX table
lines = []
lines.append(r"\begin{table*}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \caption{Performance of different methods under various data distributions on the EMNIST dataset with 50 clients.}")
lines.append(r"  \label{tab:results}")
lines.append(r"  \begin{tabularx}{\textwidth}{m{1.5cm} m{2cm} *{15}{X}}")
lines.append(r"    \toprule")
# first header
hdr = r"    Resource & Method "
for d in DISTRIBUTIONS:
    hdr += f"& \\multicolumn{{5}}{{c}}{{{d.replace('_',' ').title()}}} "
hdr += r"\\"
lines.append(hdr)
lines.append(r"    \cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}")
# second header
sec = r"      &  "
for _ in DISTRIBUTIONS:
    sec += " & ".join(m[2] for m in METRICS) + "  \\\\  "
lines.append(sec)
lines.append(r"    \midrule")
# body
for mode in MODES:
    lines.append(f"    \\multirow{{{len(METHODS)}}}{{*}}{{{mode.capitalize()}}}")
    for meth in METHODS:
        row = f"      & {meth.replace('_','-').capitalize():<12}"
        for dist in DISTRIBUTIONS:
            vals = results[mode][meth][dist]
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

# 4) Output
print("\n".join(lines))
