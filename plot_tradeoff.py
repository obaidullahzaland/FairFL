import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_all(results_dir):
    """
    Returns a dict:
      data[experiment_type][strategy][dataset][split] → DataFrame
    where experiment_type ∈ {'by_clients','by_rate'},
          split ∈ {'iid','qty','class'}.
    """
    data = {'by_clients':{}, 'by_rate':{}}
    for exp in data:
        for csv_path in glob.glob(os.path.join(results_dir, exp, '*.csv')):
            fname = os.path.basename(csv_path).replace('.csv','')
            ds, tag, strat, mode = fname.split('_',3)
            # tag is like '10cli_iid' or '30pct_class'
            if strat not in data[exp]:
                data[exp][strat] = {}
            if ds not in data[exp][strat]:
                data[exp][strat][ds] = {}
            # extract split and x-value
            if 'cli' in tag:
                x_val, split = tag.split('cli_')
                x_val = int(x_val)
            else:
                x_val, split = tag.split('pct_')
                x_val = int(x_val)
            df = pd.read_csv(csv_path)
            df['x'] = x_val
            df['split'] = split
            df['mode'] = mode  # 'static' or 'dynamic'
            data[exp][strat][ds][split + '_' + mode] = df
    return data

def plot_experiment(data, experiment_type, x_label):

    strategies = sorted(data[experiment_type].keys())
    datasets = sorted(next(iter(data[experiment_type].values())).keys())
    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 4), sharey=False)
    for ax, strat in zip(axes, strategies):
        ax2 = ax.twinx()
        for ds in datasets:
            for split in ['iid','qty','class']:
                for mode, ls in [('static','--'),('dynamic','-')]:
                    key = f"{split}_{mode}"
                    df = data[experiment_type][strat][ds].get(key)
                    if df is None: continue
                    # sort by x
                    df = df.sort_values('x')
                    # accuracy on left y
                    ax.plot(df['x'], df['acc']*100,
                            linestyle=ls, marker='o',
                            label=f"{ds}-{split}-{mode}")
                    # jfi on right y
                    ax2.plot(df['x'], df['jfi'],
                             linestyle=ls, marker='x')
        ax.set_title(strat)
        ax.set_xlabel(x_label)
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel('Accuracy (%)')
            ax2.set_ylabel('JFI')
        # build custom legend once on top
    handles, labels = [], []
    for ds in datasets:
        for split in ['iid','qty','class']:
            handles.append(plt.Line2D([],[], color='black', linestyle='-', marker='o'))
            labels.append(f"{ds}-{split}-dynamic")
            handles.append(plt.Line2D([],[], color='black', linestyle='--', marker='o'))
            labels.append(f"{ds}-{split}-static")
    fig.legend(handles, labels, loc='upper center', ncol=3*len(datasets))
    fig.tight_layout(rect=[0,0,1,0.9])
    return fig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_tradeoff')
    args = parser.parse_args()

    data = load_all(args.results_dir)
    # Plot #1: by number of clients
    fig1 = plot_experiment(data, 'by_clients', x_label='Number of clients')
    fig1.savefig('accuracy_jfi_by_clients.png', dpi=300)

    # Plot #2: by participation rate
    fig2 = plot_experiment(data, 'by_rate', x_label='Participation rate (%)')
    fig2.savefig('accuracy_jfi_by_rate.png', dpi=300)

    print("Plots saved: accuracy_jfi_by_clients.png, accuracy_jfi_by_rate.png")
