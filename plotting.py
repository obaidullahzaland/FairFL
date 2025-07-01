import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

# -----------------------
# Plotting Configuration
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate federated learning performance and fairness plots.")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='newresults',
        help='Directory containing result CSV files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='newplots',
        help='Directory to save generated plots'
    )
    return parser.parse_args()


def load_results(results_dir):
    records = []
    distributions = ['iid', 'quantity_skew', 'class_noniid']
    strategies = ['comm_greedy', 'comp_greedy', 'rbff', 'rbcsf', 'random']
    modes = ['static', 'dynamic']

    for fname in os.listdir(results_dir):
        if not fname.endswith('.csv'):
            continue
        name = fname[:-4]
        # Extract mode
        mode = next((m for m in modes if name.endswith('_' + m)), None)
        if not mode:
            print(f"Skipping {fname}: unknown mode")
            continue
        name = name[:-(len(mode) + 1)]
        # Extract strategy
        strategy = next((s for s in strategies if name.endswith('_' + s)), None)
        if not strategy:
            print(f"Skipping {fname}: unknown strategy")
            continue
        name = name[:-(len(strategy) + 1)]
        # Extract distribution
        distribution = next((d for d in distributions if name.endswith('_' + d)), None)
        if not distribution:
            print(f"Skipping {fname}: unknown distribution")
            continue
        name = name[:-(len(distribution) + 1)]
        # Remaining: dataset_clients
        try:
            dataset, clients = name.rsplit('_', 1)
            clients = int(clients)
        except ValueError:
            print(f"Skipping {fname}: invalid dataset/clients")
            continue

        # Read CSV
        path = os.path.join(results_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: could not read {fname}: {e}")
            continue
        df['dataset'] = dataset
        df['clients'] = clients
        df['distribution'] = distribution
        df['strategy'] = strategy
        df['mode'] = mode
        records.append(df)

    if not records:
        raise RuntimeError(f"No valid CSV files found in {results_dir}")
    return pd.concat(records, ignore_index=True)


def generate_plots(df, output_dir):
    # now include 'auc' as a fifth metric
    metrics = ['acc', 'loss', 'time', 'jfi', 'auc']
    ylabels = ['Accuracy (%)', 'Log Loss', 'Time (s)', 'JFI', 'AUC']

    datasets     = sorted(df['dataset'].unique())
    distributions= sorted(df['distribution'].unique())
    clients_list = sorted(df['clients'].unique())
    modes        = sorted(df['mode'].unique())

    for mode in modes:
        df_mode = df[df['mode'] == mode]
        for distribution in distributions:
            df_dist = df_mode[df_mode['distribution'] == distribution]
            for clients in clients_list:
                df_sub = df_dist[df_dist['clients'] == clients]
                if df_sub.empty:
                    continue

                # 3 rows (datasets) x 5 cols (metrics)
                fig, axes = plt.subplots(
                    nrows=len(datasets), ncols=len(metrics),
                    figsize=(5 * len(metrics), 4 * len(datasets))
                )

                subtitle_text = f"Distribution: {distribution} | Clients: {clients} | Mode: {mode.capitalize()}"
                fig.suptitle(subtitle_text, fontsize=16, y=0.95)

                for i, dataset in enumerate(datasets):
                    df_data = df_sub[df_sub['dataset'] == dataset]
                    for j, metric in enumerate(metrics):
                        ax = axes[i][j] if len(datasets) > 1 else axes[j]
                        for strategy in sorted(df_data['strategy'].unique()):
                            strat_df = df_data[df_data['strategy'] == strategy]
                            x = strat_df['round']
                            if metric == 'acc':
                                y = strat_df['acc'] * 100
                            elif metric == 'loss':
                                y = np.log(strat_df['loss'])
                            else:
                                y = strat_df[metric]
                            ax.plot(
                                x, y,
                                label=strategy,
                                linewidth=3
                            )
                        # Axis scaling and labels
                        if metric == 'acc':
                            ax.set_ylim(0, 100)
                        if j == 0:
                            ax.set_ylabel(dataset, fontsize=14)
                        ax.set_xlabel('Round', fontsize=14)
                        ax.set_title('')        # remove individual subplot titles
                        ax.tick_params(axis='both', labelsize=12)

                        # only set y-label on the top row for clarity
                        ax.set_ylabel(ylabels[j] if i == 0 else '')

                # Shared legend on top, horizontal
                handles, labels = axes[0][0].get_legend_handles_labels()
                fig.legend(
                    handles, labels,
                    loc='upper center',
                    ncol=len(labels),
                    fontsize=12,
                    bbox_to_anchor=(0.5, 1.02)
                )

                plt.tight_layout(rect=[0, 0, 1, 0.92])

                # Save figure
                fname = f"{distribution}_{clients}_{mode}.png"
                save_path = os.path.join(output_dir, fname)
                fig.savefig(save_path)
                plt.close(fig)
                print(f"Saved plot: {save_path}")


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.results_dir)
    generate_plots(df, args.output_dir)
