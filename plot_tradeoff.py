import os
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size":        16,   # base font size for text
    "axes.titlesize":   16,   # subplot title
    "axes.labelsize":   16,   # x- and y-axis labels
    "xtick.labelsize":  16,   # x-tick labels
    "ytick.labelsize":  16,   # y-tick labels
    "legend.fontsize":  16,   # legend text
})


# --- Configuration ---
datasets      = ["EMNIST", "FashionMNIST", "CIFAR10"]
clients       = [10, 20, 30, 40, 50]
methods       = ["comm_greedy", "comp_greedy", "random", "rbff", "rbcsf"]
display_names = {
    "random":      "Random",
    "comm_greedy": "Comm_Greedy",
    "comp_greedy": "Comp_Greedy",
    "rbff":        "RBFF",
    "rbcsf":       "RBCSF",
}
distribution = "iid"
data_dir     = "results_tradeoff/by_clients"

# --- Figure setup: each subplot region is 5×5 inches, so square ---
subplot_size = 5  # inches
n_methods    = len(methods)
fig, axes    = plt.subplots(
    1, n_methods,
    figsize=(subplot_size * n_methods, subplot_size),
    sharey=False
)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Collect legend entries from first subplot
legend_handles, legend_labels = [], []

for idx, (ax, method) in enumerate(zip(axes, methods)):
    ax2 = ax.twinx()

    for di, dataset in enumerate(datasets):
        accs, jf_is = [], []
        for n in clients:
            fname = f"{dataset}_{n}cli_{distribution}_{method}_dynamic.csv"
            path  = os.path.join(data_dir, fname)
            df    = pd.read_csv(path)
            accs.append(df["acc"].iloc[-1])
            jf_is.append(df["jfi"].iloc[-1])

        color = colors[di]
        ax.plot(clients, accs,
                label=f"{dataset} ACC",
                linestyle="solid",
                color=color)
        ax2.plot(clients, jf_is,
                 label=f"{dataset} JFI",
                 linestyle="dashed",
                 color=color)

    # x-ticks and axis limits
    ax.set_xticks(clients)
    ax.set_ylim(0.3, 1.0)
    ax2.set_ylim(0.9, 1.0)

    # method name underneath
    ax.set_xlabel(display_names[method], labelpad=12, fontsize=16)

    # only first subplot gets y-labels and legend capture
    if idx == 0:
        ax.set_ylabel("Accuracy")
        ax2.set_ylabel("JFI")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        legend_handles = h1 + h2
        legend_labels  = l1 + l2

# # global legend at top
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=len(legend_labels),     # one column per entry → one row
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
    borderaxespad=0,             # no extra padding
    handletextpad=0.5,           # tighten space between legend handle and text
    columnspacing=1.0            # space between columns
)


# --- Configuration ---
# datasets     = ["EMNIST", "FashionMNIST", "CIFAR10"]
# percentages  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# methods      = ["comm_greedy", "comp_greedy", "random", "rbff", "rbcsf"]
# distribution = "iid"
# data_dir     = "results_tradeoff/by_rate"

# # --- Set up figure ---
# fig, axes = plt.subplots(1, len(methods), figsize=(25, 5), sharey=False)
# colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

# # Collect legend entries from first subplot
# legend_handles = []
# legend_labels  = []

# for mi, (ax, method) in enumerate(zip(axes, methods)):
#     ax2 = ax.twinx()
#     for di, dataset in enumerate(datasets):
#         accs, jf_is = [], []
#         for p in percentages:
#             fname = f"{dataset}_{p}pct_{distribution}_{method}_static.csv"
#             fpath = os.path.join(data_dir, fname)
#             df    = pd.read_csv(fpath)
#             # take the final round’s metrics
#             accs.append(df["acc"].iloc[-1])
#             jf_is.append(df["jfi"].iloc[-1])

#         color = colors[di]
#         ax.plot(percentages, accs,
#                 label=f"{dataset} ACC",
#                 linestyle="solid",
#                 color=color)
#         ax2.plot(percentages, jf_is,
#                  label=f"{dataset} JFI",
#                  linestyle="dashed",
#                  color=color)

#     ax.set_title(method.replace("_", " ").title())
#     ax.set_xlabel("Percentage of Data (%)")
#     #     ax.set_xticks(clients)
#     ax.set_ylim(0.3, 1.0)
#     ax2.set_ylim(0.3, 1.0)
#     ax.set_xticks(percentages)
#     if mi == 0:
#         ax.set_ylabel("Accuracy")
#         ax2.set_ylabel("JFI")
#         # capture legend entries once
#         h1, l1 = ax.get_legend_handles_labels()
#         h2, l2 = ax2.get_legend_handles_labels()
#         legend_handles = h1 + h2
#         legend_labels  = l1 + l2

# # --- Add common legend ---
# fig.legend(
#     legend_handles,
#     legend_labels,
#     loc="upper center",
#     ncol=len(legend_labels),     # one column per entry → one row
#     frameon=False,
#     bbox_to_anchor=(0.5, 1.),
#     borderaxespad=0,             # no extra padding
#     handletextpad=0.5,           # tighten space between legend handle and text
#     columnspacing=1.0            # space between columns
# )

plt.tight_layout(rect=[0, 0, 1, 0.95])
fname = "clients_jfiacc_dynamic"
save_path = os.path.join("plots", fname)
fig.savefig(save_path)
plt.close(fig)
print(f"Saved plot: {save_path}")
plt.show()

