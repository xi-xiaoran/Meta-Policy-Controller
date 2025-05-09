import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "figure.dpi": 150,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# X-axis: KL coefficients and their positions
lambdas = [0.01, 0.1, 1, 10]
x = np.arange(len(lambdas))  # Positions: 0, 1, 2, 3
xtick_labels = ["0.01", "0.1", "1", "10"]

# Dataset settings
datasets = ["MNIST", "SVHN", "CIFAR10"]
palette = sns.color_palette("deep")
colors = dict(zip(datasets, palette[:3]))
markers = dict(zip(datasets, ["o", "s", "D"]))

# Mean and standard deviation of metrics
acc_mean = np.array([
    [0.9920, 0.9912, 0.9911, 0.9993],
    [0.8752, 0.8669, 0.8592, 0.9689],
    [0.5770, 0.5968, 0.5454, 0.7832]
])
acc_std = np.array([
    [0.0005, 0.0006, 0.0015, 0.0010],
    [0.0029, 0.0045, 0.0045, 0.0056],
    [0.0325, 0.0051, 0.0021, 0.0035]
])

ece_mean = np.array([
    [0.1966, 0.1960, 0.1979, 0.2257],
    [0.1248, 0.1157, 0.1421, 0.2071],
    [0.1134, 0.1310, 0.0863, 0.1375]
])
ece_std = np.array([
    [0.0006, 0.0006, 0.0007, 0.0055],
    [0.0012, 0.0026, 0.0021, 0.0065],
    [0.0245, 0.0185, 0.0021, 0.0345]
])

mue_mean = np.array([
    [0.0449, 0.0542, 0.0619, 0.0397],
    [0.1680, 0.1666, 0.1426, 0.0856],
    [0.3093, 0.2938, 0.2703, 0.1505]
])
mue_std = np.array([
    [0.0123, 0.0074, 0.0263, 0.0100],
    [0.0032, 0.0051, 0.0015, 0.0081],
    [0.0104, 0.0045, 0.0035, 0.0150]
])

racc_mean = np.array([
    [0.9998, 0.9996, 0.9993, 0.9988],
    [0.9759, 0.9737, 0.9763, 0.9689],
    [0.7935, 0.8128, 0.7832, 0.8125]
])
racc_std = np.array([
    [0.0001, 0.0001, 0.0002, 0.0003],
    [0.0020, 0.0023, 0.0017, 0.0055],
    [0.0151, 0.0125, 0.0122, 0.0175]
])

# Start plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
axes = axes.flatten()
fig.suptitle("Experimental results of different KL coefficients λ", fontsize=22, y=0.95)

metrics = ['Accuracy↑', 'ECE↓', 'MUE↓', 'Rejection Accuracy↑']
mean_list = [acc_mean, ece_mean, mue_mean, racc_mean]
std_list = [acc_std, ece_std, mue_std, racc_std]

# List to store Line2D objects for the legend
lines = []

for idx, ax in enumerate(axes):
    for i, dataset in enumerate(datasets):
        mean = mean_list[idx][i]
        std = std_list[idx][i]
        line = ax.plot(x, mean, label=dataset, color=colors[dataset], marker=markers[dataset])
        ax.fill_between(x, mean - std, mean + std, color=colors[dataset], alpha=0.2)
        if idx == 0:  # Record only once for the legend
            lines.append(line[0])

    ax.set_title(metrics[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("KL Coefficient λ")
    if idx in [0, 2]:
        ax.set_ylabel("Metric Value")
    ax.grid(True)

# Place the legend at the bottom
fig.legend(handles=lines, labels=datasets, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=15)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("kl_ablation_final.png", dpi=300, bbox_inches='tight')
plt.show()