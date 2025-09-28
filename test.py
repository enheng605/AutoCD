import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 19,
    'legend.fontsize': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    #'font.weight': 'bold'  # 加粗字体
})
# Data provided by the user
lambdas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
datasets = ['Cora', 'Citeseer', 'Wiki']
custom_colors = {
    'Cora': '#4C9AC9',  # blue
    'Citeseer': '#F5B99E', # orange
    'Wiki': '#E2745E'  # red
}
Q_values = {
    'Cora': [72.28, 72.51, 73.92, 73.88, 73.23, 74.55, 73.57],
    'Citeseer': [70.00, 72.87, 72.97, 73.99, 74.57, 75.56, 74.3],
    'Wiki': [69.33, 70.17, 70.31, 71.39, 71.59, 72.04, 71.83],
}
NMI_values = {
    'Cora': [51.11, 51.98, 53.96, 51.15, 52.75, 55.32, 56.73],
    'Citeseer': [36.9, 37.14, 36.91, 38.24, 41.24, 42.58, 42.78],
    'Wiki': [49.8, 50.36, 49.09, 50.53, 51.89, 50.62, 51.06],
}

# Q_stddev = {
#     'Cora': np.random.uniform(0.5, 0., len(lambdas)),
#     'Citeseer': np.random.uniform(0.5, 1, len(lambdas)),
#     'Wiki': np.random.uniform(0.5, 1, len(lambdas))
# }
# NMI_stddev = {
#     'Cora': np.random.uniform(0.5, 1, len(lambdas)),
#     'Citeseer': np.random.uniform(0.5, 1, len(lambdas)),
#     'Wiki': np.random.uniform(0.5, 1, len(lambdas))
# }

# Generate line plots with shaded regions
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot Q values with shaded regions
for dataset in datasets:
    mean = Q_values[dataset]
    #std = Q_stddev[dataset]
    axes[0].plot(lambdas, mean, label=dataset, color=custom_colors[dataset], marker='o', linewidth=4.5)
    #axes[0].fill_between(lambdas, np.array(mean) - std, np.array(mean) + std, color=custom_colors[dataset], alpha=0.3)
#axes[0].set_title('Q / λ',fontweight='bold')
axes[0].set_xlabel('λ')
axes[0].set_ylabel('Q ')
axes[0].legend()
axes[0].grid(True)

# Plot NMI values with shaded regions
for dataset in datasets:
    mean = NMI_values[dataset]
    #std = NMI_stddev[dataset]
    axes[1].plot(lambdas, mean, label=dataset, color=custom_colors[dataset], marker='o', linewidth=4.5)
    #axes[1].fill_between(lambdas, np.array(mean) - std, np.array(mean) + std, color=custom_colors[dataset], alpha=0.3)
#axes[1].set_title('NMI / λ',fontweight='bold')
axes[1].set_xlabel('λ')
axes[1].set_ylabel('NMI ')
axes[1].legend()
axes[1].grid(True)

# Show the plots
plt.tight_layout()
plt.savefig("Q_NMI.png")
plt.show()



