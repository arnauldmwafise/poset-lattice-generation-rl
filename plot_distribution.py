# plot_distribution.py
import os
import matplotlib.pyplot as plt

# Professional typography and styling configurations
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Dataset distribution metrics from your verified GPU sweep metrics
dimensions = ['N=30', 'N=40', 'N=50']
counts = [1835, 1726, 516]

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# Render professional desaturated bar structure with structural borders
bars = ax.bar(dimensions, counts, color='#1f77b4', alpha=0.85, edgecolor='black', width=0.4, linewidth=1.0)

# Visual polish parameters
ax.set_ylabel('Unique Verified Lattices Discovered', fontweight='bold', labelpad=10)
ax.set_xlabel('Matrix Dimension ($N$)', fontweight='bold', labelpad=10)
ax.set_title('Combinatorial Lattice Generation Yield across Scales', fontweight='bold', pad=15)
ax.grid(True, axis='y')

# Remove unnecessary structural boundary lines (Spines) for publication minimalism
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Add exact value labels on top of each bar chart element
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
output_path = 'plots/dataset_distribution_yield.pdf'
plt.savefig(output_path, format='pdf', dpi=300)
plt.close()
print(f"[+] Publication-ready PDF chart saved to: {output_path}")
