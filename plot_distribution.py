# plot_distribution.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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

def generate_dynamic_publication_chart():
    # Scan the directory dynamically to isolate valid node size indices
    files = glob.glob("outputs/outputs_n*_lattice_v5_colab.csv")
    if not files:
        print("[!] Target tracking output files missing from outputs/ folder.")
        return

    data_map = {}
    for f in files:
        # Isolate size identifier tokens safely (e.g. outputs_n30_... -> 30)
        filename = os.path.basename(f)
        n_val = filename.split('_')[1].replace('n', '')
        
        # Read unique shapes metrics across data columns
        df = pd.read_csv(f, dtype=str)
        data_map[int(n_val)] = len(df)

    # Sort keys sequentially to maintain progressive linear scales
    sorted_sizes = sorted(data_map.keys())
    dimensions = [f"N={size}" for size in sorted_sizes]
    counts = [data_map[size] for size in sorted_sizes]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    bars = ax.bar(dimensions, counts, color='#1f77b4', alpha=0.85, edgecolor='black', width=0.4, linewidth=1.0)

    ax.set_ylabel('Unique Verified Lattices Discovered', fontweight='bold', labelpad=10)
    ax.set_xlabel('Matrix Dimension ($N$)', fontweight='bold', labelpad=10)
    ax.set_title('Combinatorial Lattice Generation Yield across Scales', fontweight='bold', pad=15)
    ax.grid(True, axis='y')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    output_path = 'plots/dataset_distribution_yield.pdf'
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()
    print(f"[+] Publication-ready vector PDF chart saved to: {output_path}")

if __name__ == "__main__":
    generate_dynamic_publication_chart()
