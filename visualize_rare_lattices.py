# visualize_rare_lattices.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def bitstring_to_matrix(bitstring, n):
    """Converts a flat bitstring back to a square NxN adjacency matrix."""
    expected_len = n * n
    if len(bitstring) < expected_len:
        bitstring = bitstring.ljust(expected_len, '0')
    bits = [int(b) for b in bitstring[:expected_len]]
    return np.array(bits).reshape((n, n))

def compute_hasse_matrix(matrix):
    """
    Computes the transitive reduction of a directed acyclic graph matrix.
    This eliminates redundant paths, isolating pure Hasse diagram cover lines.
    """
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    TR = nx.transitive_reduction(G)
    hasse_matrix = nx.to_numpy_array(TR, dtype=int)
    return hasse_matrix

def generate_visualizations():
    results_file = "lattice_analysis_results.csv"
    output_dir = "rare_lattice_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading analysis data from {results_file}...")
    df = pd.read_csv(results_file, dtype=str)
    
    rare_df = df[(df['is_modular'] == 'True') & (df['is_distributive'] == 'False')]
    
    if rare_df.empty:
        print("No rare modular, non-distributive structures found in the file.")
        return
        
    print(f"Found {len(rare_df)} rare structures. Plotting Hasse adjacency matrices via Matplotlib...")
    
    for idx, (_, row) in enumerate(rare_df.iterrows(), 1):
        n = int(row['n'])
        hash_id = row['hash']
        bitstring = row['matrix_bits']
        
        adj_matrix = bitstring_to_matrix(bitstring, n)
        hasse_matrix = compute_hasse_matrix(adj_matrix)
        
        # Pure Matplotlib alternative to Seaborn Heatmap
        fig, ax = plt.subplots(figsize=(6.5, 6))
        
        # Display matrix as an image (Blues colormap)
        cax = ax.imshow(hasse_matrix, cmap="Blues", interpolation="nearest", aspect="equal")
        
        # Add a subtle grid to clearly outline matrix cells
        ax.set_xticks(np.arange(n) - 0.5, minor=True)
        ax.set_yticks(np.arange(n) - 0.5, minor=True)
        ax.grid(which="minor", color="lightgrey", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)  # Hide secondary tick marks
        
        # Set primary labels and ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(n), fontsize=8)
        ax.set_yticklabels(range(n), fontsize=8)
        
        ax.set_title(f"Rare Lattice Hasse Adjacency [N={n}]\nHash ID: {hash_id}", fontsize=11, pad=10)
        ax.set_xlabel("Element (To)", fontsize=9)
        ax.set_ylabel("Element (From)", fontsize=9)
        
        # Clean up bounds appearance
        ax.spines[:].set_color("grey")
        ax.spines[:].set_linewidth(0.5)
        
        save_path = os.path.join(output_dir, f"rare_lattice_{idx}_{hash_id}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f" -> Saved visualization {idx}/{len(rare_df)} to {save_path}")

    print(f"\nDone! View plots inside directory: '{output_dir}/'")

if __name__ == "__main__":
    generate_visualizations()
