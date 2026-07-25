# visualize_n5_lattice.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def bitstring_to_matrix(bitstring, n=None):
    """Dynamically reconstructs a square matrix from any bitstring length."""
    if n is None:
        # Since it is a square matrix, n is the square root of the total bits
        n = int(len(bitstring) ** 0.5)
    
    bits = [int(b) for b in bitstring]
    return np.array(bits).reshape((n, n))


def plot_hasse_diagram():
    input_file = "outputs/classified_categories_colab/3_semimodular_non_modular_n5_colab.csv"
    
    if not os.path.exists(input_file):
        print(f"[!] Target file missing at {input_file}. Pull your Colab results first.")
        return

    df = pd.read_csv(input_file, dtype=str)
    if df.empty:
        print("[!] No records found inside the targeted N5 category file.")
        return

    # Extract the first ultra-rare N5 instance
    row = df.iloc[0]
    n = int(row['n'])
    hash_id = row['hash']
    
    print(f"[+] Reconstructing Hasse Topology for N5 Lattice. Hash: {hash_id}")
    adj_matrix = bitstring_to_matrix(row['matrix_bits'], n)
    
    # Generate Directed Acyclic Graph and strip transitive shortcuts
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    Hasse = nx.transitive_reduction(G)
    
    # Calculate layered vertical coordinates using topological layout scaling
    try:
        layers = list(nx.topological_generations(Hasse))
    except nx.NetworkXUnfeasible:
        print("[!] Graph contains cycles. Verification failed.")
        return

    pos = {}
    for idx, layer in enumerate(layers):
        width = len(layer)
        for j, node in enumerate(layer):
            # Center-align nodes across layered horizontal steps
            pos[node] = (j - (width - 1) / 2.0, idx)

    # Render Node-Link Structural Visualization Layout
    plt.figure(figsize=(8, 9))
    
    # Draw connections explicitly reversing the arrows to look like an upscale Hasse orientation
    nx.draw_networkx_edges(Hasse, pos, edge_color="darkgrey", arrows=False, width=1.2)
    nx.draw_networkx_nodes(Hasse, pos, node_size=400, node_color="#1f77b4", edgecolors="black", linewidths=1.0)
    nx.draw_networkx_labels(Hasse, pos, font_color="white", font_size=9, font_weight="bold")
    
    plt.title(f"Topological Hasse Diagram of Isolated Ultra-Rare Semimodular ($N_5$) Lattice\nHash ID: {hash_id} | Node Horizon: N={n}", fontsize=11, pad=15)
    plt.axis("off")
    
    output_path = f"plots/hasse_n5_{hash_id}.png"
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Hasse structure rendered successfully! View image path: {output_path}")

if __name__ == "__main__":
    plot_hasse_diagram()
