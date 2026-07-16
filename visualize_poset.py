import argparse
import csv
import os
import matplotlib.pyplot as plt
from src.data_utils import bitstring_to_matrix
from src.visualization import render_with_matplotlib, export_to_graphviz_dot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Discovered Order-Theoretic Structures")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to outputs_*.csv file")
    parser.add_argument("--row_index", type=int, default=0, help="Row index of the matrix to visualize")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save visual files")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the target logged parameters
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if not rows or args.row_index >= len(rows):
        print(f"Error: Selected row index {args.row_index} invalid or target CSV empty.")
        exit(1)
        
    target_row = rows[args.row_index]
    n = int(target_row["n"])
    target_type = target_row.get("target", "Structure")
    matrix_bits = target_row["matrix_bits"]
    structure_hash = target_row["hash"]
    
    print(f"Unpacking hash {structure_hash} ({target_type}, n={n})...")
    matrix = bitstring_to_matrix(matrix_bits, n)
    
    # 1. Matplotlib Plot Generation
    title_str = f"Hasse Diagram: Discovered {target_type.capitalize()} (n={n})"
    fig = render_with_matplotlib(matrix, title=title_str)
    png_path = os.path.join(args.output_dir, f"poset_{structure_hash}.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Matplotlib image saved to: {png_path}")
    
    # 2. Graphviz DOT Notation Export
    dot_path = os.path.join(args.output_dir, f"poset_{structure_hash}.dot")
    export_to_graphviz_dot(matrix, dot_path, graph_name=f"Hash_{structure_hash}")
    print(f"Graphviz DOT structural map written to: {dot_path}")
