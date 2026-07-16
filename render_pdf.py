import csv
import os
from src.data_utils import bitstring_to_matrix
from src.visualization import render_vector_pdf

csv_path = "./outputs/outputs_n20_lattice_v5.csv"
output_pdf_path = "./plots/poset_20_nodes_native.pdf"

os.makedirs("./plots", exist_ok=True)

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

target_row = rows[0]
n = int(target_row["n"])
matrix = bitstring_to_matrix(target_row["matrix_bits"], n)

print(f"Rendering vector lattice layout for hash {target_row['hash']}...")
render_vector_pdf(matrix, output_pdf_path, title=f"Hasse Diagram: Discovered Lattice (n={n})")
print(f"Success! Vector PDF generated at: {output_pdf_path}")
