# extract_rare_lattices.py
import pandas as pd

def isolate_rare():
    # Read raw dataset safely as strings
    df = pd.read_csv("lattice_analysis_results.csv", dtype=str)
    
    # Isolate the rare modular, non-distributive rows
    rare_df = df[(df['is_modular'] == 'True') & (df['is_distributive'] == 'False')]
    
    # Save them to a specialized subset file
    output_file = "rare_modular_non_distributive.csv"
    rare_df.to_csv(output_file, index=False)
    
    print(f"Successfully extracted {len(rare_df)} rare structures to {output_file}!")
    print("\nExtracted Row Profiles:")
    for idx, row in rare_df.iterrows():
        print(f"Hash: {row['hash']} | Target Type: {row['target']} | Bits Preview: {row['matrix_bits'][:30]}...")

if __name__ == "__main__":
    isolate_rare()
