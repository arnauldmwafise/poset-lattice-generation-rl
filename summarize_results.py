# summarize_results.py
import pandas as pd

def analyze_distributions(file_path="lattice_analysis_results.csv"):
    print(f"Reading generated results from: {file_path}")
    
    # Load completely as string to preserve giant hash formats safely
    df = pd.read_csv(file_path, dtype=str)
    
    # Map text flags directly back into explicit booleans
    bool_map = {'True': True, 'False': False}
    df['is_modular'] = df['is_modular'].map(bool_map)
    df['is_semimodular'] = df['is_semimodular'].map(bool_map)
    df['is_distributive'] = df['is_distributive'].map(bool_map)
    
    total_structures = len(df)
    
    # Calculate raw counts
    distributive_count = df['is_distributive'].sum()
    modular_count = df['is_modular'].sum()
    semimodular_count = df['is_semimodular'].sum()
    
    # Check for anomaly/rare intermediate variants
    # Semimodular but NOT Modular (e.g., Geometric Lattices)
    strictly_semimodular = df[df['is_semimodular'] & ~df['is_modular']]
    # Modular but NOT Distributive (e.g., Diamond Lattices M3)
    strictly_modular = df[df['is_modular'] & ~df['is_distributive']]
    
    print("\n" + "="*40)
    print("        LATTICE PROPERTY SUITE REPORT      ")
    print("="*40)
    print(f"Total Matrices Evaluated : {total_structures}")
    print(f"Distributive Lattices   : {distributive_count} ({distributive_count/total_structures*100:.2f}%)")
    print(f"Modular Lattices        : {modular_count} ({modular_count/total_structures*100:.2f}%)")
    print(f"Semimodular Lattices     : {semimodular_count} ({semimodular_count/total_structures*100:.2f}%)")
    print("-"*40)
    print(f"Rare Non-Modular Semimodular Found: {len(strictly_semimodular)}")
    print(f"Rare Non-Distributive Modular Found: {len(strictly_modular)}")
    print("="*40)

    if len(strictly_modular) > 0:
        print("\nSample Hash of rare Modular (Non-Distributive) structure:")
        print(strictly_modular['hash'].iloc[0])

if __name__ == "__main__":
    analyze_distributions()
