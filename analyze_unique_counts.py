# analyze_unique_counts.py
import os
import glob
import pandas as pd

def audit_output_metrics():
    target_pattern = "outputs/*_v5_colab.csv"
    files = glob.glob(target_pattern)
    
    if not files:
        print("[!] No Colab outputs tracking files found inside the outputs/ directory.")
        return

    print("="*65)
    print("      DATASET UNIQUE QUANTITY AND DUPLICATION AUDIT REPORT")
    print("="*65)
    
    registry = {}
    
    for f in sorted(files):
        filename = os.path.basename(f)
        df = pd.read_csv(f, dtype=str)
        
        total_rows = len(df)
        unique_hashes = df['hash'].nunique() if 'hash' in df.columns else 0
        unique_bitstrings = df['matrix_bits'].nunique() if 'matrix_bits' in df.columns else 0
        
        if 'matrix_bits' in df.columns:
            registry[filename] = set(df['matrix_bits'].dropna().tolist())
        else:
            registry[filename] = set()
        
        print(f"File: {filename}")
        print(f" -> Total Sample Rows Written: {total_rows}")
        print(f" -> Unique Topological Hashes: {unique_hashes}")
        print(f" -> Unique Data Bitstrings  : {unique_bitstrings}")
        print("-" * 45)

    # Cross-compare intersection sets to track mathematical structural overlaps
    print("\n" + "="*65)
    print("      CROSS-TARGET PHENOTYPE OVERLAP ANALYSIS")
    print("="*65)
    
    filenames = list(registry.keys())
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            f1, f2 = filenames[i], filenames[j] # FIX: Corrected index referencing array structure
            
            # Extract matrix size metadata token (e.g., 'n30') to ensure consistent horizon bounds mapping
            n1 = [t for t in f1.split('_') if t.startswith('n')]
            n2 = [t for t in f2.split('_') if t.startswith('n')]
            
            if n1 and n2 and n1 == n2:
                overlap = len(registry[f1].intersection(registry[f2]))
                print(f"Intersection [{f1}] <--> [{f2}]: {overlap} shared shapes")

if __name__ == "__main__":
    audit_output_metrics()
