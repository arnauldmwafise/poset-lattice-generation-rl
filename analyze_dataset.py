# analyze_dataset.py
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch  # <-- Add PyTorch import

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure the local 'src' directory is discoverable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import analyze_lattice_properties, batch_lattice_diagnostics_detailed
    from src.data_utils import bitstring_to_matrix  # Assuming this utility exists per your code
except ImportError as e:
    logger.error("Failed to import required modules from the src package.")
    logger.error(f"Error details: {e}")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze lattice structural properties and run detailed batch diagnostics."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input CSV dataset containing matrix bitstrings"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="lattice_analysis_results.csv", 
        help="Path to save the evaluated properties dataset"
    )
    parser.add_argument(
        "--run-diagnostics",
        action="store_true",
        help="If set, executes detailed batch lattice diagnostics alongside basic checks"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found at: '{args.input}'")
        sys.exit(1)

    logger.info(f"Loading matrix dataset from: {args.input}")
    try:
        # Force all columns to read as strings to keep massive integers safe from float overflow bugs.
        df = pd.read_csv(args.input, dtype=str)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    if df.empty or "matrix_bits" not in df.columns or "n" not in df.columns:
        logger.error("Dataset must contain 'matrix_bits' and 'n' columns.")
        sys.exit(1)

    logger.info("Evaluating axiomatic lattice properties across all rows...")
    
    modular_flags = []
    semimodular_flags = []
    distributive_flags = []
    matrices_list = []
    sizes_list = []

    # Process each structure in the dataset
    for _, row in df.iterrows():
        try:
            # Safe conversion: Python handles arbitrarily large integers natively
            n = int(row["n"])
            matrix = bitstring_to_matrix(row["matrix_bits"], n)
            
            # Keep matrices for batch diagnostics if needed
            matrices_list.append(matrix)
            sizes_list.append(n)
            
            # Execute the function from src/classifier.py
            props = analyze_lattice_properties(matrix, n)
            
            modular_flags.append(props.get("modular", False))
            semimodular_flags.append(props.get("semimodular", False))
            distributive_flags.append(props.get("distributive", False))
        except Exception as e:
            logger.warning(f"Skipping an invalid row representation due to error: {e}")
            modular_flags.append(None)
            semimodular_flags.append(None)
            distributive_flags.append(None)
            matrices_list.append(None)
            sizes_list.append(None)

    # Append basic structural properties back into the dataframe
    df["is_modular"] = modular_flags
    df["is_semimodular"] = semimodular_flags
    df["is_distributive"] = distributive_flags

    # Conditional Execution of Detailed Diagnostics Suite
    if args.run_diagnostics:
        logger.info("Running batch_lattice_diagnostics_detailed pipeline...")
        try:
            # Filter out any rows that failed matrix extraction
            valid_matrices = [m for m in matrices_list if m is not None]
            
            if valid_matrices:
                # 1. Convert to a standard numpy array first
                valid_matrices_np = np.array(valid_matrices)
                
                # 2. Convert to an actual PyTorch tensor. 
                # This naturally satisfies .shape, .float(), .device, and internal dimension transpositions!
                patched_tensor = torch.from_numpy(valid_matrices_np)
                
                # Execute the diagnostics tool
                diagnostic_results = batch_lattice_diagnostics_detailed(patched_tensor)
                
                # Merge diagnostic metrics into your output dataframe
                if isinstance(diagnostic_results, pd.DataFrame):
                    df = pd.concat([df, diagnostic_results], axis=1)
                elif isinstance(diagnostic_results, dict):
                    for key, values in diagnostic_results.items():
                        df[f"diag_{key}"] = values
                logger.info("Detailed diagnostics metrics merged successfully.")
            else:
                logger.warning("No valid matrices found to run diagnostics on.")
        except Exception as e:
            logger.error(f"Detailed batch diagnostics pipeline failed: {e}")

    try:
        df.to_csv(args.output, index=False)
        logger.info(f"Analysis complete! Saved verified properties to: {args.output}")
    except Exception as e:
        logger.error(f"Failed to save output results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
