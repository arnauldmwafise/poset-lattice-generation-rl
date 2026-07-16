import pytest
import numpy as np
import torch
from src.diagnostics import batch_lattice_diagnostics_detailed, check_boolean_lattice

def test_identity_is_not_lattice():
    """An identity matrix has no global top or bottom bounds, so it must fail lattice properties."""
    mat = torch.eye(4).unsqueeze(0)  # Shape (1, 4, 4)
    is_lattice, _, _, _, _ = batch_lattice_diagnostics_detailed(mat)
    assert is_lattice.item() is False

def test_boolean_algebra_four_elements():
    """
    Validates properties against a known 4-element Boolean Algebra lattice (2^2).
    Convention: M[i, j] == 1 implies element j is less than or equal to element i (j <= i).
    
    Elements representation (Power Set of {a, b}):
      0: Bottom Element {}
      1: Atom 1 {a}
      2: Atom 2 {b}
      3: Top Element {a, b}
    """
    # Expanded 4x4 matrix with explicit row-by-row lists
    row0 = [1, 0, 0, 0]  # Element 0: only 0 <= 0
    row1 = [1, 1, 0, 0]  # Element 1: 0 <= 1, 1 <= 1
    row2 = [1, 0, 1, 0]  # Element 2: 0 <= 2, 2 <= 2
    row3 = [1, 1, 1, 1]  # Element 3: 0, 1, 2, 3 <= 3
    
    mat_np = np.array([row0, row1, row2, row3], dtype=int)
    
    is_bool, dist, comp = check_boolean_lattice(mat_np, n=4)
    assert is_bool is True
    assert dist == 1.0
    assert comp == 1.0
