import numpy as np

def matrix_to_bitstring(mat_np: np.ndarray) -> str:
    """
    Serializes a binary adjacency matrix into a row-major bitstring representation.
    Ensures minimal storage footprint for large combinatoric search spaces.
    
    Time Complexity: O(n^2)
    """
    return "".join(str(int(v)) for v in mat_np.flatten())

def bitstring_to_matrix(bits: str, n: int) -> np.ndarray:
    """
    Deserializes a row-major bitstring back into an (n x n) binary matrix representation.
    
    Inverse operation of matrix_to_bitstring.
    """
    return np.array([int(c) for c in bits], dtype=int).reshape(n, n)
