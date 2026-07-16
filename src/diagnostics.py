import numpy as np
import torch

def _extremal_counts(bound_source: torch.Tensor, path_source: torch.Tensor) -> torch.Tensor:
    """
    Evaluates reachability predicates over elements to compute unique bounds.
    
    Computes intersection masks over the transitively closed relation matrix to count 
    the elements satisfying the criteria for being a supremum (LUB) or infimum (GLB).
    
    Tensor Dimensions:
        bound_source: (B, n, n)
        path_source: (B, n, n)
        Returns: (B, n, n) representing match counts per element pair.
    """
    B, n, _ = bound_source.shape
    common_bounds = bound_source.unsqueeze(2) * bound_source.unsqueeze(1)  # (B, n, n, n)
    bound_is = common_bounds.unsqueeze(-1)                                 # (B, n, n, n, 1)
    path = path_source.view(B, 1, 1, n, n)                                 # (B, 1, 1, n, n)
    implication = (bound_is == 0) | (path == 1)
    is_extreme_candidate = implication.all(dim=3)
    is_extreme = is_extreme_candidate & common_bounds.bool()
    return is_extreme.sum(dim=-1)

def batch_lattice_diagnostics_detailed(lower_tri_matrices):
    """
    Computes algebraic properties of posets defined by lower triangular matrices.
    Verifies existence of unique joins and meets across all pairs in parallel.
    
    Mathematical Formulation:
        Let P be a poset. P is a lattice iff for all x, y in P:

        |LUB(x, y)| = 1 (Join) AND |GLB(x, y)| = 1 (Meet).
    """
    M = lower_tri_matrices.float()
    B, n, _ = M.shape
    device = M.device
    Mt = M.transpose(1, 2)
    
    join_counts = _extremal_counts(bound_source=Mt, path_source=M)
    meet_counts = _extremal_counts(bound_source=M, path_source=Mt)
    
    pair_mask = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).repeat(B, 1, 1)
    join_ok = (join_counts == 1) & pair_mask
    meet_ok = (meet_counts == 1) & pair_mask
    num_pairs = pair_mask[0].sum()
    
    join_frac = join_ok.sum(dim=(1, 2)).float() / num_pairs
    meet_frac = meet_ok.sum(dim=(1, 2)).float() / num_pairs
    is_lattice = (join_frac >= 1.0) & (meet_frac >= 1.0)
    
    return is_lattice, join_ok, meet_ok, join_frac, meet_frac

def _build_join_meet_tables(R: np.ndarray, n: int):
    """
    Constructs the operational join (v) and meet (^) Cayley tables for a finite poset.
    Requires that the underlying structure is verified to be a lattice.
    """
    join_table = np.full((n, n), -1, dtype=int)
    meet_table = np.full((n, n), -1, dtype=int)
    for x in range(n):
        for y in range(n):
            ubs = [z for z in range(n) if R[x, z] and R[y, z]]
            lub = [u for u in ubs if all(R[u, z] for z in ubs)]
            if len(lub) != 1:
                return None, None
            join_table[x, y] = lub[0]
            lbs = [z for z in range(n) if R[z, x] and R[z, y]]
            glb = [g for g in lbs if all(R[z, g] for z in lbs)]
            if len(glb) != 1:
                return None, None
            meet_table[x, y] = glb[0]
    return join_table, meet_table

def check_boolean_lattice(lower_tri_np: np.ndarray, n: int):
    """
    Validates if a lattice structure represents a Boolean Algebra.
    
    A lattice is Boolean iff it is:
        1. Complemented: For all x, exists unique y such that x ^ y = 0 and x v y = 1.
        2. Distributive: For all x, y, z: x ^ (y v z) = (x ^ y) v (x ^ z).
    
    Returns fractions for use as numerical reward signals during policy shaping.
    """
    R = lower_tri_np.T  
    top_c = [t for t in range(n) if R[:, t].sum() == n]
    bot_c = [z for z in range(n) if R[z, :].sum() == n]
    if len(top_c) != 1 or len(bot_c) != 1:
        return False, 0.0, 0.0
    top, bottom = top_c[0], bot_c[0]

    join_table, meet_table = _build_join_meet_tables(R, n)
    if join_table is None:
        return False, 0.0, 0.0

    correct, total = 0, n * n * n
    for a in range(n):
        for b in range(n):
            for c in range(n):
                lhs = meet_table[a, join_table[b, c]]
                rhs = join_table[meet_table[a, b], meet_table[a, c]]
                if lhs == rhs:
                    correct += 1
    dist_frac = correct / total

    comp_correct = 0
    for x in range(n):
        comps = [y for y in range(n) if join_table[x, y] == top and meet_table[x, y] == bottom]
        if len(comps) == 1:
            comp_correct += 1
    comp_frac = comp_correct / n

    return (dist_frac >= 1.0 and comp_frac >= 1.0), dist_frac, comp_frac

def is_valid_dag(R_bool_np):
    """
    Enforces the antisymmetry constraint of a partial order relation matrix.
    Ensures that no directed cycles exist within the evaluated structure.
    """
    n = R_bool_np.shape[0]
    for a in range(n):
        for b in range(n):
            if a != b and R_bool_np[a, b] and R_bool_np[b, a]:
                return False
    return True
