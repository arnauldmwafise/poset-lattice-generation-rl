import numpy as np
from src.diagnostics import _build_join_meet_tables

def analyze_lattice_properties(lower_tri_np: np.ndarray, n: int) -> dict:
    """
    Performs axiomatic verification over a validated lattice structure.
    
    Axiomatic Criteria Evaluated:
      1. Modularity: x <= b implies x v (a ^ b) == (x v a) ^ b
      2. Upper Semimodularity: x covers (x ^ y) implies (x v y) covers y
      3. Distributivity: a ^ (b v c) == (a ^ b) v (a ^ c)
    """
    R = lower_tri_np.T  # R[a, b] == 1 implies a <= b
    join_table, meet_table = _build_join_meet_tables(R, n)
    
    # Fallback default flags if structure is not a perfect lattice
    properties = {"modular": False, "semimodular": False, "distributive": False}
    if join_table is None or meet_table is None:
        return properties

    # 1. Evaluate Distributivity (O(n^3))
    is_distributive = True
    for a in range(n):
        for b in range(n):
            for c in range(n):
                lhs = meet_table[a, join_table[b, c]]
                rhs = join_table[meet_table[a, b], meet_table[a, c]]
                if lhs != rhs:
                    is_distributive = False
                    break
            if not is_distributive: break
        if not is_distributive: break
    properties["distributive"] = is_distributive

    # Distributivity strictly implies Modularity (Lattice Theory fundamental theorem)
    if is_distributive:
        properties["modular"] = True
        properties["semimodular"] = True
        return properties

    # 2. Evaluate Modularity (O(n^3))
    is_modular = True
    for x in range(n):
        for b in range(n):
            if R[x, b]:  # Condition: x <= b
                for a in range(n):
                    lhs = join_table[x, meet_table[a, b]]
                    rhs = meet_table[join_table[x, a], b]
                    if lhs != rhs:
                        is_modular = False
                        break
            if not is_modular: break
        if not is_modular: break
    properties["modular"] = is_modular

    # 3. Evaluate Upper Semimodularity (Using the covering relationship)
    # Step 3a: Extract the covering relation (Transitive Reduction)
    strict_order = R.copy()
    np.fill_diagonal(strict_order, 0)
    covering = strict_order.copy()
    for w in range(n):
        for u in range(n):
            for v in range(n):
                if strict_order[u, w] and strict_order[w, v]:
                    covering[u, v] = 0
                    
    # Step 3b: Verify upper covering condition
    is_semimodular = True
    for x in range(n):
        for y in range(n):
            meet_xy = meet_table[x, y]
            if covering[meet_xy, x]:  # Does x cover (x ^ y)?
                join_xy = join_table[x, y]
                if not covering[y, join_xy]:  # Then (x v y) must cover y
                    is_semimodular = False
                    break
        if not is_semimodular: break
    properties["semimodular"] = is_semimodular

    return properties
