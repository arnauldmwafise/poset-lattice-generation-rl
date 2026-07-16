import pytest
import torch
import numpy as np
from src.models import GeneralDAGPolicy, all_pairs

def test_policy_dimensions_and_device():
    """
    Verifies that the policy correctly builds batch outputs with expected
    tensor shapes matching the combinatorics of pair choices C(n, 2).
    """
    n = 6
    batch_size = 4
    device = "cpu"
    
    policy = GeneralDAGPolicy(hidden=16)
    pair_order = all_pairs(n)
    num_pairs = len(pair_order)  # Should equal 15 for n=6
    
    adj, log_probs, values, masks, entropies, lower_tri, actions = policy.generate_batch(
        n=n, batch_size=batch_size, device=device, pair_order=pair_order
    )
    
    # Assert structural tensor geometry
    assert adj.shape == (batch_size, n, n)
    assert log_probs.shape == (batch_size, num_pairs)
    assert values.shape == (batch_size, num_pairs)
    assert masks.shape == (batch_size, num_pairs)
    assert entropies.shape == (batch_size, num_pairs)
    assert lower_tri.shape == (batch_size, n, n)
    assert actions.shape == (batch_size, num_pairs)

def test_incremental_transitive_closure_dag_property():
    """
    Validates that the policy's incremental Warshall-like update engine 
    preserves asymmetry. The resulting lower triangular matrix representation
    must never contain structural cycles.
    """
    n = 10
    batch_size = 8
    policy = GeneralDAGPolicy(hidden=16)
    
    _, _, _, _, _, lower_tri, _ = policy.generate_batch(n=n, batch_size=batch_size)
    
    lower_tri_np = lower_tri.cpu().numpy()
    
    for b in range(batch_size):
        mat = lower_tri_np[b]
        # Check reflexivity along the main diagonal
        for i in range(n):
            assert mat[i, i] == 1.0, f"Element ({i},{i}) must be reflexive."
            
        # Check strict antisymmetry: if i <= j and j <= i, then i == j.
        # Translated to relation matrix convention: mat[i, j] == 1 means j <= i.
        for i in range(n):
            for j in range(n):
                if i != j:
                    # It is impossible for both i <= j and j <= i to coexist
                    assert not (mat[i, j] == 1.0 and mat[j, i] == 1.0), \
                        f"Cycle detected between node {i} and node {j} in batch {b}."

def test_fixed_action_replay_consistency():
    """
    Critical correctness test for the PPO importance sampling ratio:
    Passing 'fixed_actions' to the policy network must produce identical 
    action trajectories, but generates new parameter-dependent log-probabilities.
    """
    n = 5
    batch_size = 2
    policy = GeneralDAGPolicy(hidden=16)
    pair_order = all_pairs(n)
    
    # 1. Generate an initial baseline rollout trajectory
    outputs_1 = policy.generate_batch(n=n, batch_size=batch_size, pair_order=pair_order)
    actions_1 = outputs_1[-1]
    lower_tri_1 = outputs_1[5]
    
    # 2. Replay the exact actions through the model under identical parameters
    outputs_2 = policy.generate_batch(
        n=n, batch_size=batch_size, pair_order=pair_order, fixed_actions=actions_1
    )
    log_probs_2 = outputs_2[1]
    lower_tri_2 = outputs_2[5]
    actions_2 = outputs_2[-1]
    
    # 3. Assert exact deterministic replication of the discrete graph outputs
    torch.testing.assert_close(actions_1, actions_2, msg="Replayed actions diverged from baseline.")
    torch.testing.assert_close(lower_tri_1, lower_tri_2, msg="Resulting poset structure diverged from baseline.")
    torch.testing.assert_close(outputs_1[1], log_probs_2, msg="Log probabilities should match under frozen weights.")
