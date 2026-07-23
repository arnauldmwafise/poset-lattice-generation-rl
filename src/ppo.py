# src/ppo.py
import torch

def compute_gae(rewards_t, values_t, structural_masks, gamma=0.95, lam=0.90, entropy_regularizer=0.01):
    """
    Generalized Advantage Estimation (GAE) with Continuous Entropy-Shaped Regularization.
    
    Breaks late-stage performance plateaus by distributing exploratory advantages 
    across intermediate decision paths, preventing premature policy convergence.
    """
    batch_size, num_steps = values_t.shape
    advantages = torch.zeros_like(values_t)
    last_gae_lam = torch.zeros(batch_size, device=values_t.device)
    next_value = torch.zeros(batch_size, device=values_t.device)

    for t in reversed(range(num_steps)):
        # 1. Capture the terminal episodic reward signal
        terminal_reward = rewards_t[:, t] if t == num_steps - 1 else torch.zeros(batch_size, device=values_t.device)
        
        # 2. FIX: Inject an intrinsic exploration step incentive.
        # This gives intermediate steps a dense reward signature based on structural activity,
        # ensuring the advantage gradient never completely drops to zero.
        mask = structural_masks[:, t].float()
        intrinsic_step_bonus = entropy_regularizer * mask
        
        total_step_reward = terminal_reward + intrinsic_step_bonus
        
        # 3. Compute temporal difference error with dense shaping signals
        delta = total_step_reward + gamma * next_value - values_t[:, t]
        
        # 4. Propagate discounted advantages back down the trajectory
        last_gae_lam = delta + gamma * lam * last_gae_lam * mask
        advantages[:, t] = last_gae_lam
        next_value = values_t[:, t]

    returns = advantages + values_t
    return advantages, returns
