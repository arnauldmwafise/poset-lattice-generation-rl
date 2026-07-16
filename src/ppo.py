import torch

def compute_gae(rewards_t, values_t, structural_masks, gamma=0.95, lam=0.90):
    """
    Generalized Advantage Estimation (GAE) over sequence decision paths.
    
    Computes exponentially discounted variance-reduced advantage estimators 
    while applying structural masking to prevent updates over unforced steps.
    """
    batch_size, num_steps = values_t.shape
    advantages = torch.zeros_like(values_t)
    last_gae_lam = torch.zeros(batch_size, device=values_t.device)
    next_value = torch.zeros(batch_size, device=values_t.device)

    for t in reversed(range(num_steps)):
        step_reward = rewards_t[:, t] if t == num_steps - 1 else torch.zeros(batch_size, device=values_t.device)
        delta = step_reward + gamma * next_value - values_t[:, t]
        mask = structural_masks[:, t].float()
        last_gae_lam = delta + gamma * lam * last_gae_lam * mask
        advantages[:, t] = last_gae_lam
        next_value = values_t[:, t]

    returns = advantages + values_t
    return advantages, returns
