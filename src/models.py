import torch
import torch.nn as nn
import torch.nn.functional as F

def all_pairs(n):
    """Generates the upper triangular indices representing unique pair relations."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

class GeneralDAGPolicy(nn.Module):
    """
    Autoregressive neural model parameterized to generate valid Directed Acyclic Graphs.
    Instead of iterating linearly over node indices, it operates over a randomized sequence 
    of node pairs, applying sequential closure rules to enforce global consistency.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.pair_rnn = nn.GRUCell(hidden + 3, hidden)  
        self.pos_embed = nn.Embedding(4096, hidden)
        self.action_out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.value_out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def generate_batch(self, n, batch_size=32, device="cpu", pair_order=None, fixed_actions=None):
        """
        Samples or replays trajectories through the combinatorial action space of node pairs.
        
        Action Space:
            0: No edge
            1: Directed Edge u -> v
            2: Directed Edge v -> u
            
        Transitive Closure Constraint:
            Upon selecting an edge, incremental Warshall-like updates propagate updates 
            to the reachability matrix R to maintain legal DAG constraints dynamically.
        """
        if pair_order is None:
            import random
            pair_order = all_pairs(n)
            random.shuffle(pair_order)

        R = torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj = torch.zeros(batch_size, n, n, device=device)  
        h = torch.zeros(batch_size, self.hidden, device=device)
        prev_action_onehot = torch.zeros(batch_size, 3, device=device)

        log_probs, values_list, masks_list, entropies, actions_list = [], [], [], [], []

        for step_idx, (u, v) in enumerate(pair_order):
            pos = self.pos_embed(torch.tensor([u], device=device)) + self.pos_embed(torch.tensor([v], device=device))
            pos = pos.expand(batch_size, -1)
            h = self.pair_rnn(torch.cat([pos, prev_action_onehot], dim=-1), h)

            already_related = R[:, u, v] | R[:, v, u]   
            legal3 = torch.stack([torch.ones(batch_size, dtype=torch.bool, device=device),
                                   ~already_related, ~already_related], dim=-1)  

            logits = self.action_out(h)  
            logits = torch.where(legal3, logits, torch.tensor(-1e9, device=device))
            probs = F.softmax(logits, dim=-1)
            value = self.value_out(h).squeeze(-1)

            dist = torch.distributions.Categorical(probs=probs)
            if fixed_actions is None:
                action = dist.sample()  
            else:
                action = fixed_actions[:, step_idx]  
                
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values_list.append(value)
            masks_list.append(~already_related)  
            actions_list.append(action)

            onehot = F.one_hot(action, num_classes=3).float()
            prev_action_onehot = onehot

            fwd_mask = (action == 1)
            bwd_mask = (action == 2)

            if fwd_mask.any():
                adj[fwd_mask, u, v] = 1.0
                col_u = R[:, :, u].clone()
                col_u[:, u] = True
                row_v = R[:, v, :].clone()
                row_v[:, v] = True
                update = col_u.unsqueeze(2) & row_v.unsqueeze(1)  
                R = torch.where(fwd_mask.view(-1, 1, 1), R | update, R)

            if bwd_mask.any():
                adj[bwd_mask, v, u] = 1.0
                col_v = R[:, :, v].clone()
                col_v[:, v] = True
                row_u = R[:, u, :].clone()
                row_u[:, u] = True
                update = col_v.unsqueeze(2) & row_u.unsqueeze(1)
                R = torch.where(bwd_mask.view(-1, 1, 1), R | update, R)

        lower_tri = R.transpose(1, 2).float()  
        return (adj,
                torch.stack(log_probs, dim=1),
                torch.stack(values_list, dim=1),
                torch.stack(masks_list, dim=1),
                torch.stack(entropies, dim=1),
                lower_tri,
                torch.stack(actions_list, dim=1))
