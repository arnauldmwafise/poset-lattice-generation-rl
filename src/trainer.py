# src/trainer.py
import os
import time
import json
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging

from src.models import GeneralDAGPolicy, all_pairs
from src.diagnostics import batch_lattice_diagnostics_detailed, check_boolean_lattice
from src.ppo import compute_gae
from src.data_utils import matrix_to_bitstring

# Configure formal scientific logging structure
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_log(n=20, total_episodes=4500, batch_size=32, ppo_epochs=4, lr=2e-4,
                   ema_decay=0.95, pair_weight=2.5, output_dir=".", target="lattice",
                   dist_shape_weight=2.0, comp_shape_weight=15.0, max_grad_norm=0.5):
    """
    Main training execution system optimizing over structured discrete search bounds.
    Utilizes dense reward shaping based on individual pair connection failure rates.
    """
    logging.info(f"Starting execution of train_and_log workflow for target: '{target}'")
    assert target in ("lattice", "join", "meet", "dual", "boolean"), "Invalid target property requested."

    if target == "boolean" and (n & (n - 1) != 0):
        logging.warning(f"Target 'boolean' requested for non-power-of-2 n={n}. Properties structurally impossible.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Target computational device allocated: {device}")
    
    policy = GeneralDAGPolicy(hidden=32).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    seen_hashes = set()  

    logging.info(f"Creating output target directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, f"history_n{n}_{target}_v5.json")

    # Tracking sets for discrete semilattice records
    seen_join_semi_hashes = set()
    seen_meet_semi_hashes = set()
    num_join_semi_confirmed = 0
    num_meet_semi_confirmed = 0

    if target == "dual":
        confirmed_join_hashes, confirmed_meet_hashes = set(), set()
        num_confirmed_join, num_confirmed_meet = 0, 0
        join_csv_file = open(os.path.join(output_dir, f"outputs_n{n}_join_v5.csv"), "w", newline="")
        meet_csv_file = open(os.path.join(output_dir, f"outputs_n{n}_meet_v5.csv"), "w", newline="")
        join_writer, meet_writer = csv.writer(join_csv_file), csv.writer(meet_csv_file)
        join_writer.writerow(["hash", "n", "target", "matrix_bits"])
        meet_writer.writerow(["hash", "n", "target", "matrix_bits"])
    else:
        confirmed_hashes = set()
        num_confirmed = 0
        csv_file = open(os.path.join(output_dir, f"outputs_n{n}_{target}_v5.csv"), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["hash", "n", "target", "matrix_bits"])
        
        # New streaming containers for isolated semi-lattices
        join_semi_file = open(os.path.join(output_dir, f"outputs_n{n}_join_semilattice_v5.csv"), "w", newline="")
        meet_semi_file = open(os.path.join(output_dir, f"outputs_n{n}_meet_semilattice_v5.csv"), "w", newline="")
        join_semi_writer = csv.writer(join_semi_file)
        meet_semi_writer = csv.writer(meet_semi_file)
        join_semi_writer.writerow(["hash", "n", "target", "matrix_bits"])
        meet_semi_writer.writerow(["hash", "n", "target", "matrix_bits"])

    pair_fail_join = torch.ones(n, n, device=device)
    pair_fail_meet = torch.ones(n, n, device=device)
    pmask = ~torch.eye(n, dtype=torch.bool, device=device)

    if target == "dual":
        history = {"iter": [], "join_in_batch": [], "meet_in_batch": [], "join_frac": [], "meet_frac": [],
                   "avg_reward": [], "confirmed_join_total": [], "confirmed_meet_total": [], "gap1_fail": []}
    else:
        history = {"iter": [], "confirmed_in_batch": [], "join_frac": [], "meet_frac": [],
                   "avg_reward": [], "confirmed_total": [], "gap1_fail": [],
                   "join_semilattices_in_batch": [], "meet_semilattices_in_batch": []}

    iterations = total_episodes // batch_size
    logging.info(f"Total optimization loops calculated: {iterations} iterations.")
    t0 = time.time()
    
    for it in range(iterations):
        progress_frac = it / iterations
        adaptive_entropy = 0.01 + 0.04 * (1.0 - progress_frac)

        pair_order = all_pairs(n)
        random.shuffle(pair_order)

        with torch.no_grad():
            adj_b, old_lp, values, masks, old_ent, lower_tri_b, actions_b = policy.generate_batch(n, batch_size, device, pair_order)
            is_lattice, join_ok, meet_ok, join_frac, meet_frac = batch_lattice_diagnostics_detailed(lower_tri_b)

        if target == "lattice": is_target = is_lattice
        elif target == "join": is_target = join_frac >= 1.0
        elif target == "meet": is_target = meet_frac >= 1.0
        elif target == "boolean": is_target = is_lattice
        else:
            is_join_b, is_meet_b = join_frac >= 1.0, meet_frac >= 1.0

        pair_fail_join = ema_decay * pair_fail_join + (1 - ema_decay) * (1.0 - join_ok.float().mean(dim=0))
        pair_fail_meet = ema_decay * pair_fail_meet + (1 - ema_decay) * (1.0 - meet_ok.float().mean(dim=0))

        # Vector validation shortcuts
        is_join_semi_b = (join_ok.float().mean(dim=(1, 2)) >= 1.0)
        is_meet_semi_b = (meet_ok.float().mean(dim=(1, 2)) >= 1.0)

        rewards = []
        for b in range(batch_size):
            adj_np = adj_b[b].cpu().numpy().astype(int)
            g_hash = hash(adj_np.tobytes())
            edge_count = adj_np.sum()
            r = 2.0 if g_hash not in seen_hashes else 0.2
            seen_hashes.add(g_hash)

            jb, mb = join_ok[b], meet_ok[b]
            join_term = torch.where(jb, pair_fail_join, -pair_fail_join)[pmask].sum()
            meet_term = torch.where(mb, pair_fail_meet, -pair_fail_meet)[pmask].sum()
            max_edges = (n * (n - 1)) // 2

            # Stream unique semi-lattices out to independent logs safely
            matrix_bits_str = matrix_to_bitstring(lower_tri_b[b].cpu().numpy().astype(int))
            if target != "dual":
                if is_join_semi_b[b].item() and g_hash not in seen_join_semi_hashes:
                    seen_join_semi_hashes.add(g_hash)
                    join_semi_writer.writerow([g_hash, n, "join_semilattice", matrix_bits_str])
                if is_meet_semi_b[b].item() and g_hash not in seen_meet_semi_hashes:
                    seen_meet_semi_hashes.add(g_hash)
                    meet_semi_writer.writerow([g_hash, n, "meet_semilattice", matrix_bits_str])

            if target == "lattice":
                r += pair_weight * (join_term + meet_term).item() / (2 * n)
                if is_target[b].item():
                    r += 20.0
                    try:
                        matrix_np = lower_tri_b[b].cpu().numpy().astype(int)
                        from src.classifier import analyze_lattice_properties
                        props = analyze_lattice_properties(matrix_np, n)
                        if props.get("distributive", False): r -= 5.0  
                        elif props.get("modular", False) and not props.get("distributive", False):
                            r += 30.0  
                            print(f" [!] RL Agent breakthrough: Discovered rare M3 Diamond Lattice structure!")
                        elif props.get("semimodular", False) and not props.get("modular", False):
                            r += 60.0  
                            print(f" [!] RL Agent breakthrough: Unlocked missing Non-Modular Semimodular space!")
                    except Exception: pass  

                    if g_hash not in confirmed_hashes:
                        confirmed_hashes.add(g_hash)
                        num_confirmed += 1
                        csv_writer.writerow([g_hash, n, target, matrix_bits_str])
                else:
                    if edge_count >= (max_edges - n): r -= 5.0
            elif target == "join":
                r += pair_weight * join_term.item() / n
                if is_target[b].item():
                    r += 20.0
                    if g_hash not in confirmed_hashes:
                        confirmed_hashes.add(g_hash)
                        num_confirmed += 1
                        csv_writer.writerow([g_hash, n, target, matrix_bits_str])
                else:
                    if edge_count >= (max_edges - n): r -= 5.0
            elif target == "meet":
                r += pair_weight * meet_term.item() / n
                if is_target[b].item():
                    r += 20.0
                    if g_hash not in confirmed_hashes:
                        confirmed_hashes.add(g_hash)
                        num_confirmed += 1
                        csv_writer.writerow([g_hash, n, target, matrix_to_bitstring(lower_tri_b[b].cpu().numpy().astype(int))])
                else:
                    if edge_count >= (max_edges - n): r -= 5.0

            elif target == "boolean":
                r += pair_weight * (join_term + meet_term).item() / (2 * n)
                if is_target[b].item():
                    is_bool, dist_frac, comp_frac = check_boolean_lattice(lower_tri_b[b].cpu().numpy().astype(int), n)
                    r += dist_shape_weight * dist_frac + comp_shape_weight * max(0.0, comp_frac - 2.0/n)
                    if is_bool:
                        r += 30.0
                        if g_hash not in confirmed_hashes:
                            confirmed_hashes.add(g_hash)
                            num_confirmed += 1
                            csv_writer.writerow([g_hash, n, target, matrix_to_bitstring(lower_tri_b[b].cpu().numpy().astype(int))])
                else:
                    if edge_count >= (max_edges - n): r -= 5.0

            else:  # "dual"
                r += pair_weight * (join_term + meet_term).item() / (2 * n)
                lower_tri_np = None
                if is_join_b[b].item():
                    r += 10.0
                    if g_hash not in confirmed_join_hashes:
                        confirmed_join_hashes.add(g_hash)
                        num_confirmed_join += 1
                        lower_tri_np = lower_tri_b[b].cpu().numpy().astype(int)
                        join_writer.writerow([g_hash, n, "join", matrix_to_bitstring(lower_tri_np)])
                if is_meet_b[b].item():
                    r += 10.0
                    if g_hash not in confirmed_meet_hashes:
                        confirmed_meet_hashes.add(g_hash)
                        num_confirmed_meet += 1
                        if lower_tri_np is None:
                            lower_tri_np = lower_tri_b[b].cpu().numpy().astype(int)
                        meet_writer.writerow([g_hash, n, "meet", matrix_to_bitstring(lower_tri_np)])
                if not is_join_b[b].item() and not is_meet_b[b].item():
                    if edge_count >= (max_edges - n): r -= 5.0

            r += 0.02 * edge_count
            rewards.append(r)

        rewards_raw = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages, returns = compute_gae(
            rewards_raw.unsqueeze(1).repeat(1, values.size(1)), 
            values, 
            masks,
            entropy_regularizer=adaptive_entropy
        )
        
        if masks.any():
            advantages = (advantages - advantages[masks].mean()) / (advantages[masks].std() + 1e-8)

        for _ in range(ppo_epochs):
            _, new_lp, new_values, _, new_ent, _, _ = policy.generate_batch(n, batch_size, device, pair_order, fixed_actions=actions_b)
            ratios = torch.exp(new_lp - old_lp)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            actor_loss = -torch.where(masks, torch.min(surr1, surr2), torch.zeros_like(surr1)).mean()
            critic_loss = torch.where(masks, F.mse_loss(new_values, returns, reduction='none'), torch.zeros_like(new_values)).mean()
            ent_loss = -torch.where(masks, new_ent, torch.zeros_like(new_ent)).mean()
            
            optimizer.zero_grad()
            (actor_loss + 0.5 * critic_loss + 0.02 * ent_loss).backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

        gap1 = np.mean([pair_fail_join[i, i+1].item() for i in range(n-1)] + [pair_fail_meet[i, i+1].item() for i in range(n-1)])
        history["iter"].append(it + 1)
        history["join_frac"].append(join_frac.mean().item())
        history["meet_frac"].append(meet_frac.mean().item())
        history["avg_reward"].append(float(np.mean(rewards)))
        history["gap1_fail"].append(float(gap1))

        if target == "dual":
            num_join_b, num_meet_b = is_join_b.sum().item(), is_meet_b.sum().item()
            history["join_in_batch"].append(num_join_b)
            history["meet_in_batch"].append(num_meet_b)
            history["confirmed_join_total"].append(num_confirmed_join)
            history["confirmed_meet_total"].append(num_confirmed_meet)
            if (it + 1) % 20 == 0 or it == 0:
                print(f"it {it+1:3d}/{iterations} | [dual] join={num_join_b:2d}/{batch_size} "
                      f"meet={num_meet_b:2d}/{batch_size} | reward={np.mean(rewards):.2f} | "
                      f"confirmed join={num_confirmed_join} meet={num_confirmed_meet} | "
                      f"gap1_fail={gap1:.3f} | elapsed={time.time()-t0:.1f}s")
        else:
            num_target = is_target.sum().item()
            num_join_semi = is_join_semi_b.sum().item()
            num_meet_semi = is_meet_semi_b.sum().item()
            
            history["confirmed_in_batch"].append(num_target)
            history["confirmed_total"].append(num_confirmed)
            history["join_semilattices_in_batch"].append(num_join_semi)
            history["meet_semilattices_in_batch"].append(num_meet_semi)
            
            # --- CLEAN VISUAL UPDATE: TARGET-AWARE CONSOLE PRINTING ---
            display_label = target
            if target == "join": display_label = "join_semilattice"
            if target == "meet": display_label = "meet_semilattice"
            
            if (it + 1) % 20 == 0 or it == 0:
                print(f"it {it+1:3d}/{iterations} | [{display_label}] found={num_target:2d}/{batch_size} | "
                      f"join_semi={num_join_semi:2d}/{batch_size} meet_semi={num_meet_semi:2d}/{batch_size} | "
                      f"reward={np.mean(rewards):.2f} | confirmed={num_confirmed} | "
                      f"gap1_fail={gap1:.3f} | elapsed={time.time()-t0:.1f}s")

        if (it + 1) % 25 == 0:
            json.dump(history, open(history_path, "w"))
            if target == "dual":
                join_csv_file.flush()
                meet_csv_file.flush()
            else:
                csv_file.flush()
                join_semi_file.flush()
                meet_semi_file.flush()

    json.dump(history, open(history_path, "w"))
    if target == "dual":
        join_csv_file.close()
        meet_csv_file.close()
    else:
        csv_file.close()
        join_semi_file.close()
        meet_semi_file.close()
        
    logging.info(f"Completed optimization sequence successfully in {time.time()-t0:.2f}s")
    return policy, history

