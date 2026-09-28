# plot_convergence.py
import os
import glob
import json
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def parse_and_plot_convergence():
    # Scan for history targets dynamically matching any dimensionality configuration array
    history_files = glob.glob("outputs/history_n*_*_v5.json")
    if not history_files:
        print("[!] No training history JSON logs found inside outputs/ container.")
        return

    os.makedirs('plots', exist_ok=True)

    for path in history_files:
        filename = os.path.basename(path)
        # Parse parameter tokens directly from string markers
        tokens = filename.split('_')
        n_val = tokens[1]
        target_name = tokens[2]

        print(f"[*] Processing learning convergence profiles for {n_val.upper()} | TARGET: {target_name.upper()}...")
        
        with open(path, 'r') as f:
            data = json.load(f)

        if "iter" not in data or "avg_reward" not in data:
            continue

        iterations = data["iter"]
        rewards = data["avg_reward"]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.plot(iterations, rewards, color='#d62728', linewidth=1.5, label='Rolling Policy Objective')
        
        ax.set_xlabel('Optimization Iterations Step Count', fontweight='bold')
        ax.set_ylabel('Mean Trajectory Policy Reward Landscape', fontweight='bold')
        ax.set_title(f'PPO Learning Trajectory Convergence ({n_val.upper()} | {target_name.upper()})', fontweight='bold', pad=12)
        ax.grid(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        output_plot = f"plots/convergence_{n_val}_{target_name}.pdf"
        plt.savefig(output_plot, format='pdf', dpi=300)
        plt.close()
        print(f" -> Generated verification plot: {output_plot}")

if __name__ == "__main__":
    parse_and_plot_convergence()
