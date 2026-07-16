import argparse
import json
import logging
from src.trainer import train_and_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-based Combinatorial Latice Generation Software")
    parser.add_argument("--config", type=str, default=None, help="Path to custom configuration JSON")
    args = parser.parse_args()

    # Load defaults
    params = {
        "n": 20, "total_episodes": 4500, "batch_size": 32, "ppo_epochs": 4, 
        "lr": 2e-4, "ema_decay": 0.95, "pair_weight": 2.5, "target": "lattice", 
        "output_dir": "./outputs"
    }
    
    if args.config:
        with open(args.config, "r") as f:
            params.update(json.load(f))

    logging.info(f"Initializing run. Target structure: {params['target']}, Size n: {params['n']}")
    train_and_log(**params)
