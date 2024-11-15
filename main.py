import numpy as np
import torch
import json
import argparse
from src.FedAVG import FedAVG

np.random.seed(42)
torch.manual_seed(42)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='FedAvg Configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)
    fedavg = FedAVG(config)
    fedavg.run()

if __name__ == "__main__":
    main()