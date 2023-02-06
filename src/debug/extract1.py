# Naive SP vs delta Uniform SP

import os
import torch
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
import numpy as np


if __name__ == '__main__':
    # Paths to dirs without ending /
    path1 = "C:/Users/jrb/Desktop/delta_uniform_sp_7x7_128_16_2"

    # Get files
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files1 = [f for f in files1 if f.startswith('p')]
    files1.sort(key=lambda f: int(f.split('_')[1]))
    print([float(f.split('_')[3].split('.pt')[0]) for f in files1])

    # Print number of files
    print(f"Found {len(files1)} files in {path1}")
