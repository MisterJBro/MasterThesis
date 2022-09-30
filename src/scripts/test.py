import torch
import numpy as np
import multiprocessing as mp
from src.env.hex import HexEnv

#print(f"CPU Cores: {mp.cpu_count()}")
#print(f"CUDA?: {torch.cuda.is_available()}")
#if torch.cuda.is_available():
#   print(f"GPU Name: {torch.cuda.get_device_name(0)}")

env = HexEnv()

print(env.step(env.available_actions()[0]))
print(env.step(env.available_actions()[5]))
print(env.step(env.available_actions()[1]))