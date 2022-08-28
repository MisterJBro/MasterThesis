import torch
import multiprocessing as mp

print(f"CPU Cores: {mp.cpu_count()}")
print(f"CUDA?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
   print(f"GPU Name: {torch.cuda.get_device_name(0)}")

#from hexgame import HexGame

#env = HexGame(19)
